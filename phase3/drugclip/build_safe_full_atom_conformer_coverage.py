"""Build the CPU-only safe265 full-atom conformer prototype cache."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

from phase3.drugclip.train_only_torsion_prior_prototype import (
    GENERATOR_VERSION,
    MAX_SLOT_ATTEMPTS,
    PANEL_SEQUENCE_TIMEOUT_SECONDS,
    canonical_json_sha256,
    file_sha256,
    generate_train_only_faspr_conformers,
    load_torsion_prior,
)
from phase3.drugclip.validate_faspr_full_atom_prototype import verify_faspr_tool
from phase3.drugclip.validate_train_only_torsion_prototype import (
    validate_attempt_payload,
    validate_panel_payload,
)


SCHEMA_VERSION = "phase3-v2-safe265-full-atom-cache-v1"
DETERMINISTIC_MANIFEST_SCHEMA_VERSION = (
    "phase3-v2-safe265-deterministic-generation-manifest-v1"
)
CANONICAL_TOPOLOGY_CONTRACT = "standard-pdb-heavy-atom-bond-templates-v1"
EXPECTED_CANDIDATES = 370
EXPECTED_SAFE_SEQUENCES = 265
EXPECTED_REJECTED_SEQUENCES = 105
EXPECTED_SAFE_QUERIES = 373
EXPECTED_RECEPTOR_BANK = 512
EXPECTED_PRIOR_MANIFEST_SHA256 = (
    "E93B24E59D5C18D7CC4213BC82D38C789CB32A279A3078AED738477246E80F94"
)
EXPECTED_PRIOR_JSONL_SHA256 = (
    "BB86912B86388CB757467D610A3EA706BE03D69A98561FA362E95B71A5F7B57B"
)
CHEMISTRY_PRECEDENCE = [
    "modified_or_nonstandard",
    "receptor_covalent",
    "known_disulfide",
    "cyclic_or_crosslinked",
    "multiple_cys_unknown",
    "chemistry_insufficient",
    "ordinary_linear_standard",
]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    replace_existing: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace_existing:
        raise FileExistsError(f"atomic_target_already_exists:{path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"atomic_temporary_already_exists:{temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(
    path: Path,
    value: Any,
    *,
    replace_existing: bool = False,
) -> None:
    _atomic_write_bytes(
        path,
        (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8"),
        replace_existing=replace_existing,
    )


def atomic_write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    replace_existing: bool = False,
) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")
    _atomic_write_bytes(
        path,
        payload,
        replace_existing=replace_existing,
    )


def sequence_cache_key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest().upper()[:20]


def enrich_attempt_audit(payload: dict[str, Any]) -> None:
    """Bind every attempt row to the canonical-topology QC outcome."""
    accepted = {
        (int(row["conformer_index"]), int(row["attempt_index"])): row
        for row in payload["conformers"]
    }
    atom_count = int(payload["atom_count"])
    for attempt in payload["attempt_audit"]:
        attempt["atom_count"] = atom_count
        key = (
            int(attempt["logical_conformer_index"]),
            int(attempt["attempt_index"]),
        )
        conformer = accepted.get(key)
        if conformer is not None:
            attempt["canonical_topology_qc"] = dict(
                conformer["geometry_audit"]
            )
        else:
            attempt["canonical_topology_qc"] = {
                "status": "REJECT",
                "topology_contract": CANONICAL_TOPOLOGY_CONTRACT,
                "rejection_reason": attempt.get("rejection_reason"),
                "completed_pass_audit_available": False,
            }


def deterministic_sequence_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the elapsed/path-free generation identity for one sequence."""
    rejection_sequence = [
        {
            "logical_conformer_index": int(row["logical_conformer_index"]),
            "attempt_index": int(row["attempt_index"]),
            "rejection_reason": row.get("rejection_reason"),
        }
        for row in payload["attempt_audit"]
        if not bool(row["accepted"])
    ]
    core = {
        "peptide_sequence": payload["peptide_sequence"],
        "atom_count": int(payload["atom_count"]),
        "atom_identity_sha256": payload["atom_identity_sha256"],
        "accepted_attempt_indices": payload["accepted_attempt_indices"],
        "rejection_sequence": rejection_sequence,
        "rejection_log_semantic_sha256": payload[
            "rejection_log_semantic_sha256"
        ],
        "backbone_sha256": [
            row["train_only_backbone_audit"]["backbone_coordinate_sha256"]
            for row in payload["conformers"]
        ],
        "coordinate_sha256": [
            row["coordinate_sha256"] for row in payload["conformers"]
        ],
        "faspr_output_sha256": [
            row["faspr_output_sha256"] for row in payload["conformers"]
        ],
        "coordinate_set_sha256": payload[
            "canonical_coordinate_set_sha256"
        ],
    }
    return {
        **core,
        "per_sequence_aggregate_sha256": canonical_json_sha256(core),
    }


def deterministic_generation_manifest_core(
    records: list[dict[str, Any]],
    *,
    prior_manifest_sha256: str,
    prior_jsonl_sha256: str,
    faspr_source_commit: str,
    faspr_binary_sha256: str,
    faspr_rotamer_library_sha256: str,
    input_candidate_list_sha256: str,
    chemistry_audit_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": DETERMINISTIC_MANIFEST_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "canonical_topology_contract": CANONICAL_TOPOLOGY_CONTRACT,
        "prior_manifest_canonical_sha256": prior_manifest_sha256,
        "prior_jsonl_sha256": prior_jsonl_sha256,
        "faspr_source_commit": faspr_source_commit,
        "faspr_binary_sha256": faspr_binary_sha256,
        "faspr_rotamer_library_sha256": faspr_rotamer_library_sha256,
        "input_candidate_list_canonical_sha256": input_candidate_list_sha256,
        "chemistry_audit_sha256": chemistry_audit_sha256,
        "sequence_count": len(records),
        "conformer_count": sum(
            len(row["coordinate_sha256"]) for row in records
        ),
        "sequence_records": sorted(
            records, key=lambda row: row["peptide_sequence"]
        ),
    }


def derive_candidate_contract(
    chemistry_rows: list[dict[str, Any]],
    *,
    enforce_expected_counts: bool = True,
) -> dict[str, Any]:
    by_sequence: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chemistry_rows:
        sequence = str(row["peptide_sequence"]).upper()
        by_sequence[sequence].append(row)
    candidates = []
    for sequence in sorted(by_sequence):
        rows = by_sequence[sequence]
        classifications = {
            str(row["chemistry_classification"]) for row in rows
        }
        classification = next(
            name for name in CHEMISTRY_PRECEDENCE if name in classifications
        )
        heavy_counts = {
            int(row["theoretical_heavy_atom_count"]) for row in rows
        }
        if len(heavy_counts) != 1:
            raise ValueError(f"sequence_heavy_atom_count_inconsistent:{sequence}")
        candidates.append(
            {
                "peptide_sequence": sequence,
                "chemistry_classification": classification,
                "query_count": len(rows),
                "safe_query_count": sum(
                    row["chemistry_classification"]
                    == "ordinary_linear_standard"
                    and classification == "ordinary_linear_standard"
                    for row in rows
                ),
                "interface_pair_ids": sorted(
                    str(row["interface_pair_id"]) for row in rows
                ),
                "sequence_length": len(sequence),
                "theoretical_heavy_atom_count": next(iter(heavy_counts)),
                "exclusion_reasons": sorted(
                    {
                        str(row["exclusion_reason"])
                        for row in rows
                        if row.get("exclusion_reason")
                    }
                ),
            }
        )
    safe = [
        row for row in candidates
        if row["chemistry_classification"] == "ordinary_linear_standard"
    ]
    rejected = [
        row for row in candidates
        if row["chemistry_classification"] != "ordinary_linear_standard"
    ]
    safe_query_ids = sorted(
        pair_id
        for row in safe
        for pair_id in row["interface_pair_ids"]
    )
    contract = {
        "candidate_count": len(candidates),
        "safe_sequence_count": len(safe),
        "rejected_sequence_count": len(rejected),
        "safe_query_count": len(safe_query_ids),
        "safe_query_interface_pair_ids": safe_query_ids,
        "safe_query_targets_all_present": all(
            int(row["safe_query_count"]) == int(row["query_count"])
            for row in safe
        ),
        "classification_counts": dict(
            sorted(Counter(
                row["chemistry_classification"] for row in candidates
            ).items())
        ),
        "candidates": candidates,
        "safe_candidates": safe,
        "rejected_candidates": rejected,
    }
    if enforce_expected_counts and len(candidates) != EXPECTED_CANDIDATES:
        raise ValueError(
            f"candidate_count_mismatch:{len(candidates)}:{EXPECTED_CANDIDATES}"
        )
    if enforce_expected_counts and len(safe) != EXPECTED_SAFE_SEQUENCES:
        raise ValueError(
            f"safe_sequence_count_mismatch:{len(safe)}:"
            f"{EXPECTED_SAFE_SEQUENCES}"
        )
    if enforce_expected_counts and len(rejected) != EXPECTED_REJECTED_SEQUENCES:
        raise ValueError(
            f"rejected_sequence_count_mismatch:{len(rejected)}:"
            f"{EXPECTED_REJECTED_SEQUENCES}"
        )
    if enforce_expected_counts and len(safe_query_ids) != EXPECTED_SAFE_QUERIES:
        raise ValueError(
            f"safe_query_count_mismatch:{len(safe_query_ids)}:"
            f"{EXPECTED_SAFE_QUERIES}"
        )
    if not contract["safe_query_targets_all_present"]:
        raise ValueError("safe_query_target_not_in_safe_sequence_set")
    return contract


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile_of_empty_values")
    index = int(round((len(ordered) - 1) * fraction))
    return float(ordered[index])


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values),
        "median": _percentile(values, 0.5),
        "p90": _percentile(values, 0.9),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def verify_input_contract(
    *,
    chemistry_audit: Path,
    prior_manifest: Path,
    prior_jsonl: Path,
    faspr_executable: Path,
    faspr_rotamer_library: Path,
) -> tuple[
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
    dict[str, Any],
]:
    chemistry_rows = list(read_jsonl(chemistry_audit))
    candidate_contract = derive_candidate_contract(chemistry_rows)
    prior_groups, manifest = load_torsion_prior(prior_jsonl, prior_manifest)
    if manifest["manifest_canonical_sha256"] != EXPECTED_PRIOR_MANIFEST_SHA256:
        raise ValueError("prior_manifest_contract_mismatch")
    if file_sha256(prior_jsonl) != EXPECTED_PRIOR_JSONL_SHA256:
        raise ValueError("prior_jsonl_contract_mismatch")
    executable = Path(faspr_executable).resolve()
    library = Path(faspr_rotamer_library).resolve()
    if library != (executable.parent / "dun2010bbdep.bin").resolve():
        raise ValueError("explicit_rotamer_library_not_adjacent_to_faspr")
    tool = verify_faspr_tool(executable.parent)
    if Path(tool["binary_path"]).resolve() != executable:
        raise ValueError("explicit_faspr_executable_contract_mismatch")
    if Path(tool["rotamer_library_path"]).resolve() != library:
        raise ValueError("explicit_rotamer_library_contract_mismatch")
    return candidate_contract, prior_groups, manifest, tool


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    chemistry_audit = Path(args.chemistry_audit).resolve()
    prior_manifest = Path(args.prior_manifest).resolve()
    prior_jsonl = Path(args.prior_jsonl).resolve()
    faspr_executable = Path(args.faspr_executable).resolve()
    faspr_rotamer_library = Path(args.faspr_rotamer_library).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output_directory_already_exists:{output_dir}")
    output_dir.mkdir(parents=True)
    candidate_contract, prior_groups, manifest, tool = verify_input_contract(
        chemistry_audit=chemistry_audit,
        prior_manifest=prior_manifest,
        prior_jsonl=prior_jsonl,
        faspr_executable=faspr_executable,
        faspr_rotamer_library=faspr_rotamer_library,
    )
    candidate_records = [
        {
            "peptide_sequence": row["peptide_sequence"],
            "chemistry_classification": row["chemistry_classification"],
            "query_count": row["query_count"],
            "sequence_length": row["sequence_length"],
            "theoretical_heavy_atom_count": row[
                "theoretical_heavy_atom_count"
            ],
            "exclusion_reasons": row["exclusion_reasons"],
        }
        for row in candidate_contract["candidates"]
    ]
    input_candidate_sha = canonical_json_sha256(candidate_records)
    atomic_write_jsonl(
        output_dir / "candidate_contract.jsonl",
        candidate_records,
    )
    atomic_write_jsonl(
        output_dir / "rejected_candidates.jsonl",
        candidate_contract["rejected_candidates"],
    )
    progress_path = output_dir / "generation_progress.json"
    cache_entries = []
    started = time.perf_counter()
    for sequence_index, candidate in enumerate(
        candidate_contract["safe_candidates"]
    ):
        sequence = candidate["peptide_sequence"]
        key = sequence_cache_key(sequence)
        payload = generate_train_only_faspr_conformers(
            sequence,
            torsion_prior_groups=prior_groups,
            torsion_prior_manifest=manifest,
            work_dir=output_dir / "work" / key,
            faspr_executable=faspr_executable,
            faspr_commit_sha=tool["commit_sha"],
            faspr_binary_sha256=tool["binary_sha256"],
            attempt_qc=validate_attempt_payload,
        )
        enrich_attempt_audit(payload)
        validation = validate_panel_payload(payload)
        if int(payload["atom_count"]) >= 192:
            raise ValueError(
                f"safe265_atom_count_not_strictly_below_192:"
                f"{sequence}:{payload['atom_count']}"
            )
        cache_path = output_dir / "cache" / f"{key}.json"
        atomic_write_json(cache_path, payload)
        entry = {
            "sequence_index": sequence_index,
            "peptide_sequence": sequence,
            "chemistry_classification": "ordinary_linear_standard",
            "cache_path": str(cache_path),
            "cache_file_sha256": file_sha256(cache_path),
            "atom_count": payload["atom_count"],
            "atom_identity_sha256": payload["atom_identity_sha256"],
            "coordinate_set_sha256": payload[
                "canonical_coordinate_set_sha256"
            ],
            "accepted_attempt_indices": payload["accepted_attempt_indices"],
            "backbone_sha256": [
                row["train_only_backbone_audit"][
                    "backbone_coordinate_sha256"
                ]
                for row in payload["conformers"]
            ],
            "coordinate_sha256": [
                row["coordinate_sha256"] for row in payload["conformers"]
            ],
            "faspr_output_sha256": [
                row["faspr_output_sha256"] for row in payload["conformers"]
            ],
            "qc_status": validation["status"],
            "total_attempt_count": payload["total_attempt_count"],
            "rejection_count": payload["rejection_count"],
            "rejection_reason_counts": payload["rejection_reason_counts"],
            "maximum_attempt_index": payload["maximum_attempt_index"],
            "generation_seconds": payload["total_generation_seconds"],
        }
        cache_entries.append(entry)
        atomic_write_json(
            progress_path,
            {
                "status": "RUNNING",
                "completed_sequence_count": len(cache_entries),
                "expected_sequence_count": EXPECTED_SAFE_SEQUENCES,
                "latest_sequence": sequence,
                "elapsed_seconds": time.perf_counter() - started,
            },
            replace_existing=True,
        )
    atomic_write_jsonl(output_dir / "cache_index.jsonl", cache_entries)
    deterministic_records = [
        deterministic_sequence_record(
            json.loads(
                (output_dir / "cache" / f"{sequence_cache_key(row['peptide_sequence'])}.json")
                .read_text(encoding="utf-8")
            )
        )
        for row in cache_entries
    ]
    deterministic_core = deterministic_generation_manifest_core(
        deterministic_records,
        prior_manifest_sha256=manifest["manifest_canonical_sha256"],
        prior_jsonl_sha256=file_sha256(prior_jsonl),
        faspr_source_commit=tool["commit_sha"],
        faspr_binary_sha256=tool["binary_sha256"],
        faspr_rotamer_library_sha256=tool["rotamer_library_sha256"],
        input_candidate_list_sha256=input_candidate_sha,
        chemistry_audit_sha256=file_sha256(chemistry_audit),
    )
    deterministic_manifest = {
        **deterministic_core,
        "deterministic_manifest_sha256": canonical_json_sha256(
            deterministic_core
        ),
    }
    atomic_write_json(
        output_dir / "deterministic_generation_manifest.json",
        deterministic_manifest,
    )
    attempts = [row["total_attempt_count"] for row in cache_entries]
    timings = [row["generation_seconds"] for row in cache_entries]
    rejection_counts = Counter()
    for row in cache_entries:
        rejection_counts.update(row["rejection_reason_counts"])
    manifest_core = {
        "schema_version": SCHEMA_VERSION,
        "status": "GENERATION_PASS_PENDING_INDEPENDENT_VERIFICATION",
        "generator_version": GENERATOR_VERSION,
        "generation_seed_contract": [
            "generator_version",
            "torsion_prior_manifest_sha256",
            "peptide_sequence",
            "logical_conformer_index",
            "attempt_index",
        ],
        "query_receptor_or_evidence_in_generation_seed": False,
        "prior_manifest_canonical_sha256": manifest[
            "manifest_canonical_sha256"
        ],
        "prior_jsonl_sha256": file_sha256(prior_jsonl),
        "faspr_tool": {
            "source_commit": tool["commit_sha"],
            "binary_sha256": tool["binary_sha256"],
            "rotamer_library_sha256": tool["rotamer_library_sha256"],
        },
        "input_candidate_list_canonical_sha256": input_candidate_sha,
        "chemistry_audit": {
            "path": str(chemistry_audit),
            "file_sha256": file_sha256(chemistry_audit),
        },
        "candidate_counts": {
            "total": EXPECTED_CANDIDATES,
            "generated": len(cache_entries),
            "explicitly_rejected": len(
                candidate_contract["rejected_candidates"]
            ),
            "classification_counts": candidate_contract[
                "classification_counts"
            ],
        },
        "safe_query_target_coverage": {
            "covered_query_count": len(
                candidate_contract["safe_query_interface_pair_ids"]
            ),
            "expected_query_count": EXPECTED_SAFE_QUERIES,
            "all_targets_in_safe_sequence_set": candidate_contract[
                "safe_query_targets_all_present"
            ],
        },
        "generation_contract": {
            "conformers_per_sequence": 10,
            "maximum_attempts_per_slot": MAX_SLOT_ATTEMPTS,
            "sequence_timeout_seconds": PANEL_SEQUENCE_TIMEOUT_SECONDS,
            "nonlocal_heavy_atom_clash_threshold_angstrom": 0.75,
            "atom_count_must_be_strictly_less_than": 192,
            "canonical_topology_contract": CANONICAL_TOPOLOGY_CONTRACT,
        },
        "generation_counts": {
            "sequence_count": len(cache_entries),
            "conformer_count": sum(
                len(row["coordinate_sha256"]) for row in cache_entries
            ),
            "total_attempt_count": sum(attempts),
            "total_rejection_count": sum(
                row["rejection_count"] for row in cache_entries
            ),
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        },
        "attempt_distribution": _distribution(
            [float(value) for value in attempts]
        ),
        "performance_seconds_distribution": _distribution(timings),
        "cache_index": {
            "path": str(output_dir / "cache_index.jsonl"),
            "file_sha256": file_sha256(output_dir / "cache_index.jsonl"),
        },
        "deterministic_generation_manifest": {
            "path": str(
                output_dir / "deterministic_generation_manifest.json"
            ),
            "file_sha256": file_sha256(
                output_dir / "deterministic_generation_manifest.json"
            ),
            "semantic_sha256": deterministic_manifest[
                "deterministic_manifest_sha256"
            ],
        },
        "cache_entries": cache_entries,
        "candidate_bank_boundary": {
            "original_peptide_candidate_count": EXPECTED_CANDIDATES,
            "safe_peptide_candidate_count": EXPECTED_SAFE_SEQUENCES,
            "excluded_peptide_candidate_count": EXPECTED_REJECTED_SEQUENCES,
            "original_receptor_candidate_count": EXPECTED_RECEPTOR_BANK,
            "full_370_candidate_bank_has_full_atom_cache": False,
        },
        "target_bound_generation_inputs_used": False,
        "training_or_retrieval_run": False,
        "total_wall_seconds": time.perf_counter() - started,
    }
    final_manifest = {
        **manifest_core,
        "manifest_canonical_sha256": canonical_json_sha256(manifest_core),
    }
    atomic_write_json(output_dir / "cache_manifest.json", final_manifest)
    atomic_write_json(
        progress_path,
        {
            "status": "GENERATION_PASS_PENDING_INDEPENDENT_VERIFICATION",
            "completed_sequence_count": len(cache_entries),
            "expected_sequence_count": EXPECTED_SAFE_SEQUENCES,
            "elapsed_seconds": time.perf_counter() - started,
            "manifest_canonical_sha256": final_manifest[
                "manifest_canonical_sha256"
            ],
        },
        replace_existing=True,
    )
    return final_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemistry-audit", required=True)
    parser.add_argument("--prior-manifest", required=True)
    parser.add_argument("--prior-jsonl", required=True)
    parser.add_argument("--faspr-executable", required=True)
    parser.add_argument("--faspr-rotamer-library", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = build_cache(args)
    except Exception as error:
        output_dir = Path(args.output_dir).resolve()
        if output_dir.is_dir():
            failure = {
                "status": "FAIL",
                "classification": (
                    "PERFORMANCE_BLOCKED"
                    if isinstance(error, TimeoutError)
                    else "SAFE265_GENERATION_COVERAGE_FAIL"
                ),
                "exception_type": type(error).__name__,
                "exception_text": str(error),
            }
            details = getattr(error, "details", None)
            if details is not None:
                failure["details"] = details
            try:
                atomic_write_json(output_dir / "build_failure.json", failure)
            except FileExistsError:
                pass
            progress_path = output_dir / "generation_progress.json"
            completed = 0
            if progress_path.is_file():
                try:
                    completed = int(
                        json.loads(progress_path.read_text(encoding="utf-8"))[
                            "completed_sequence_count"
                        ]
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    completed = 0
            atomic_write_json(
                progress_path,
                {
                    "status": "FAIL",
                    "classification": failure["classification"],
                    "completed_sequence_count": completed,
                    "expected_sequence_count": EXPECTED_SAFE_SEQUENCES,
                    "exception_type": type(error).__name__,
                    "exception_text": str(error),
                },
                replace_existing=progress_path.exists(),
            )
        print(f"{type(error).__name__}:{error}")
        return 2
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "sequence_count": manifest["generation_counts"][
                    "sequence_count"
                ],
                "conformer_count": manifest["generation_counts"][
                    "conformer_count"
                ],
                "manifest_canonical_sha256": manifest[
                    "manifest_canonical_sha256"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

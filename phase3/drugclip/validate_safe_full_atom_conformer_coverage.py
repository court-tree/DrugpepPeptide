"""Independently regenerate and validate the safe265 prototype cache."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from phase3.drugclip.build_safe_full_atom_conformer_coverage import (
    CANONICAL_TOPOLOGY_CONTRACT,
    EXPECTED_CANDIDATES,
    EXPECTED_RECEPTOR_BANK,
    EXPECTED_REJECTED_SEQUENCES,
    EXPECTED_SAFE_QUERIES,
    EXPECTED_SAFE_SEQUENCES,
    atomic_write_json,
    canonical_json_sha256,
    deterministic_generation_manifest_core,
    deterministic_sequence_record,
    enrich_attempt_audit,
    file_sha256,
    read_jsonl,
    sequence_cache_key,
    verify_input_contract,
)
from phase3.drugclip.train_only_torsion_prior_prototype import (
    generate_train_only_faspr_conformers,
)
from phase3.drugclip.validate_train_only_torsion_prototype import (
    cpu_egnn_forward_all,
    validate_attempt_payload,
    validate_panel_payload,
)


SCHEMA_VERSION = "phase3-v2-safe265-full-atom-cache-validation-v1"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_canonical_manifest(manifest: dict[str, Any]) -> None:
    recorded = str(manifest.get("manifest_canonical_sha256") or "")
    core = dict(manifest)
    core.pop("manifest_canonical_sha256", None)
    if canonical_json_sha256(core) != recorded:
        raise ValueError("cache_manifest_canonical_sha256_mismatch")


def _validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    structural = validate_panel_payload(payload)
    if int(payload["atom_count"]) >= 192:
        raise ValueError(
            f"safe265_atom_count_not_strictly_below_192:"
            f"{payload['peptide_sequence']}:{payload['atom_count']}"
        )
    for conformer in payload["conformers"]:
        geometry = conformer["geometry_audit"]
        if geometry.get("status") != "PASS":
            raise ValueError("safe265_canonical_geometry_not_pass")
        if geometry.get("topology_contract") != CANONICAL_TOPOLOGY_CONTRACT:
            raise ValueError("safe265_canonical_topology_contract_mismatch")
        if geometry.get("coordinate_chirality_match") is not True:
            raise ValueError("safe265_canonical_chirality_not_pass")
        if geometry.get("chirality_audit", {}).get("status") != "PASS":
            raise ValueError("safe265_canonical_chirality_audit_not_pass")
        minimum_nonlocal = geometry.get(
            "minimum_nonlocal_heavy_atom_distance_angstrom"
        )
        if minimum_nonlocal is not None and float(minimum_nonlocal) < 0.75:
            raise ValueError("safe265_nonlocal_clash_threshold_violation")
    egnn = cpu_egnn_forward_all(payload)
    if egnn.get("status") != "PASS" or not egnn.get("embedding_finite"):
        raise ValueError("safe265_cpu_egnn_forward_not_finite")
    return {"structural": structural, "cpu_egnn_forward": egnn}


def _determinism_comparison(
    cached: dict[str, Any],
    regenerated: dict[str, Any],
) -> dict[str, Any]:
    scalar_fields = (
        "peptide_sequence",
        "atom_count",
        "atom_identity_sha256",
        "canonical_coordinate_set_sha256",
        "accepted_attempt_indices",
        "rejection_log_semantic_sha256",
    )
    for field in scalar_fields:
        if regenerated[field] != cached[field]:
            raise ValueError(
                f"determinism_field_mismatch:{cached['peptide_sequence']}:{field}"
            )
    conformer_fields = (
        "coordinate_sha256",
        "faspr_output_sha256",
        "attempt_index",
        "seed",
    )
    matched = 0
    for cached_row, regenerated_row in zip(
        cached["conformers"], regenerated["conformers"], strict=True
    ):
        for field in conformer_fields:
            if regenerated_row[field] != cached_row[field]:
                raise ValueError(
                    f"determinism_conformer_field_mismatch:"
                    f"{cached['peptide_sequence']}:{cached_row['conformer_index']}:"
                    f"{field}"
                )
        cached_backbone = cached_row["train_only_backbone_audit"][
            "backbone_coordinate_sha256"
        ]
        regenerated_backbone = regenerated_row["train_only_backbone_audit"][
            "backbone_coordinate_sha256"
        ]
        if regenerated_backbone != cached_backbone:
            raise ValueError(
                f"determinism_backbone_hash_mismatch:"
                f"{cached['peptide_sequence']}:{cached_row['conformer_index']}"
            )
        matched += 1
    return {
        "matched_conformer_count": matched,
        "accepted_attempt_indices": cached["accepted_attempt_indices"],
        "coordinate_set_sha256": cached["canonical_coordinate_set_sha256"],
        "atom_identity_sha256": cached["atom_identity_sha256"],
    }


def validate_cache(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"coverage_output_directory_missing:{output_dir}")
    chemistry_audit = Path(args.chemistry_audit).resolve()
    prior_manifest = Path(args.prior_manifest).resolve()
    prior_jsonl = Path(args.prior_jsonl).resolve()
    faspr_executable = Path(args.faspr_executable).resolve()
    faspr_rotamer_library = Path(args.faspr_rotamer_library).resolve()
    candidate_contract, prior_groups, torsion_manifest, tool = (
        verify_input_contract(
            chemistry_audit=chemistry_audit,
            prior_manifest=prior_manifest,
            prior_jsonl=prior_jsonl,
            faspr_executable=faspr_executable,
            faspr_rotamer_library=faspr_rotamer_library,
        )
    )
    manifest = _load_json(output_dir / "cache_manifest.json")
    _verify_canonical_manifest(manifest)
    if manifest["status"] != "GENERATION_PASS_PENDING_INDEPENDENT_VERIFICATION":
        raise ValueError("cache_manifest_generation_status_mismatch")
    if manifest["chemistry_audit"]["file_sha256"] != file_sha256(
        chemistry_audit
    ):
        raise ValueError("cache_manifest_chemistry_audit_sha256_mismatch")
    deterministic_manifest = _load_json(
        output_dir / "deterministic_generation_manifest.json"
    )
    recorded_deterministic_sha = deterministic_manifest.pop(
        "deterministic_manifest_sha256", ""
    )
    if canonical_json_sha256(deterministic_manifest) != (
        recorded_deterministic_sha
    ):
        raise ValueError("deterministic_generation_manifest_sha256_mismatch")
    if manifest["deterministic_generation_manifest"]["semantic_sha256"] != (
        recorded_deterministic_sha
    ):
        raise ValueError("cache_deterministic_manifest_binding_mismatch")
    cache_entries = list(read_jsonl(output_dir / "cache_index.jsonl"))
    if len(cache_entries) != EXPECTED_SAFE_SEQUENCES:
        raise ValueError("cache_index_sequence_count_mismatch")
    entries_by_sequence = {
        str(row["peptide_sequence"]): row for row in cache_entries
    }
    expected_sequences = [
        str(row["peptide_sequence"])
        for row in candidate_contract["safe_candidates"]
    ]
    if sorted(entries_by_sequence) != sorted(expected_sequences):
        raise ValueError("cache_index_safe_sequence_set_mismatch")

    progress_path = output_dir / "validation_progress.json"
    verification_rows = []
    regenerated_deterministic_records = []
    regenerated_seconds = []
    total_matched = 0
    started = time.perf_counter()
    for sequence_index, sequence in enumerate(expected_sequences):
        entry = entries_by_sequence[sequence]
        expected_cache_path = (
            output_dir / "cache" / f"{sequence_cache_key(sequence)}.json"
        )
        cache_path = Path(entry["cache_path"]).resolve()
        if cache_path != expected_cache_path.resolve():
            raise ValueError(f"cache_path_contract_mismatch:{sequence}")
        if file_sha256(cache_path) != entry["cache_file_sha256"]:
            raise ValueError(f"cache_file_sha256_mismatch:{sequence}")
        cached = _load_json(cache_path)
        cached_validation = _validate_payload(cached)
        sequence_started = time.perf_counter()
        with tempfile.TemporaryDirectory(
            prefix=f".verify_{sequence_cache_key(sequence)}_",
            dir=output_dir,
        ) as temporary:
            regenerated = generate_train_only_faspr_conformers(
                sequence,
                torsion_prior_groups=prior_groups,
                torsion_prior_manifest=torsion_manifest,
                work_dir=Path(temporary),
                faspr_executable=faspr_executable,
                faspr_commit_sha=tool["commit_sha"],
                faspr_binary_sha256=tool["binary_sha256"],
                attempt_qc=validate_attempt_payload,
            )
            enrich_attempt_audit(regenerated)
            regenerated_validation = _validate_payload(regenerated)
            comparison = _determinism_comparison(cached, regenerated)
            regenerated_deterministic_records.append(
                deterministic_sequence_record(regenerated)
            )
        elapsed = time.perf_counter() - sequence_started
        regenerated_seconds.append(elapsed)
        total_matched += int(comparison["matched_conformer_count"])
        verification_rows.append(
            {
                "sequence_index": sequence_index,
                "peptide_sequence": sequence,
                "status": "PASS",
                "cached_qc_status": cached_validation["structural"]["status"],
                "regenerated_qc_status": regenerated_validation["structural"][
                    "status"
                ],
                "cpu_egnn_forward_finite": True,
                "matched_conformer_count": comparison[
                    "matched_conformer_count"
                ],
                "accepted_attempt_indices": comparison[
                    "accepted_attempt_indices"
                ],
                "coordinate_set_sha256": comparison[
                    "coordinate_set_sha256"
                ],
                "atom_identity_sha256": comparison["atom_identity_sha256"],
                "regeneration_seconds": elapsed,
            }
        )
        atomic_write_json(
            progress_path,
            {
                "status": "RUNNING",
                "completed_sequence_count": len(verification_rows),
                "expected_sequence_count": EXPECTED_SAFE_SEQUENCES,
                "matched_conformer_count": total_matched,
                "latest_sequence": sequence,
                "elapsed_seconds": time.perf_counter() - started,
            },
            replace_existing=True,
        )
    if total_matched != EXPECTED_SAFE_SEQUENCES * 10:
        raise ValueError("deterministic_conformer_match_count_mismatch")
    regenerated_deterministic_core = deterministic_generation_manifest_core(
        regenerated_deterministic_records,
        prior_manifest_sha256=torsion_manifest[
            "manifest_canonical_sha256"
        ],
        prior_jsonl_sha256=file_sha256(prior_jsonl),
        faspr_source_commit=tool["commit_sha"],
        faspr_binary_sha256=tool["binary_sha256"],
        faspr_rotamer_library_sha256=tool["rotamer_library_sha256"],
        input_candidate_list_sha256=manifest[
            "input_candidate_list_canonical_sha256"
        ],
        chemistry_audit_sha256=file_sha256(chemistry_audit),
    )
    regenerated_deterministic_sha = canonical_json_sha256(
        regenerated_deterministic_core
    )
    if regenerated_deterministic_sha != recorded_deterministic_sha:
        raise ValueError("deterministic_global_manifest_hash_mismatch")
    rejected_counts = Counter(
        row["chemistry_classification"]
        for row in candidate_contract["rejected_candidates"]
    )
    result_core = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "classification": "SAFE265_FULL_ATOM_CACHE_PASS",
        "cache_manifest_canonical_sha256": manifest[
            "manifest_canonical_sha256"
        ],
        "chemistry_audit_sha256": file_sha256(chemistry_audit),
        "candidate_counts": {
            "total": EXPECTED_CANDIDATES,
            "generated": EXPECTED_SAFE_SEQUENCES,
            "explicitly_rejected": EXPECTED_REJECTED_SEQUENCES,
            "rejected_reason_counts": dict(sorted(rejected_counts.items())),
        },
        "safe_query_target_coverage": {
            "covered": EXPECTED_SAFE_QUERIES,
            "expected": EXPECTED_SAFE_QUERIES,
            "all_targets_in_safe265": True,
        },
        "determinism": {
            "matched_sequence_count": len(verification_rows),
            "matched_conformer_count": total_matched,
            "expected_conformer_count": EXPECTED_SAFE_SEQUENCES * 10,
            "generation_manifest_sha256": recorded_deterministic_sha,
            "regenerated_manifest_sha256": regenerated_deterministic_sha,
            "global_manifest_hash_match": True,
        },
        "performance": {
            "per_sequence_limit_seconds": 300,
            "maximum_regeneration_seconds": max(regenerated_seconds),
            "all_within_limit": all(value <= 300 for value in regenerated_seconds),
            "total_regeneration_seconds": sum(regenerated_seconds),
        },
        "candidate_bank_boundary": {
            "r2p_original_candidate_count": EXPECTED_CANDIDATES,
            "r2p_safe_candidate_count": EXPECTED_SAFE_SEQUENCES,
            "r2p_candidate_bank_changed": True,
            "r2p_original_370_metrics_directly_comparable": False,
            "r2p_baselines_must_be_recomputed_on_exact_safe265_bank": True,
            "p2r_receptor_bank_may_remain_at": EXPECTED_RECEPTOR_BANK,
            "p2r_baselines_must_be_recomputed_for_exact_safe_query_subset": True,
        },
        "target_bound_generation_inputs_used": False,
        "training_or_gpu_retrieval_run": False,
        "verification_rows": verification_rows,
        "total_wall_seconds": time.perf_counter() - started,
    }
    result = {
        **result_core,
        "validation_manifest_canonical_sha256": canonical_json_sha256(
            result_core
        ),
    }
    atomic_write_json(output_dir / "validation_report.json", result)
    atomic_write_json(
        progress_path,
        {
            "status": "PASS",
            "classification": result["classification"],
            "completed_sequence_count": len(verification_rows),
            "matched_conformer_count": total_matched,
            "validation_manifest_canonical_sha256": result[
                "validation_manifest_canonical_sha256"
            ],
        },
        replace_existing=True,
    )
    return result


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
        result = validate_cache(args)
    except Exception as error:
        output_dir = Path(args.output_dir).resolve()
        classification = "PERFORMANCE_BLOCKED" if isinstance(
            error, TimeoutError
        ) else (
            "DETERMINISM_FAIL"
            if "determin" in str(error).lower()
            else "MODEL_INPUT_FAIL"
            if "egnn" in str(error).lower() or "tensor" in str(error).lower()
            else "MANIFEST_CONTRACT_FAIL"
            if "manifest" in str(error).lower() or "sha256" in str(error).lower()
            else "CHEMISTRY_TARGET_COVERAGE_FAIL"
            if "target" in str(error).lower() or "candidate" in str(error).lower()
            else "SAFE265_GENERATION_COVERAGE_FAIL"
        )
        if output_dir.is_dir():
            failure = {
                "status": "FAIL",
                "classification": classification,
                "exception_type": type(error).__name__,
                "exception_text": str(error),
            }
            details = getattr(error, "details", None)
            if details is not None:
                failure["details"] = details
            try:
                atomic_write_json(
                    output_dir / "validation_failure.json", failure
                )
            except FileExistsError:
                pass
        print(f"{type(error).__name__}:{error}")
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "classification": result["classification"],
                "matched_conformer_count": result["determinism"][
                    "matched_conformer_count"
                ],
                "validation_manifest_canonical_sha256": result[
                    "validation_manifest_canonical_sha256"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

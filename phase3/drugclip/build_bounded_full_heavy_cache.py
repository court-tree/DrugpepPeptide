"""Resumable materializer for the frozen Phase-3 v2 bounded full-heavy plan.

The module deliberately separates a formal 2,085-sequence materialization
from small smoke runs.  A formal ``cache_manifest.json`` is written only after
every descriptor sequence has a validated, atomically-written sequence file.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable

from phase3.drugclip.full_heavy_adaptation_contract import (
    ATOM_CAP_EXCLUSIVE,
    CACHE_MANIFEST_SCHEMA,
    CANONICAL_TOPOLOGY_CONTRACT,
    CONFORMERS_PER_SEQUENCE,
    TORSION_PRIOR_JSONL_SHA256,
    TORSION_PRIOR_MANIFEST_SHA256,
    canonical_json_sha256,
    sequence_sha256,
    sha256_file,
)
from phase3.drugclip.train_only_torsion_prior_prototype import (
    ConformerCoverageError,
    GENERATOR_VERSION,
    MAX_SLOT_ATTEMPTS,
    PANEL_SEQUENCE_TIMEOUT_SECONDS,
    generate_train_only_faspr_conformers,
    load_torsion_prior,
)
from phase3.drugclip.validate_faspr_full_atom_prototype import (
    EXPECTED_FASPR_BINARY_SHA256,
    EXPECTED_FASPR_COMMIT,
    verify_faspr_tool,
)
from phase3.drugclip.validate_train_only_torsion_prototype import (
    cpu_egnn_forward_all,
    validate_attempt_payload,
    validate_panel_payload,
)


CACHE_CONTRACT_SCHEMA = "phase3-v2-bounded-full-heavy-cache-contract-v1"
SEQUENCE_FILE_SCHEMA = "phase3-v2-bounded-full-heavy-sequence-cache-v1"
SMOKE_MANIFEST_SCHEMA = "phase3-v2-bounded-full-heavy-cache-smoke-v1"
PROGRESS_SCHEMA = "phase3-v2-bounded-full-heavy-cache-progress-v1"
FAILURE_SCHEMA = "phase3-v2-bounded-full-heavy-cache-failure-v1"
FORMAL_DESCRIPTOR_FILE_SHA256 = (
    "1894F635E352D127AC79DF226E4F50A7451B8E47C43D6388239A23752721957D"
)
FORMAL_DESCRIPTOR_CANONICAL_SHA256 = (
    "2F8FF55185DE5E87861687CA564EC4851E186C16C4C8158B9C1168D8E32D8DE0"
)
EXPECTED_FORMAL_SEQUENCE_COUNT = 2085
EXPECTED_FORMAL_CONFORMER_COUNT = 20850
FASPR_ROTAMER_LIBRARY_SHA256 = (
    "ED3F7BE5F33B5FA947AC5E83CB024C6A6AF6440BB50A1C8073AACABE6D792D0E"
)
NONLOCAL_CLASH_THRESHOLD_ANGSTROM = 0.75
ALLOWED_GENERATION_INPUTS = (
    "generator_version",
    "torsion_prior_manifest_sha256",
    "faspr_source_commit",
    "faspr_binary_sha256",
    "faspr_rotamer_library_sha256",
    "peptide_sequence",
    "conformer_index",
    "attempt_index",
)
FORBIDDEN_GENERATION_INPUTS = (
    "receptor",
    "interface_pair",
    "biological_relation",
    "contact",
    "evidence",
    "bound_coordinates",
    "bound_pose",
)

SequenceGenerator = Callable[[str, Path], dict[str, Any]]
PayloadValidator = Callable[[dict[str, Any]], dict[str, Any]]


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sequence_file_key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest().upper()


def coordinate_sha256(coordinates: list[list[float]]) -> str:
    canonical = ";".join(
        ",".join(format(float(value), ".12f") for value in xyz)
        for xyz in coordinates
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest().upper()


def atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    replace_existing: bool = False,
) -> None:
    """Flush and atomically rename without treating a stale temp as complete."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale_atomic_temporary_file:{temporary}")
    if path.exists() and not replace_existing:
        raise FileExistsError(f"atomic_target_already_exists:{path}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(
    path: Path,
    value: Any,
    *,
    replace_existing: bool = False,
) -> None:
    atomic_write_bytes(
        path,
        canonical_json_bytes(value),
        replace_existing=replace_existing,
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_record(value: dict[str, Any], hash_field: str) -> dict[str, Any]:
    core = {key: item for key, item in value.items() if key != hash_field}
    return {**core, hash_field: canonical_json_sha256(core)}


def _verify_canonical_record(
    value: dict[str, Any],
    hash_field: str,
    error: str,
) -> None:
    recorded = str(value.get(hash_field) or "")
    core = {key: item for key, item in value.items() if key != hash_field}
    if canonical_json_sha256(core) != recorded:
        raise ValueError(error)


def load_descriptor_contract(
    descriptor_path: Path,
    *,
    expected_file_sha256: str = FORMAL_DESCRIPTOR_FILE_SHA256,
    expected_canonical_sha256: str = FORMAL_DESCRIPTOR_CANONICAL_SHA256,
    enforce_formal_counts: bool = True,
) -> dict[str, Any]:
    descriptor_path = Path(descriptor_path).resolve()
    actual_file_sha = sha256_file(descriptor_path)
    if actual_file_sha != expected_file_sha256:
        raise ValueError(
            f"plan_descriptor_file_sha256_mismatch:{actual_file_sha}:"
            f"{expected_file_sha256}"
        )
    descriptor = _load_json(descriptor_path)
    if descriptor.get("schema_version") != (
        "phase3-v2-bounded-full-heavy-plan-v1"
    ):
        raise ValueError("unsupported_bounded_plan_descriptor")
    recorded = str(descriptor.get("descriptor_canonical_sha256") or "")
    core = {
        key: value
        for key, value in descriptor.items()
        if key != "descriptor_canonical_sha256"
    }
    if (
        recorded != expected_canonical_sha256
        or canonical_json_sha256(core) != recorded
    ):
        raise ValueError("plan_descriptor_canonical_sha256_mismatch")
    requirement = descriptor.get("future_cache_requirement", {})
    sequences = [str(value) for value in requirement.get(
        "required_peptide_sequences", []
    )]
    if sequences != sorted(set(sequences)):
        raise ValueError("descriptor_required_sequence_order_or_duplicate")
    if requirement.get("required_peptide_sequences_sha256") != (
        sequence_sha256(sequences)
    ):
        raise ValueError("descriptor_required_sequence_sha256_mismatch")
    if (
        int(requirement.get("conformers_per_sequence", -1))
        != CONFORMERS_PER_SEQUENCE
        or int(requirement.get("future_required_conformer_count", -1))
        != len(sequences) * CONFORMERS_PER_SEQUENCE
        or requirement.get("generation_status") != "NOT_BUILT"
        or requirement.get("cache_status") != "NOT_BUILT"
    ):
        raise ValueError("descriptor_future_cache_contract_mismatch")
    if enforce_formal_counts and (
        len(sequences) != EXPECTED_FORMAL_SEQUENCE_COUNT
        or len(sequences) * CONFORMERS_PER_SEQUENCE
        != EXPECTED_FORMAL_CONFORMER_COUNT
    ):
        raise ValueError("descriptor_formal_count_mismatch")
    return {
        "path": descriptor_path,
        "file_sha256": actual_file_sha,
        "canonical_sha256": recorded,
        "required_sequences": sequences,
        "descriptor": descriptor,
    }


def _plan_sequence_records(
    descriptor_contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    descriptor = descriptor_contract["descriptor"]
    descriptor_path = Path(descriptor_contract["path"])
    output: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid"):
        contract = descriptor["plan_files"][split]
        plan_path = (descriptor_path.parent / contract["path"]).resolve()
        if sha256_file(plan_path) != contract["file_sha256"]:
            raise ValueError(f"{split}_plan_file_sha256_mismatch")
        plan = _load_json(plan_path)
        for row in plan["sequence_records"]:
            sequence = str(row["peptide_sequence"])
            atom_count = int(row["theoretical_heavy_atom_count"])
            if atom_count >= ATOM_CAP_EXCLUSIVE:
                raise ValueError(f"plan_sequence_atom_cap_violation:{sequence}")
            record = output.setdefault(
                sequence,
                {
                    "peptide_sequence": sequence,
                    "theoretical_heavy_atom_count": atom_count,
                    "split_roles": [],
                },
            )
            if record["theoretical_heavy_atom_count"] != atom_count:
                raise ValueError(f"plan_sequence_atom_count_mismatch:{sequence}")
            record["split_roles"].append(split)
    if set(output) != set(descriptor_contract["required_sequences"]):
        raise ValueError("plan_sequence_record_set_mismatch")
    for record in output.values():
        record["split_roles"].sort()
    return output


def select_smoke_sequences(
    descriptor_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Mechanically select five distinct descriptor sequences."""

    records = list(_plan_sequence_records(descriptor_contract).values())
    by_length = sorted(
        records,
        key=lambda row: (
            len(row["peptide_sequence"]),
            row["theoretical_heavy_atom_count"],
            row["peptide_sequence"],
        ),
    )
    by_size = sorted(
        records,
        key=lambda row: (
            -row["theoretical_heavy_atom_count"],
            -len(row["peptide_sequence"]),
            row["peptide_sequence"],
        ),
    )
    by_key = sorted(
        records,
        key=lambda row: (
            hashlib.sha256(
                (
                    "phase3-v2-bounded-full-heavy-plan-v1"
                    + "\0smoke\0"
                    + row["peptide_sequence"]
                ).encode("utf-8")
            ).hexdigest(),
            row["peptide_sequence"],
        ),
    )
    candidates = [
        ("shortest_length", by_length[0]),
        ("maximum_heavy_atom_count", by_size[0]),
        (
            "contains_proline",
            next(row for row in by_key if "P" in row["peptide_sequence"]),
        ),
        (
            "contains_glycine",
            next(row for row in by_key if "G" in row["peptide_sequence"]),
        ),
        ("minimum_deterministic_smoke_key", by_key[0]),
    ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reason, row in candidates:
        sequence = row["peptide_sequence"]
        if sequence not in seen:
            selected.append({**row, "selection_reason": [reason]})
            seen.add(sequence)
        else:
            existing = next(
                value for value in selected
                if value["peptide_sequence"] == sequence
            )
            existing["selection_reason"].append(reason)
    for row in by_key:
        if len(selected) == 5:
            break
        if row["peptide_sequence"] not in seen:
            selected.append({**row, "selection_reason": ["key_fill"]})
            seen.add(row["peptide_sequence"])
    if len(selected) != 5:
        raise ValueError("smoke_sequence_selection_count_mismatch")
    return selected


def verify_tool_and_prior(
    *,
    prior_manifest_path: Path,
    prior_jsonl_path: Path,
    faspr_executable: Path,
    faspr_rotamer_library: Path,
) -> dict[str, Any]:
    prior_groups, prior_manifest = load_torsion_prior(
        prior_jsonl_path, prior_manifest_path
    )
    if (
        prior_manifest["manifest_canonical_sha256"]
        != TORSION_PRIOR_MANIFEST_SHA256
        or sha256_file(prior_jsonl_path) != TORSION_PRIOR_JSONL_SHA256
    ):
        raise ValueError("frozen_torsion_prior_sha256_mismatch")
    faspr_executable = Path(faspr_executable).resolve()
    faspr_rotamer_library = Path(faspr_rotamer_library).resolve()
    if faspr_executable.parent != faspr_rotamer_library.parent:
        raise ValueError("faspr_binary_library_directory_mismatch")
    if faspr_executable.name != "FASPR":
        raise ValueError("faspr_executable_name_mismatch")
    if faspr_rotamer_library.name != "dun2010bbdep.bin":
        raise ValueError("faspr_rotamer_library_name_mismatch")
    tool = verify_faspr_tool(
        faspr_executable.parent,
        expected_commit=EXPECTED_FASPR_COMMIT,
        expected_binary_sha256=EXPECTED_FASPR_BINARY_SHA256,
    )
    if (
        tool["rotamer_library_sha256"]
        != FASPR_ROTAMER_LIBRARY_SHA256
    ):
        raise ValueError("faspr_rotamer_library_sha256_mismatch")
    return {
        "prior_groups": prior_groups,
        "prior_manifest": prior_manifest,
        "tool": tool,
        "prior_manifest_file_sha256": sha256_file(prior_manifest_path),
        "prior_jsonl_file_sha256": sha256_file(prior_jsonl_path),
    }


def build_cache_contract(
    *,
    descriptor_contract: dict[str, Any],
    required_sequences: list[str],
    formal_cache: bool,
    prior_manifest_file_sha256: str,
    prior_jsonl_file_sha256: str,
    tool: dict[str, Any],
) -> dict[str, Any]:
    core = {
        "schema_version": CACHE_CONTRACT_SCHEMA,
        "formal_cache": bool(formal_cache),
        "purpose": (
            "bounded_train_valid_only" if formal_cache else "five_sequence_smoke"
        ),
        "not_valid_for_training": not formal_cache,
        "plan_descriptor_file_sha256": descriptor_contract["file_sha256"],
        "plan_descriptor_canonical_sha256": descriptor_contract[
            "canonical_sha256"
        ],
        "required_peptide_sequences": required_sequences,
        "required_peptide_sequences_sha256": sequence_sha256(
            required_sequences
        ),
        "sequence_count": len(required_sequences),
        "conformers_per_sequence": CONFORMERS_PER_SEQUENCE,
        "required_conformer_count": (
            len(required_sequences) * CONFORMERS_PER_SEQUENCE
        ),
        "generator_version": GENERATOR_VERSION,
        "torsion_prior_manifest_canonical_sha256": (
            TORSION_PRIOR_MANIFEST_SHA256
        ),
        "torsion_prior_manifest_file_sha256": prior_manifest_file_sha256,
        "torsion_prior_jsonl_sha256": prior_jsonl_file_sha256,
        "faspr_source_commit": tool["commit_sha"],
        "faspr_binary_sha256": tool["binary_sha256"],
        "faspr_rotamer_library_sha256": tool[
            "rotamer_library_sha256"
        ],
        "canonical_topology_contract": CANONICAL_TOPOLOGY_CONTRACT,
        "maximum_attempts_per_logical_conformer": MAX_SLOT_ATTEMPTS,
        "sequence_timeout_seconds": PANEL_SEQUENCE_TIMEOUT_SECONDS,
        "nonlocal_heavy_atom_clash_threshold_angstrom": (
            NONLOCAL_CLASH_THRESHOLD_ANGSTROM
        ),
        "atom_cap_exclusive": ATOM_CAP_EXCLUSIVE,
        "generation_input_contract": {
            "allowed_fields": list(ALLOWED_GENERATION_INPUTS),
            "forbidden_fields": list(FORBIDDEN_GENERATION_INPUTS),
            "target_bound_generation_inputs_used": False,
        },
    }
    return _canonical_record(core, "contract_canonical_sha256")


def _check_no_stale_temporary_files(cache_root: Path) -> None:
    stale = sorted(
        path.relative_to(cache_root).as_posix()
        for path in cache_root.rglob("*.tmp")
    )
    if stale:
        raise RuntimeError(f"stale_temporary_files_present:{stale}")


def _validate_generic_payload(payload: dict[str, Any], sequence: str) -> None:
    if payload.get("peptide_sequence") != sequence:
        raise ValueError(f"sequence_payload_sequence_mismatch:{sequence}")
    if payload.get("sequence_sha256") != sequence_file_key(sequence):
        raise ValueError(f"sequence_payload_sequence_sha256_mismatch:{sequence}")
    _verify_canonical_record(
        payload,
        "payload_canonical_sha256",
        f"sequence_payload_canonical_sha256_mismatch:{sequence}",
    )
    identities = payload.get("atom_identity")
    conformers = payload.get("conformers")
    if (
        not isinstance(identities, list)
        or payload.get("atom_identity_sha256")
        != canonical_json_sha256(identities)
    ):
        raise ValueError(f"sequence_atom_identity_sha256_mismatch:{sequence}")
    if not isinstance(conformers, list) or len(conformers) != 10:
        raise ValueError(f"sequence_conformer_count_mismatch:{sequence}")
    if [int(row.get("conformer_index", -1)) for row in conformers] != list(
        range(10)
    ):
        raise ValueError(f"sequence_conformer_index_mismatch:{sequence}")
    hashes: list[str] = []
    for conformer in conformers:
        coordinates = conformer.get("coordinates")
        coordinate_sha = coordinate_sha256(coordinates)
        if conformer.get("coordinate_sha256") != coordinate_sha:
            raise ValueError(f"sequence_coordinate_sha256_mismatch:{sequence}")
        hashes.append(coordinate_sha)
    if len(set(hashes)) != 10:
        raise ValueError(f"sequence_coordinate_hash_duplicate:{sequence}")


def production_payload_validator(payload: dict[str, Any]) -> dict[str, Any]:
    sequence = str(payload["peptide_sequence"])
    _validate_generic_payload(payload, sequence)
    panel = validate_panel_payload(payload)
    egnn = cpu_egnn_forward_all(payload)
    if egnn.get("status") != "PASS" or not egnn.get("embedding_finite"):
        raise ValueError(f"sequence_cpu_egnn_forward_nonfinite:{sequence}")
    if int(egnn.get("tensorization_unk_count", -1)) != 0:
        raise ValueError(f"sequence_cpu_egnn_forward_unk:{sequence}")
    return {"status": "PASS", "structural_qc": panel, "cpu_egnn": egnn}


def _bind_sequence_payload(
    generated: dict[str, Any],
    *,
    sequence: str,
    cache_contract: dict[str, Any],
) -> dict[str, Any]:
    if generated.get("peptide_sequence") != sequence:
        raise ValueError("generator_sequence_mismatch")
    forbidden_present = sorted(
        field for field in FORBIDDEN_GENERATION_INPUTS if field in generated
    )
    if (
        forbidden_present
        or generated.get("target_bound_generation_inputs_used") is True
    ):
        raise ValueError(
            f"forbidden_generation_input:{forbidden_present}"
        )
    core = {
        **generated,
        "sequence_file_schema_version": SEQUENCE_FILE_SCHEMA,
        "sequence_sha256": sequence_file_key(sequence),
        "chemistry_classification": "ordinary_linear_standard",
        "cache_contract_canonical_sha256": cache_contract[
            "contract_canonical_sha256"
        ],
        "plan_descriptor_file_sha256": cache_contract[
            "plan_descriptor_file_sha256"
        ],
        "plan_descriptor_canonical_sha256": cache_contract[
            "plan_descriptor_canonical_sha256"
        ],
        "target_bound_generation_inputs_used": False,
    }
    return _canonical_record(core, "payload_canonical_sha256")


def validate_existing_sequence_file(
    path: Path,
    *,
    sequence: str,
    cache_contract: dict[str, Any],
    payload_validator: PayloadValidator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_path = path.parent / f"{sequence_file_key(sequence)}.json"
    if path.resolve() != expected_path.resolve():
        raise ValueError(f"sequence_cache_path_mismatch:{sequence}")
    payload = _load_json(path)
    _validate_generic_payload(payload, sequence)
    bindings = {
        "cache_contract_canonical_sha256": cache_contract[
            "contract_canonical_sha256"
        ],
        "plan_descriptor_file_sha256": cache_contract[
            "plan_descriptor_file_sha256"
        ],
        "plan_descriptor_canonical_sha256": cache_contract[
            "plan_descriptor_canonical_sha256"
        ],
    }
    for field, expected in bindings.items():
        if payload.get(field) != expected:
            raise ValueError(f"sequence_cache_contract_binding_mismatch:{field}")
    if payload.get("target_bound_generation_inputs_used") is not False:
        raise ValueError("sequence_target_bound_generation_input_violation")
    audit = payload_validator(payload)
    return payload, audit


def _sequence_index_row(
    *,
    sequence: str,
    path: Path,
    payload: dict[str, Any],
    record: dict[str, Any],
    cache_root: Path,
) -> dict[str, Any]:
    return {
        "peptide_sequence": sequence,
        "sequence_sha256": sequence_file_key(sequence),
        "chemistry_classification": "ordinary_linear_standard",
        "split_roles": record["split_roles"],
        "theoretical_heavy_atom_count": record[
            "theoretical_heavy_atom_count"
        ],
        "atom_count": int(payload["atom_count"]),
        "atom_identity_sha256": payload["atom_identity_sha256"],
        "accepted_attempt_indices": payload["accepted_attempt_indices"],
        "coordinate_sha256": [
            row["coordinate_sha256"] for row in payload["conformers"]
        ],
        "cache_path": path.relative_to(cache_root).as_posix(),
        "cache_file_sha256": sha256_file(path),
        "payload_canonical_sha256": payload["payload_canonical_sha256"],
        "generation_seconds": float(payload.get("total_generation_seconds", 0.0)),
        "total_attempt_count": int(payload.get("total_attempt_count", 10)),
        "rejection_count": int(payload.get("rejection_count", 0)),
        "rejection_reason_counts": payload.get("rejection_reason_counts", {}),
    }


def _write_progress(
    cache_root: Path,
    *,
    status: str,
    required_count: int,
    rows: list[dict[str, Any]],
    resumed_count: int,
    failure: dict[str, Any] | None = None,
) -> None:
    value = {
        "schema_version": PROGRESS_SCHEMA,
        "status": status,
        "required_sequence_count": required_count,
        "completed_sequence_count": len(rows),
        "completed_conformer_count": len(rows) * CONFORMERS_PER_SEQUENCE,
        "resumed_validated_sequence_count": resumed_count,
        "sequence_file_sha256": {
            row["peptide_sequence"]: row["cache_file_sha256"] for row in rows
        },
        "failure": failure,
    }
    atomic_write_json(
        cache_root / "progress.json",
        value,
        replace_existing=(cache_root / "progress.json").exists(),
    )


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")
    atomic_write_bytes(path, payload, replace_existing=path.exists())


def _next_work_directory(cache_root: Path, sequence: str) -> Path:
    base = cache_root / "work" / sequence_file_key(sequence)
    base.mkdir(parents=True, exist_ok=True)
    for index in range(1000):
        candidate = base / f"run_{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"sequence_work_directory_namespace_exhausted:{sequence}")


def _final_manifest(
    *,
    cache_root: Path,
    cache_contract: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    index_path = cache_root / "cache_index.jsonl"
    _write_jsonl_atomic(index_path, rows)
    attempts = [
        int(value)
        for row in rows
        for value in row["accepted_attempt_indices"]
    ]
    core = {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "status": "MATERIALIZED",
        "purpose": "bounded_train_valid_only",
        "plan_descriptor_file_sha256": cache_contract[
            "plan_descriptor_file_sha256"
        ],
        "plan_descriptor_canonical_sha256": cache_contract[
            "plan_descriptor_canonical_sha256"
        ],
        "cache_contract_canonical_sha256": cache_contract[
            "contract_canonical_sha256"
        ],
        "required_peptide_sequences": cache_contract[
            "required_peptide_sequences"
        ],
        "required_peptide_sequences_sha256": cache_contract[
            "required_peptide_sequences_sha256"
        ],
        "sequence_count": len(rows),
        "conformers_per_sequence": CONFORMERS_PER_SEQUENCE,
        "conformer_count": len(rows) * CONFORMERS_PER_SEQUENCE,
        "atom_cap_exclusive": ATOM_CAP_EXCLUSIVE,
        "index_path": index_path.relative_to(cache_root).as_posix(),
        "index_sha256": sha256_file(index_path),
        "sequence_file_sha256": {
            row["peptide_sequence"]: row["cache_file_sha256"] for row in rows
        },
        "sequence_file_sha256_aggregate": canonical_json_sha256(
            [
                [row["peptide_sequence"], row["cache_file_sha256"]]
                for row in rows
            ]
        ),
        "accepted_attempt_index_distribution": dict(
            sorted(Counter(attempts).items())
        ),
        "target_bound_generation_inputs_used": False,
        "training_or_retrieval_run": False,
    }
    return _canonical_record(core, "manifest_canonical_sha256")


def _smoke_manifest(
    *,
    cache_contract: dict[str, Any],
    rows: list[dict[str, Any]],
    selection_records: list[dict[str, Any]],
) -> dict[str, Any]:
    rejection_counts: Counter[str] = Counter()
    for row in rows:
        rejection_counts.update(row["rejection_reason_counts"])
    core = {
        "schema_version": SMOKE_MANIFEST_SCHEMA,
        "classification": "CACHE_MATERIALIZER_SMOKE_PASS",
        "formal_cache": False,
        "sequence_count": len(rows),
        "conformer_count": len(rows) * CONFORMERS_PER_SEQUENCE,
        "not_valid_for_training": True,
        "final_cache_manifest_created": False,
        "selection_records": selection_records,
        "cache_contract_canonical_sha256": cache_contract[
            "contract_canonical_sha256"
        ],
        "sequence_file_sha256": {
            row["peptide_sequence"]: row["cache_file_sha256"] for row in rows
        },
        "target_bound_generation_inputs_used": False,
        "generation_summary": {
            "total_generation_seconds": sum(
                row["generation_seconds"] for row in rows
            ),
            "total_attempt_count": sum(
                row["total_attempt_count"] for row in rows
            ),
            "total_rejection_count": sum(
                row["rejection_count"] for row in rows
            ),
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
            "maximum_atom_count": max(row["atom_count"] for row in rows),
        },
    }
    return _canonical_record(core, "smoke_manifest_canonical_sha256")


def materialize_cache(
    *,
    descriptor_contract: dict[str, Any],
    cache_root: Path,
    prior_manifest_file_sha256: str,
    prior_jsonl_file_sha256: str,
    tool: dict[str, Any],
    sequence_generator: SequenceGenerator,
    payload_validator: PayloadValidator = production_payload_validator,
    formal_cache: bool = True,
    selected_records: list[dict[str, Any]] | None = None,
    resume: bool = False,
    stop_after_new_sequences: int | None = None,
    expected_formal_sequence_count: int = EXPECTED_FORMAL_SEQUENCE_COUNT,
) -> dict[str, Any]:
    cache_root = Path(cache_root).resolve()
    all_records = _plan_sequence_records(descriptor_contract)
    if formal_cache:
        records = [all_records[value] for value in descriptor_contract[
            "required_sequences"
        ]]
    else:
        if not selected_records or len(selected_records) != 5:
            raise ValueError("smoke_requires_exactly_five_sequence_records")
        records = [
            {
                "peptide_sequence": row["peptide_sequence"],
                "theoretical_heavy_atom_count": int(
                    row["theoretical_heavy_atom_count"]
                ),
                "split_roles": list(row["split_roles"]),
            }
            for row in selected_records
        ]
    required_sequences = sorted(row["peptide_sequence"] for row in records)
    record_by_sequence = {row["peptide_sequence"]: row for row in records}
    if any("safe265" in part.lower() or "safe373" in part.lower()
           for part in cache_root.parts):
        raise ValueError("evaluation_cache_path_forbidden")
    contract = build_cache_contract(
        descriptor_contract=descriptor_contract,
        required_sequences=required_sequences,
        formal_cache=formal_cache,
        prior_manifest_file_sha256=prior_manifest_file_sha256,
        prior_jsonl_file_sha256=prior_jsonl_file_sha256,
        tool=tool,
    )
    contract_path = cache_root / "cache_contract.json"
    if cache_root.exists():
        if not resume:
            raise FileExistsError(f"cache_root_exists_without_resume:{cache_root}")
        if not contract_path.is_file():
            raise ValueError("resume_cache_contract_missing")
        existing_contract = _load_json(contract_path)
        _verify_canonical_record(
            existing_contract,
            "contract_canonical_sha256",
            "resume_cache_contract_canonical_sha256_mismatch",
        )
        if existing_contract != contract:
            raise ValueError("resume_cache_contract_mismatch")
    else:
        cache_root.mkdir(parents=True)
        (cache_root / "sequences").mkdir()
        (cache_root / "failures").mkdir()
        (cache_root / "work").mkdir()
        atomic_write_json(contract_path, contract)
    _check_no_stale_temporary_files(cache_root)
    sequences_dir = cache_root / "sequences"
    sequences_dir.mkdir(exist_ok=True)
    (cache_root / "failures").mkdir(exist_ok=True)
    (cache_root / "work").mkdir(exist_ok=True)
    blocking_failures = []
    for failure_path in sorted((cache_root / "failures").glob("*.json")):
        failure = _load_json(failure_path)
        if failure.get("classification") == "CONFORMER_COVERAGE_BLOCKED":
            blocking_failures.append(failure_path.name)
    if blocking_failures:
        raise RuntimeError(
            f"resume_blocked_by_slot_exhaustion:{blocking_failures}"
        )
    expected_names = {
        f"{sequence_file_key(sequence)}.json" for sequence in required_sequences
    }
    extra_names = {
        path.name for path in sequences_dir.glob("*.json")
    } - expected_names
    if extra_names:
        raise ValueError(f"extra_sequence_cache_files:{sorted(extra_names)}")

    recorded_progress_hashes: dict[str, str] = {}
    progress_path = cache_root / "progress.json"
    if progress_path.is_file():
        try:
            progress_value = _load_json(progress_path)
            raw_hashes = progress_value.get("sequence_file_sha256", {})
            if isinstance(raw_hashes, dict):
                recorded_progress_hashes = {
                    str(sequence): str(value)
                    for sequence, value in raw_hashes.items()
                }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            recorded_progress_hashes = {}

    rows: list[dict[str, Any]] = []
    resumed_count = 0
    new_count = 0
    for sequence in required_sequences:
        path = sequences_dir / f"{sequence_file_key(sequence)}.json"
        if path.exists():
            payload, _ = validate_existing_sequence_file(
                path,
                sequence=sequence,
                cache_contract=contract,
                payload_validator=payload_validator,
            )
            recorded_file_sha = recorded_progress_hashes.get(sequence)
            if (
                recorded_file_sha is not None
                and sha256_file(path) != recorded_file_sha
            ):
                raise ValueError(
                    f"completed_sequence_file_sha256_changed:{sequence}"
                )
            resumed_count += 1
        else:
            try:
                generated = sequence_generator(
                    sequence, _next_work_directory(cache_root, sequence)
                )
                payload = _bind_sequence_payload(
                    generated,
                    sequence=sequence,
                    cache_contract=contract,
                )
                _validate_generic_payload(payload, sequence)
                payload_validator(payload)
                atomic_write_json(path, payload)
            except Exception as error:
                failure = _canonical_record(
                    {
                        "schema_version": FAILURE_SCHEMA,
                        "classification": (
                            "CONFORMER_COVERAGE_BLOCKED"
                            if isinstance(error, ConformerCoverageError)
                            else "CACHE_MATERIALIZER_GENERATION_FAIL"
                        ),
                        "peptide_sequence": sequence,
                        "sequence_sha256": sequence_file_key(sequence),
                        "exception_type": type(error).__name__,
                        "exception_text": str(error),
                        "details": getattr(error, "details", None),
                        "completed_sequence_count": len(rows),
                        "cache_manifest_created": False,
                    },
                    "failure_canonical_sha256",
                )
                failure_path = (
                    cache_root / "failures"
                    / f"{sequence_file_key(sequence)}.json"
                )
                if not failure_path.exists():
                    atomic_write_json(failure_path, failure)
                _write_progress(
                    cache_root,
                    status="FAIL",
                    required_count=len(required_sequences),
                    rows=rows,
                    resumed_count=resumed_count,
                    failure=failure,
                )
                raise
            new_count += 1
        rows.append(
            _sequence_index_row(
                sequence=sequence,
                path=path,
                payload=payload,
                record=record_by_sequence[sequence],
                cache_root=cache_root,
            )
        )
        _write_progress(
            cache_root,
            status="IN_PROGRESS",
            required_count=len(required_sequences),
            rows=rows,
            resumed_count=resumed_count,
        )
        if (
            stop_after_new_sequences is not None
            and new_count >= stop_after_new_sequences
            and len(rows) < len(required_sequences)
        ):
            return {
                "classification": "INTERRUPTED_FOR_RESUME_SMOKE",
                "formal_cache": formal_cache,
                "completed_sequence_count": len(rows),
                "required_sequence_count": len(required_sequences),
                "resumed_validated_sequence_count": resumed_count,
                "newly_generated_sequence_count": new_count,
                "cache_manifest_created": False,
            }

    if len(rows) != len(required_sequences):
        raise AssertionError("materializer_completion_count_mismatch")
    if formal_cache:
        if len(rows) != expected_formal_sequence_count:
            raise ValueError("formal_manifest_requires_2085_sequences")
        manifest = _final_manifest(
            cache_root=cache_root, cache_contract=contract, rows=rows
        )
        atomic_write_json(cache_root / "cache_manifest.json", manifest)
        classification = "BOUNDED_FULL_HEAVY_CACHE_MATERIALIZED"
        manifest_name = "cache_manifest.json"
    else:
        manifest = _smoke_manifest(
            cache_contract=contract,
            rows=rows,
            selection_records=selected_records or [],
        )
        atomic_write_json(
            cache_root / "smoke_manifest.json",
            manifest,
            replace_existing=(cache_root / "smoke_manifest.json").exists(),
        )
        classification = "CACHE_MATERIALIZER_SMOKE_PASS"
        manifest_name = "smoke_manifest.json"
    _write_progress(
        cache_root,
        status="PASS",
        required_count=len(required_sequences),
        rows=rows,
        resumed_count=resumed_count,
    )
    return {
        "classification": classification,
        "formal_cache": formal_cache,
        "sequence_count": len(rows),
        "conformer_count": len(rows) * CONFORMERS_PER_SEQUENCE,
        "resumed_validated_sequence_count": resumed_count,
        "newly_generated_sequence_count": new_count,
        "manifest_name": manifest_name,
        "manifest": manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a resumable bounded full-heavy sequence cache."
    )
    parser.add_argument("--plan-descriptor", required=True)
    parser.add_argument("--prior-manifest", required=True)
    parser.add_argument("--prior-jsonl", required=True)
    parser.add_argument("--faspr-executable", required=True)
    parser.add_argument("--faspr-rotamer-library", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke-five",
        action="store_true",
        help="Generate only the mechanical five-sequence non-training smoke.",
    )
    parser.add_argument(
        "--stop-after-new-sequences",
        type=int,
        default=None,
        help="Test-only interruption hook; never use for a formal run.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    descriptor = load_descriptor_contract(Path(args.plan_descriptor))
    tool_prior = verify_tool_and_prior(
        prior_manifest_path=Path(args.prior_manifest).resolve(),
        prior_jsonl_path=Path(args.prior_jsonl).resolve(),
        faspr_executable=Path(args.faspr_executable).resolve(),
        faspr_rotamer_library=Path(args.faspr_rotamer_library).resolve(),
    )

    def generator(sequence: str, work_dir: Path) -> dict[str, Any]:
        return generate_train_only_faspr_conformers(
            sequence,
            torsion_prior_groups=tool_prior["prior_groups"],
            torsion_prior_manifest=tool_prior["prior_manifest"],
            work_dir=work_dir,
            faspr_executable=Path(args.faspr_executable).resolve(),
            faspr_commit_sha=tool_prior["tool"]["commit_sha"],
            faspr_binary_sha256=tool_prior["tool"]["binary_sha256"],
            attempt_qc=validate_attempt_payload,
        )

    result = materialize_cache(
        descriptor_contract=descriptor,
        cache_root=Path(args.output_dir),
        prior_manifest_file_sha256=tool_prior[
            "prior_manifest_file_sha256"
        ],
        prior_jsonl_file_sha256=tool_prior["prior_jsonl_file_sha256"],
        tool=tool_prior["tool"],
        sequence_generator=generator,
        formal_cache=not args.smoke_five,
        selected_records=(
            select_smoke_sequences(descriptor) if args.smoke_five else None
        ),
        resume=args.resume,
        stop_after_new_sequences=args.stop_after_new_sequences,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

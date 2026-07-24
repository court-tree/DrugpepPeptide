"""Freeze a deterministic Phase-3 v2 bounded full-heavy plan descriptor.

This command consumes the completed full-split eligibility audit.  It does
not inspect structures, generate conformers, invoke FASPR, create a cache, or
construct a trainable adaptation manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from phase3.drugclip.full_heavy_adaptation_contract import (
    ATOM_CAP_EXCLUSIVE,
    CONFORMERS_PER_SEQUENCE,
    ELIGIBILITY_REGISTRY_SCHEMA,
    MAX_TRAIN_PAIRS,
    MAX_VALID_PAIRS,
    PLAN_SCHEMA,
    PLAN_SELECTION_ALGORITHM_VERSION,
    SAFE373_PLAN_CANONICAL_SHA256,
    bounded_plan_selection_key,
    canonical_json_sha256,
    sequence_sha256,
    sha256_file,
    validate_bounded_plan_descriptor,
)


FORMAL_TRAIN_PAIR_COUNT = 19_707
FORMAL_VALID_PAIR_COUNT = 2_463
CLASSIFICATION_PASS = "EXPLICIT_BOUNDED_PLAN_PASS"


class PlanCapacityError(RuntimeError):
    pass


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"expected_jsonl_object:{path}:{line_number}"
                )
            rows.append(value)
    return rows


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _relative_path(target: Path, anchor: Path) -> str:
    value = os.path.relpath(target.resolve(), anchor.resolve())
    return value.replace("\\", "/")


def _biological_relation_index(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], str]:
    output: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (
            str(row["biological_receptor_id"]),
            str(row["peptide_sequence"]).upper(),
        )
        if key in output:
            raise ValueError(f"duplicate_biological_relation_key:{key}")
        output[key] = str(row["pair_id"])
    return output


def _formal_pair_rows(
    rows: list[dict[str, Any]],
    split: str,
    relation_index: dict[tuple[str, str], str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        sequence = str(row["peptide_sequence"]).upper()
        relation_key = (str(row["biological_receptor_id"]), sequence)
        if relation_key not in relation_index:
            raise ValueError(
                f"missing_biological_relation:{row['pair_id']}"
            )
        output.append({
            "interface_pair_id": str(row["pair_id"]),
            "split": split,
            "peptide_sequence": sequence,
            "biological_relation_id": relation_index[relation_key],
            "source_database": str(
                row.get("source_database") or "unspecified"
            ),
        })
    return output


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _distribution(values: Iterable[int]) -> dict[str, Any]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
            "histogram": {},
        }
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": _percentile(ordered, 0.25),
        "median": _percentile(ordered, 0.50),
        "p75": _percentile(ordered, 0.75),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "histogram": {
            str(key): value
            for key, value in sorted(Counter(ordered).items())
        },
    }


def _selection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sequences = Counter(row["peptide_sequence"] for row in rows)
    relations = {
        row["biological_relation_id"] for row in rows
    }
    sources = Counter(row["source_database"] for row in rows)
    return {
        "pair_count": len(rows),
        "unique_sequence_count": len(sequences),
        "biological_relation_count": len(relations),
        "source_database_pair_counts": dict(sorted(sources.items())),
        "source_database_pair_fractions": {
            key: value / len(rows)
            for key, value in sorted(sources.items())
        } if rows else {},
        "peptide_length_distribution": _distribution(
            len(row["peptide_sequence"]) for row in rows
        ),
        "pairs_per_sequence_distribution": _distribution(
            sequences.values()
        ),
    }


def _distribution_audit(
    eligible: dict[str, list[dict[str, Any]]],
    selected: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": (
            "phase3-v2-bounded-full-heavy-plan-distribution-audit-v1"
        ),
        "selection_was_not_manually_adjusted": True,
        "splits": {},
    }
    for split in ("train", "valid"):
        universe = _selection_summary(eligible[split])
        plan = _selection_summary(selected[split])
        source_names = sorted(
            set(universe["source_database_pair_fractions"])
            | set(plan["source_database_pair_fractions"])
        )
        output["splits"][split] = {
            "eligible_universe": universe,
            "selected_plan": plan,
            "bias_relative_to_eligible_universe": {
                "source_database_pair_fraction_delta": {
                    source: (
                        plan["source_database_pair_fractions"].get(
                            source, 0.0
                        )
                        - universe[
                            "source_database_pair_fractions"
                        ].get(source, 0.0)
                    )
                    for source in source_names
                },
                "mean_peptide_length_delta": (
                    plan["peptide_length_distribution"]["mean"]
                    - universe["peptide_length_distribution"]["mean"]
                ),
                "median_peptide_length_delta": (
                    plan["peptide_length_distribution"]["median"]
                    - universe["peptide_length_distribution"]["median"]
                ),
                "mean_pairs_per_sequence_delta": (
                    plan["pairs_per_sequence_distribution"]["mean"]
                    - universe["pairs_per_sequence_distribution"]["mean"]
                ),
                "biological_relation_coverage_fraction": (
                    plan["biological_relation_count"]
                    / universe["biological_relation_count"]
                ),
            },
        }
    return output


def _plan_payload(
    split: str,
    selected: list[dict[str, Any]],
    registry_by_sequence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pair_ids = [row["interface_pair_id"] for row in selected]
    sequences = sorted({row["peptide_sequence"] for row in selected})
    pair_counts = Counter(row["peptide_sequence"] for row in selected)
    return {
        "schema_version": PLAN_SCHEMA,
        "split": split,
        "selection_algorithm_version": PLAN_SELECTION_ALGORITHM_VERSION,
        "pair_count": len(pair_ids),
        "interface_pair_ids": pair_ids,
        "interface_pair_ids_sha256": sequence_sha256(pair_ids),
        "unique_peptide_sequence_count": len(sequences),
        "unique_peptide_sequences": sequences,
        "unique_peptide_sequences_sha256": sequence_sha256(sequences),
        "sequence_records": [{
            "peptide_sequence": sequence,
            "selected_pair_count": pair_counts[sequence],
            "theoretical_heavy_atom_count": int(
                registry_by_sequence[sequence][
                    "theoretical_heavy_atom_count"
                ]
            ),
        } for sequence in sequences],
        "conformers_per_sequence": CONFORMERS_PER_SEQUENCE,
        "future_required_conformer_count": (
            len(sequences) * CONFORMERS_PER_SEQUENCE
        ),
    }


def _bundle(
    *,
    output_dir: Path,
    registry_path: Path,
    audit_report_path: Path,
    train_path: Path,
    valid_path: Path,
    biological_path: Path,
    safe373_plan_path: Path,
) -> dict[str, Any]:
    registry_rows = read_jsonl(registry_path)
    audit_report = read_json(audit_report_path)
    train_source_rows = read_jsonl(train_path)
    valid_source_rows = read_jsonl(valid_path)
    if len(train_source_rows) != FORMAL_TRAIN_PAIR_COUNT:
        raise ValueError("formal_train_pair_count_mismatch")
    if len(valid_source_rows) != FORMAL_VALID_PAIR_COUNT:
        raise ValueError("formal_valid_pair_count_mismatch")
    if audit_report.get("classification") != "CORE_LINEAR_SUBSET_SUFFICIENT":
        raise ValueError("eligibility_audit_not_core_linear_sufficient")
    if int(
        audit_report.get("registry", {}).get("sequence_count", -1)
    ) != len(registry_rows):
        raise ValueError("eligibility_audit_registry_count_mismatch")
    if audit_report.get("registry", {}).get(
        "file_sha256"
    ) != sha256_file(registry_path):
        raise ValueError("eligibility_audit_registry_sha256_mismatch")

    relation_index = _biological_relation_index(
        read_jsonl(biological_path)
    )
    formal = {
        "train": _formal_pair_rows(
            train_source_rows, "train", relation_index
        ),
        "valid": _formal_pair_rows(
            valid_source_rows, "valid", relation_index
        ),
    }
    formal_by_id = {
        split: {
            row["interface_pair_id"]: row
            for row in formal[split]
        }
        for split in ("train", "valid")
    }
    registry_by_sequence: dict[str, dict[str, Any]] = {}
    eligible_ids: dict[str, list[str]] = {"train": [], "valid": []}
    for row in registry_rows:
        sequence = str(row.get("peptide_sequence") or "")
        split = str(row.get("split") or "")
        if sequence in registry_by_sequence:
            raise ValueError(f"duplicate_registry_sequence:{sequence}")
        if split not in {"train", "valid"}:
            raise ValueError(f"invalid_registry_split:{sequence}")
        if bool(row.get("eligible", False)):
            if (
                row.get("chemistry_classification")
                != "ordinary_linear_standard"
                or int(row.get("theoretical_heavy_atom_count", -1))
                >= ATOM_CAP_EXCLUSIVE
                or not bool(row.get("torsion_prior_covered", False))
            ):
                raise ValueError(
                    f"registry_eligible_contract_mismatch:{sequence}"
                )
            eligible_ids[split].extend(
                str(value)
                for value in row.get("interface_pair_ids", [])
            )
        registry_by_sequence[sequence] = row

    targets = {"train": MAX_TRAIN_PAIRS, "valid": MAX_VALID_PAIRS}
    eligible_rows: dict[str, list[dict[str, Any]]] = {}
    selected_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "valid"):
        missing = sorted(
            set(eligible_ids[split]) - set(formal_by_id[split])
        )
        if missing:
            raise ValueError(
                f"eligible_registry_pair_missing_from_{split}:{missing[:3]}"
            )
        ordered_ids = sorted(
            eligible_ids[split],
            key=lambda pair_id: (
                bounded_plan_selection_key(split, pair_id),
                pair_id,
            ),
        )
        if len(ordered_ids) < targets[split]:
            raise PlanCapacityError(
                f"PLAN_CAPACITY_FAIL:{split}:"
                f"{len(ordered_ids)}<{targets[split]}"
            )
        eligible_rows[split] = [
            formal_by_id[split][pair_id] for pair_id in ordered_ids
        ]
        selected_rows[split] = eligible_rows[split][:targets[split]]

    plans = {
        split: _plan_payload(
            split, selected_rows[split], registry_by_sequence
        )
        for split in ("train", "valid")
    }
    plan_bytes = {
        split: canonical_json_bytes(plans[split])
        for split in ("train", "valid")
    }
    safe_plan = read_json(safe373_plan_path)
    if (
        safe_plan.get("plan_canonical_sha256")
        != SAFE373_PLAN_CANONICAL_SHA256
    ):
        raise ValueError("safe373_plan_canonical_sha256_mismatch")
    valid_relation_by_pair = {
        row["interface_pair_id"]: row["biological_relation_id"]
        for row in formal["valid"]
    }
    train_relations = {
        row["biological_relation_id"] for row in selected_rows["train"]
    }
    valid_relations = {
        row["biological_relation_id"] for row in selected_rows["valid"]
    }
    train_sequences = set(plans["train"]["unique_peptide_sequences"])
    valid_sequences = set(plans["valid"]["unique_peptide_sequences"])
    safe_pairs = {
        str(value)
        for value in safe_plan.get("safe_query_interface_pair_ids", [])
    }
    safe_sequences = {
        str(value)
        for value in safe_plan.get("safe_peptide_candidate_ids", [])
    }
    safe_relations = {
        valid_relation_by_pair[pair_id]
        for pair_id in safe_pairs
        if pair_id in valid_relation_by_pair
    }
    if (
        set(plans["train"]["interface_pair_ids"]) & safe_pairs
        or train_sequences & safe_sequences
        or train_relations & safe_relations
    ):
        raise ValueError("train_safe373_leakage")
    if train_sequences & valid_sequences:
        raise ValueError("train_valid_sequence_leakage")
    if train_relations & valid_relations:
        raise ValueError("train_valid_relation_leakage")

    valid_overlap_values = {
        "query_pair_ids": sorted(
            set(plans["valid"]["interface_pair_ids"]) & safe_pairs
        ),
        "peptide_sequences": sorted(valid_sequences & safe_sequences),
        "biological_relation_ids": sorted(valid_relations & safe_relations),
    }
    valid_overlap: dict[str, Any] = {}
    for key, values in valid_overlap_values.items():
        valid_overlap[key] = values
        valid_overlap[f"{key}_sha256"] = sequence_sha256(values)

    required_sequences = sorted(train_sequences | valid_sequences)
    descriptor_core = {
        "schema_version": PLAN_SCHEMA,
        "frozen_inputs": {
            "eligibility_registry": {
                "path": _relative_path(registry_path, output_dir),
                "file_sha256": sha256_file(registry_path),
                "canonical_sha256": canonical_json_sha256(registry_rows),
                "schema_version": ELIGIBILITY_REGISTRY_SCHEMA,
                "sequence_count": len(registry_rows),
            },
            "full_split_audit_report": {
                "path": _relative_path(audit_report_path, output_dir),
                "file_sha256": sha256_file(audit_report_path),
            },
            "formal_split_sources": {
                "train": {
                    "path": _relative_path(train_path, output_dir),
                    "file_sha256": sha256_file(train_path),
                    "pair_count": len(formal["train"]),
                },
                "valid": {
                    "path": _relative_path(valid_path, output_dir),
                    "file_sha256": sha256_file(valid_path),
                    "pair_count": len(formal["valid"]),
                },
                "biological_relations": {
                    "path": _relative_path(biological_path, output_dir),
                    "file_sha256": sha256_file(biological_path),
                },
            },
        },
        "eligibility_contract": {
            "rule_version": ELIGIBILITY_REGISTRY_SCHEMA,
            "required_chemistry_classification": (
                "ordinary_linear_standard"
            ),
            "theoretical_heavy_atom_count_operator": "<",
            "theoretical_heavy_atom_count_limit": ATOM_CAP_EXCLUSIVE,
            "torsion_prior_coverage_required": True,
            "sequence_level_all_structure_instances_required": True,
        },
        "selection_contract": {
            "algorithm_version": PLAN_SELECTION_ALGORITHM_VERSION,
            "namespace": PLAN_SCHEMA,
            "primary_key": (
                'SHA256("phase3-v2-bounded-full-heavy-plan-v1" + "\\0" '
                '+ split + "\\0" + interface_pair_id)'
            ),
            "tie_breaker": "interface_pair_id",
            "target_train_pair_count": MAX_TRAIN_PAIRS,
            "target_valid_pair_count": MAX_VALID_PAIRS,
        },
        "plan_files": {
            split: {
                "path": f"{split}_interface_pair_plan.json",
                "file_sha256": hashlib.sha256(plan_bytes[split]).hexdigest(
                ).upper(),
            }
            for split in ("train", "valid")
        },
        "plans": plans,
        "safe373_evaluation_exclusion": {
            "plan_path": _relative_path(safe373_plan_path, output_dir),
            "plan_file_sha256": sha256_file(safe373_plan_path),
            "plan_canonical_sha256": SAFE373_PLAN_CANONICAL_SHA256,
            "train_overlap_required_zero": {
                "query_pair": True,
                "peptide_sequence": True,
                "biological_relation": True,
            },
            "valid_overlap_report": valid_overlap,
        },
        "future_cache_requirement": {
            "generation_status": "NOT_BUILT",
            "cache_status": "NOT_BUILT",
            "cache_manifest_path": None,
            "cache_manifest_sha256": None,
            "conformers_per_sequence": CONFORMERS_PER_SEQUENCE,
            "required_peptide_sequences": required_sequences,
            "required_peptide_sequences_sha256": sequence_sha256(
                required_sequences
            ),
            "future_required_conformer_count": (
                len(required_sequences) * CONFORMERS_PER_SEQUENCE
            ),
            "safe373_evaluation_cache_reuse_forbidden": True,
        },
    }
    descriptor = {
        **descriptor_core,
        "descriptor_canonical_sha256": canonical_json_sha256(
            descriptor_core
        ),
    }
    distribution = _distribution_audit(
        eligible_rows, selected_rows
    )
    return {
        "descriptor": descriptor,
        "descriptor_bytes": canonical_json_bytes(descriptor),
        "plans": plans,
        "plan_bytes": plan_bytes,
        "distribution": distribution,
        "distribution_bytes": canonical_json_bytes(distribution),
        "formal": formal,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a deterministic bounded full-heavy plan descriptor "
            "from the completed full-split eligibility audit."
        )
    )
    parser.add_argument("--eligibility-registry", required=True)
    parser.add_argument("--full-split-audit-report", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--safe373-plan", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output_dir_already_exists:{output_dir}")
    dataset_root = Path(args.dataset_root).resolve()
    paths = {
        "output_dir": output_dir,
        "registry_path": Path(args.eligibility_registry).resolve(),
        "audit_report_path": Path(args.full_split_audit_report).resolve(),
        "train_path": (
            dataset_root / "02_leakage_safe_split" / "train.jsonl"
        ),
        "valid_path": (
            dataset_root / "02_leakage_safe_split" / "valid.jsonl"
        ),
        "biological_path": (
            dataset_root / "dependencies" / "biological_pairs.jsonl"
        ),
        "safe373_plan_path": Path(args.safe373_plan).resolve(),
    }
    first = _bundle(**paths)
    second = _bundle(**paths)
    byte_identical = all((
        first["descriptor_bytes"] == second["descriptor_bytes"],
        first["plan_bytes"]["train"] == second["plan_bytes"]["train"],
        first["plan_bytes"]["valid"] == second["plan_bytes"]["valid"],
        first["distribution_bytes"] == second["distribution_bytes"],
    ))
    if not byte_identical:
        raise RuntimeError("PLAN_IMPLEMENTATION_FAIL:non_deterministic_bundle")

    output_dir.mkdir(parents=True, exist_ok=False)
    for split in ("train", "valid"):
        _atomic_bytes(
            output_dir / f"{split}_interface_pair_plan.json",
            first["plan_bytes"][split],
        )
    descriptor_path = output_dir / "bounded_plan_descriptor.json"
    _atomic_bytes(descriptor_path, first["descriptor_bytes"])
    _atomic_bytes(
        output_dir / "plan_distribution_audit.json",
        first["distribution_bytes"],
    )

    formal = first["formal"]
    train_sequence_by_pair = {
        row["interface_pair_id"]: row["peptide_sequence"]
        for row in formal["train"]
    }
    valid_sequence_by_pair = {
        row["interface_pair_id"]: row["peptide_sequence"]
        for row in formal["valid"]
    }
    train_relation_by_pair = {
        row["interface_pair_id"]: row["biological_relation_id"]
        for row in formal["train"]
    }
    valid_relation_by_pair = {
        row["interface_pair_id"]: row["biological_relation_id"]
        for row in formal["valid"]
    }
    try:
        validated = validate_bounded_plan_descriptor(
            descriptor_path,
            train_interface_pair_ids=train_sequence_by_pair,
            valid_interface_pair_ids=valid_sequence_by_pair,
            train_sequence_by_pair=train_sequence_by_pair,
            valid_sequence_by_pair=valid_sequence_by_pair,
            train_relation_by_pair=train_relation_by_pair,
            valid_relation_by_pair=valid_relation_by_pair,
        )
    except Exception as exc:
        report = {
            "status": "FAIL",
            "classification": "PLAN_IMPLEMENTATION_FAIL",
            "exception_type": type(exc).__name__,
            "exception_text": str(exc),
            "descriptor_double_build_byte_identical": byte_identical,
        }
        _atomic_bytes(
            output_dir / "plan_validation_report.json",
            canonical_json_bytes(report),
        )
        raise

    descriptor = first["descriptor"]
    report = {
        "schema_version": (
            "phase3-v2-bounded-full-heavy-plan-validation-v1"
        ),
        "status": "PASS",
        "classification": CLASSIFICATION_PASS,
        "descriptor_file_sha256": sha256_file(descriptor_path),
        "descriptor_canonical_sha256": descriptor[
            "descriptor_canonical_sha256"
        ],
        "descriptor_double_build_byte_identical": byte_identical,
        "train_pair_count": len(validated["train_interface_pair_ids"]),
        "valid_pair_count": len(validated["valid_interface_pair_ids"]),
        "train_unique_sequence_count": len(
            validated["train_unique_peptide_sequences"]
        ),
        "valid_unique_sequence_count": len(
            validated["valid_unique_peptide_sequences"]
        ),
        "future_required_unique_sequence_count": len(
            validated["required_peptide_sequences"]
        ),
        "future_required_conformer_count": descriptor[
            "future_cache_requirement"
        ]["future_required_conformer_count"],
        "train_valid_overlap": validated["train_valid_overlap"],
        "train_safe373_overlap": validated["train_safe373_overlap"],
        "valid_safe373_overlap": validated["valid_safe373_overlap"],
        "generation_status": "NOT_BUILT",
        "cache_status": "NOT_BUILT",
        "adaptation_manifest_created": False,
        "forbidden_actions": {
            "full_split_chemistry_audit_rerun": False,
            "faspr_invoked": False,
            "conformer_generated": False,
            "cache_generated": False,
            "optimizer_created": False,
            "backward_executed": False,
            "training_executed": False,
            "gpu_retrieval_executed": False,
        },
    }
    _atomic_bytes(
        output_dir / "plan_validation_report.json",
        canonical_json_bytes(report),
    )
    summary = (
        "# Phase-3 v2 explicit bounded full-heavy plan\n\n"
        f"- Classification: `{CLASSIFICATION_PASS}`\n"
        f"- Descriptor canonical SHA256: "
        f"`{descriptor['descriptor_canonical_sha256']}`\n"
        f"- Train/valid pairs: "
        f"{report['train_pair_count']}/{report['valid_pair_count']}.\n"
        f"- Train/valid unique sequences: "
        f"{report['train_unique_sequence_count']}/"
        f"{report['valid_unique_sequence_count']}.\n"
        f"- Future required conformers: "
        f"{report['future_required_conformer_count']}.\n"
        "- Generation/cache status: `NOT_BUILT` / `NOT_BUILT`.\n"
        "- No adaptation manifest, conformer cache, optimizer, training, "
        "or retrieval was created or run.\n"
    )
    _atomic_bytes(output_dir / "summary.md", summary.encode("utf-8"))
    print(json.dumps({
        "classification": CLASSIFICATION_PASS,
        "descriptor_canonical_sha256": descriptor[
            "descriptor_canonical_sha256"
        ],
        "output_dir": str(output_dir),
        "train_pair_count": report["train_pair_count"],
        "valid_pair_count": report["valid_pair_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

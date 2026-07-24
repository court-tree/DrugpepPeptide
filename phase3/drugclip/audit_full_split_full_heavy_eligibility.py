"""Audit full formal train/valid chemistry eligibility for full-heavy input.

This command is read-only with respect to formal data.  It does not select a
bounded plan, generate coordinates, invoke FASPR, load a model, or create a
training/cache manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Any, Iterable

from phase3.drugclip.full_atom_conformer_prototype import _base_molecule
from phase3.drugclip.full_heavy_adaptation_contract import (
    ATOM_CAP_EXCLUSIVE,
    TORSION_PRIOR_JSONL_SHA256,
    TORSION_PRIOR_MANIFEST_SHA256,
    canonical_json_sha256,
    sha256_file,
)
from phase3.drugclip.train_only_torsion_prior_prototype import (
    context_key,
    load_torsion_prior,
)
from phase3.drugclip.validate_full_atom_conformer_prototype import (
    audit_one_chemistry,
)
from phase3.drugclip.validate_train_only_torsion_prototype import (
    _evidence_index,
    _select_evidence_row,
)


FORMAL_TRAIN_PAIR_COUNT = 19_707
FORMAL_VALID_PAIR_COUNT = 2_463
CHEMISTRY_PRECEDENCE = (
    "modified_or_nonstandard",
    "receptor_covalent",
    "known_disulfide",
    "cyclic_or_crosslinked",
    "multiple_cys_unknown",
    "chemistry_insufficient",
    "ordinary_linear_standard",
)
AUDIT_SCHEMA = "phase3-v2-full-formal-split-chemistry-applicability-audit-v1"
REGISTRY_SCHEMA = "phase3-v2-full-formal-split-sequence-eligibility-v1"


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
                raise ValueError(f"expected_jsonl_object:{path}:{line_number}")
            rows.append(value)
    return rows


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    _atomic_text(path, text)


def _biological_relation_index(
    biological_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], str]:
    output: dict[tuple[str, str], str] = {}
    for row in biological_rows:
        key = (
            str(row["biological_receptor_id"]),
            str(row["peptide_sequence"]).upper(),
        )
        if key in output:
            raise ValueError(f"ambiguous_biological_relation:{key}")
        output[key] = str(row["pair_id"])
    return output


def _formal_pair(
    row: dict[str, Any],
    split: str,
    relation_index: dict[tuple[str, str], str],
) -> dict[str, Any]:
    sequence = str(row["peptide_sequence"]).upper()
    relation_key = (str(row["biological_receptor_id"]), sequence)
    if relation_key not in relation_index:
        raise ValueError(f"missing_biological_relation:{row['pair_id']}")
    return {
        "interface_pair_id": str(row["pair_id"]),
        "split": split,
        "biological_pair_id": relation_index[relation_key],
        "biological_receptor_id": str(row["biological_receptor_id"]),
        "receptor_interface_id": str(row["receptor_interface_id"]),
        "peptide_sequence": sequence,
        "evidence_ids": [str(value) for value in row.get("evidence_ids", [])],
        "source_database": str(row.get("source_database") or ""),
        "source_databases": sorted({
            str(value) for value in row.get("source_databases", [])
            if str(value)
        } | ({str(row.get("source_database"))} if row.get("source_database") else set())),
        "receptor_family": str(row.get("receptor_family") or ""),
        "structure_pdb_ids": sorted(
            str(value).lower() for value in row.get("structure_pdb_ids", [])
        ),
    }


def _structure_audit(
    pair: dict[str, Any],
    evidence_id: str,
    evidence: dict[str, list[dict[str, Any]]],
    mmcif_root: Path,
    qbiolip_root: Path,
    biolip_root: Path,
) -> dict[str, Any]:
    try:
        evidence_row = _select_evidence_row(
            evidence_id,
            pair["peptide_sequence"],
            evidence.get(evidence_id, []),
            mmcif_root,
            qbiolip_root,
            biolip_root,
        )
        result = audit_one_chemistry(
            {
                "interface_pair_id": pair["interface_pair_id"],
                "peptide_sequence": pair["peptide_sequence"],
            },
            {
                "evidence_id": evidence_id,
                "source_database": str(
                    evidence_row.get("source_database")
                    or pair["source_database"]
                ),
                "peptide_chain": str(
                    evidence_row.get("peptide_chain_id") or ""
                ),
                "structure_type": "coordinate",
            },
            evidence_row,
            qbiolip_root,
            biolip_root,
        )
        return {
            "evidence_id": evidence_id,
            "chemistry_classification": result["chemistry_classification"],
            "exclusion_reason": result.get("exclusion_reason"),
            "structure_path": result.get("structure_path"),
            "receptor_structure_path": result.get("receptor_structure_path"),
            "source_database": result.get("source_database"),
            "resolved_peptide_chain": result.get("resolved_peptide_chain"),
            "terminal_state_determined": result.get(
                "terminal_state_determined"
            ),
            "peptide_receptor_covalent_connection_detected": result.get(
                "peptide_receptor_covalent_connection_detected"
            ),
            "peptide_receptor_explicit_connections": result.get(
                "peptide_receptor_explicit_connections", []
            ),
            "peptide_other_covalent_geometry": result.get(
                "peptide_other_covalent_geometry", []
            ),
            "minimum_peptide_other_covalent_distance_angstrom": result.get(
                "minimum_peptide_other_covalent_distance_angstrom"
            ),
            "detectable_ss_bond": result.get("detectable_ss_bond"),
            "ss_bond_evidence": result.get("ss_bond_evidence", []),
            "head_to_tail_closure_detected": result.get(
                "head_to_tail_closure_detected"
            ),
            "noncanonical_internal_connections": result.get(
                "noncanonical_internal_connections", []
            ),
            "modified_residue_detected": result.get(
                "modified_residue_detected"
            ),
            "modified_residue_positions": result.get(
                "modified_residue_positions", []
            ),
            "residue_names": result.get("residue_names", []),
        }
    except Exception as exc:
        return {
            "evidence_id": evidence_id,
            "chemistry_classification": "chemistry_insufficient",
            "exclusion_reason": (
                f"chemistry_audit_error:{type(exc).__name__}:{exc}"
            ),
            "structure_path": None,
            "receptor_structure_path": None,
            "source_database": pair["source_database"],
            "resolved_peptide_chain": None,
            "terminal_state_determined": False,
            "peptide_receptor_covalent_connection_detected": None,
            "peptide_receptor_explicit_connections": [],
            "peptide_other_covalent_geometry": [],
            "minimum_peptide_other_covalent_distance_angstrom": None,
            "detectable_ss_bond": None,
            "ss_bond_evidence": [],
            "head_to_tail_closure_detected": None,
            "noncanonical_internal_connections": [],
            "modified_residue_detected": None,
            "modified_residue_positions": [],
            "residue_names": [],
        }


def build_sequence_eligibility_registry(
    formal_pairs: list[dict[str, Any]],
    *,
    evidence: dict[str, list[dict[str, Any]]],
    mmcif_root: Path,
    qbiolip_root: Path,
    biolip_root: Path,
    torsion_groups: dict[str, list[dict[str, Any]]],
    progress_every: int = 500,
) -> list[dict[str, Any]]:
    """Conservatively collapse every formal structure instance by sequence."""

    sequence_pairs: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for index, pair in enumerate(formal_pairs, start=1):
        instance_audits = []
        for evidence_id in pair["evidence_ids"]:
            key = (evidence_id, pair["peptide_sequence"])
            if key not in evidence_cache:
                evidence_cache[key] = _structure_audit(
                    pair,
                    evidence_id,
                    evidence,
                    mmcif_root,
                    qbiolip_root,
                    biolip_root,
                )
            instance_audits.append(dict(evidence_cache[key]))
        if not instance_audits:
            instance_audits = [{
                "evidence_id": None,
                "chemistry_classification": "chemistry_insufficient",
                "exclusion_reason": "formal_pair_has_no_evidence_id",
                "structure_path": None,
                "receptor_structure_path": None,
                "source_database": pair["source_database"],
                "resolved_peptide_chain": None,
                "terminal_state_determined": False,
            }]
        sequence_pairs[pair["peptide_sequence"]].append({
            "interface_pair_id": pair["interface_pair_id"],
            "biological_pair_id": pair["biological_pair_id"],
            "split": pair["split"],
            "source_database": pair["source_database"],
            "source_databases": pair["source_databases"],
            "receptor_family": pair["receptor_family"],
            "structure_pdb_ids": pair["structure_pdb_ids"],
            "evidence_instances": instance_audits,
        })
        if progress_every and (
            index % progress_every == 0 or index == len(formal_pairs)
        ):
            print(
                f"chemistry_audit={index}/{len(formal_pairs)}",
                flush=True,
            )

    registry: list[dict[str, Any]] = []
    split_by_sequence: dict[str, str] = {}
    for pair in formal_pairs:
        sequence = pair["peptide_sequence"]
        prior = split_by_sequence.setdefault(sequence, pair["split"])
        if prior != pair["split"]:
            raise ValueError(f"formal_train_valid_sequence_leakage:{sequence}")
    for sequence in sorted(sequence_pairs):
        instances = sorted(
            sequence_pairs[sequence],
            key=lambda row: row["interface_pair_id"],
        )
        structure_classes = [
            audit["chemistry_classification"]
            for instance in instances
            for audit in instance["evidence_instances"]
        ]
        chemistry = next(
            name for name in CHEMISTRY_PRECEDENCE if name in structure_classes
        )
        atom_count = int(_base_molecule(sequence).GetNumAtoms())
        contexts = sorted({
            context_key(sequence, index) for index in range(len(sequence))
        })
        missing_contexts = [
            key for key in contexts if key not in torsion_groups
        ]
        all_ordinary = bool(structure_classes) and all(
            value == "ordinary_linear_standard"
            for value in structure_classes
        )
        exclusion_reasons = sorted({
            str(audit["exclusion_reason"])
            for instance in instances
            for audit in instance["evidence_instances"]
            if audit.get("exclusion_reason")
        })
        if not all_ordinary:
            exclusion_reasons.append(
                f"sequence_has_nonordinary_structure_instance:{chemistry}"
            )
        if atom_count >= ATOM_CAP_EXCLUSIVE:
            exclusion_reasons.append(
                f"theoretical_heavy_atom_count_not_below_{ATOM_CAP_EXCLUSIVE}"
            )
        if missing_contexts:
            exclusion_reasons.append("torsion_prior_context_missing")
        eligible = (
            chemistry == "ordinary_linear_standard"
            and all_ordinary
            and atom_count < ATOM_CAP_EXCLUSIVE
            and not missing_contexts
        )
        registry.append({
            "schema_version": REGISTRY_SCHEMA,
            "peptide_sequence": sequence,
            "sequence_length": len(sequence),
            "split": split_by_sequence[sequence],
            "interface_pair_ids": [
                row["interface_pair_id"] for row in instances
            ],
            "biological_pair_ids": sorted({
                row["biological_pair_id"] for row in instances
            }),
            "source_databases": sorted({
                source
                for row in instances
                for source in row["source_databases"]
            }),
            "receptor_families": sorted({
                row["receptor_family"] for row in instances
                if row["receptor_family"]
            }),
            "structure_pdb_ids": sorted({
                pdb_id
                for row in instances
                for pdb_id in row["structure_pdb_ids"]
            }),
            "pair_count": len(instances),
            "structure_instance_count": len(structure_classes),
            "structure_instance_classifications": structure_classes,
            "chemistry_classification": chemistry,
            "all_structure_instances_ordinary_linear_standard": all_ordinary,
            "theoretical_heavy_atom_count": atom_count,
            "atom_cap_exclusive": ATOM_CAP_EXCLUSIVE,
            "torsion_prior_contexts": contexts,
            "missing_torsion_prior_contexts": missing_contexts,
            "torsion_prior_covered": not missing_contexts,
            "eligible": eligible,
            "exclusion_reasons": sorted(set(exclusion_reasons)),
            "evidence_instances": instances,
        })
    return registry


def select_eligible_pair_plan(
    formal_pairs: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    *,
    split: str,
    target_count: int,
) -> list[dict[str, Any]]:
    eligible_sequences = {
        row["peptide_sequence"]
        for row in registry
        if row["split"] == split and row["eligible"]
    }
    eligible_pairs = sorted(
        (
            pair for pair in formal_pairs
            if pair["split"] == split
            and pair["peptide_sequence"] in eligible_sequences
        ),
        key=lambda row: row["interface_pair_id"],
    )
    if len(eligible_pairs) < target_count:
        raise RuntimeError(
            f"ELIGIBLE_PLAN_CAPACITY_FAIL:{split}:"
            f"{len(eligible_pairs)}<{target_count}"
        )
    return eligible_pairs[:target_count]


def _plan_payload(
    selected: list[dict[str, Any]],
    registry_by_sequence: dict[str, dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    pair_ids = [row["interface_pair_id"] for row in selected]
    sequences = sorted({row["peptide_sequence"] for row in selected})
    pair_counts = Counter(row["peptide_sequence"] for row in selected)
    return {
        "schema_version": PLAN_SCHEMA,
        "split": split,
        "selection_policy": (
            "filter by frozen sequence eligibility; sort eligible "
            "interface_pair_id lexicographically; take exact prefix"
        ),
        "interface_pair_ids": pair_ids,
        "interface_pair_ids_sha256": sequence_sha256(pair_ids),
        "unique_peptide_sequences": sequences,
        "unique_peptide_sequences_sha256": sequence_sha256(sequences),
        "pair_count": len(pair_ids),
        "unique_peptide_sequence_count": len(sequences),
        "sequence_records": [{
            "peptide_sequence": sequence,
            "selected_pair_count": pair_counts[sequence],
            "formal_split_pair_count": registry_by_sequence[sequence][
                "pair_count"
            ],
            "theoretical_heavy_atom_count": registry_by_sequence[sequence][
                "theoretical_heavy_atom_count"
            ],
            "chemistry_classification": registry_by_sequence[sequence][
                "chemistry_classification"
            ],
            "eligible": registry_by_sequence[sequence]["eligible"],
        } for sequence in sequences],
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build explicit bounded full-heavy plans without a cache."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--candidate-evidence-jsonl", required=True)
    parser.add_argument("--expanded-evidence-jsonl", required=True)
    parser.add_argument("--mmcif-root", required=True)
    parser.add_argument("--qbiolip-root", required=True)
    parser.add_argument("--biolip-root", required=True)
    parser.add_argument("--torsion-prior-manifest", required=True)
    parser.add_argument("--torsion-prior-jsonl", required=True)
    parser.add_argument("--safe373-plan", required=True)
    parser.add_argument("--phase2-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output_dir_already_exists:{output_dir}")
    dataset_root = Path(args.dataset_root).resolve()
    train_rows = read_jsonl(
        dataset_root / "02_leakage_safe_split" / "train.jsonl"
    )
    valid_rows = read_jsonl(
        dataset_root / "02_leakage_safe_split" / "valid.jsonl"
    )
    if len(train_rows) != FORMAL_TRAIN_PAIR_COUNT:
        raise ValueError("formal_train_pair_count_mismatch")
    if len(valid_rows) != FORMAL_VALID_PAIR_COUNT:
        raise ValueError("formal_valid_pair_count_mismatch")
    relation_index = _biological_relation_index(
        read_jsonl(dataset_root / "dependencies" / "biological_pairs.jsonl")
    )
    formal_pairs = (
        [_formal_pair(row, "train", relation_index) for row in train_rows]
        + [_formal_pair(row, "valid", relation_index) for row in valid_rows]
    )
    evidence = _evidence_index([
        Path(args.candidate_evidence_jsonl).resolve(),
        Path(args.expanded_evidence_jsonl).resolve(),
    ])
    torsion_groups, torsion_manifest = load_torsion_prior(
        Path(args.torsion_prior_jsonl).resolve(),
        Path(args.torsion_prior_manifest).resolve(),
    )
    if (
        torsion_manifest["manifest_canonical_sha256"]
        != TORSION_PRIOR_MANIFEST_SHA256
        or sha256_file(args.torsion_prior_jsonl) != TORSION_PRIOR_JSONL_SHA256
    ):
        raise ValueError("torsion_prior_contract_mismatch")
    registry = build_sequence_eligibility_registry(
        formal_pairs,
        evidence=evidence,
        mmcif_root=Path(args.mmcif_root).resolve(),
        qbiolip_root=Path(args.qbiolip_root).resolve(),
        biolip_root=Path(args.biolip_root).resolve(),
        torsion_groups=torsion_groups,
    )
    registry_by_sequence = {
        row["peptide_sequence"]: row for row in registry
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    registry_path = output_dir / "sequence_eligibility_registry.jsonl"
    write_jsonl(registry_path, registry)

    classification = "EXPLICIT_BOUNDED_PLAN_PASS"
    failure: str | None = None
    try:
        train_selected = select_eligible_pair_plan(
            formal_pairs,
            registry,
            split="train",
            target_count=MAX_TRAIN_PAIRS,
        )
        valid_selected = select_eligible_pair_plan(
            formal_pairs,
            registry,
            split="valid",
            target_count=MAX_VALID_PAIRS,
        )
    except RuntimeError as exc:
        classification = "ELIGIBLE_PLAN_CAPACITY_FAIL"
        failure = str(exc)
        train_selected = []
        valid_selected = []

    train_plan = _plan_payload(
        train_selected, registry_by_sequence, "train"
    )
    valid_plan = _plan_payload(
        valid_selected, registry_by_sequence, "valid"
    )
    write_json(output_dir / "train_interface_pair_plan.json", train_plan)
    write_json(output_dir / "valid_interface_pair_plan.json", valid_plan)

    train_pair_ids = train_plan["interface_pair_ids"]
    valid_pair_ids = valid_plan["interface_pair_ids"]
    train_sequences = train_plan["unique_peptide_sequences"]
    valid_sequences = valid_plan["unique_peptide_sequences"]
    required_sequences = sorted(set(train_sequences) | set(valid_sequences))
    safe_plan_path = Path(args.safe373_plan).resolve()
    safe_plan = read_json(safe_plan_path)
    plans = {
        "schema_version": PLAN_SCHEMA,
        "target_train_pair_count": MAX_TRAIN_PAIRS,
        "target_valid_pair_count": MAX_VALID_PAIRS,
        "selection_policy": (
            "filter by frozen sequence eligibility; sort eligible "
            "interface_pair_id lexicographically; take exact target count"
        ),
        "train_interface_pair_ids": train_pair_ids,
        "train_interface_pair_ids_sha256": sequence_sha256(train_pair_ids),
        "valid_interface_pair_ids": valid_pair_ids,
        "valid_interface_pair_ids_sha256": sequence_sha256(valid_pair_ids),
        "train_unique_peptide_sequences": train_sequences,
        "train_unique_peptide_sequences_sha256": sequence_sha256(
            train_sequences
        ),
        "valid_unique_peptide_sequences": valid_sequences,
        "valid_unique_peptide_sequences_sha256": sequence_sha256(
            valid_sequences
        ),
    }
    core = {
        "schema_version": CONTRACT_SCHEMA,
        "initialization": {
            "role": "phase2_learned_concat_baseline",
            "checkpoint_sha256": PHASE2_INITIALIZATION_SHA256,
        },
        "source_policy": {
            "training_source_split": "formal_train_only",
            "chemistry_classification": "ordinary_linear_standard",
            "evaluation_cache_used_for_training": False,
            "target_bound_generation_inputs_used": False,
            "generation_seed_inputs": ALLOWED_GENERATION_INPUTS,
        },
        "generator": {
            "torsion_prior_manifest_sha256": TORSION_PRIOR_MANIFEST_SHA256,
            "torsion_prior_jsonl_sha256": TORSION_PRIOR_JSONL_SHA256,
            "backbone_contract": "train-only-residue-context-trans-only-v1",
            "sidechain_packer": "FASPR-fixed-backbone",
            "canonical_topology_contract": CANONICAL_TOPOLOGY_CONTRACT,
            "max_attempts_per_slot": 25,
            "nonlocal_clash_threshold_angstrom": 0.75,
            "candidate_independent": True,
            "faspr_source_commit": FASPR_SOURCE_COMMIT,
            "faspr_binary_sha256": FASPR_BINARY_SHA256,
            "faspr_rotamer_library_sha256": (
                FASPR_ROTAMER_LIBRARY_SHA256
            ),
            "generator_version": GENERATOR_VERSION,
        },
        "eligibility_registry": {
            "schema_version": ELIGIBILITY_REGISTRY_SCHEMA,
            "path": registry_path.name,
            "file_sha256": sha256_file(registry_path),
            "canonical_sha256": canonical_json_sha256(registry),
            "sequence_count": len(registry),
            "formal_train_pair_count": len(train_rows),
            "formal_valid_pair_count": len(valid_rows),
        },
        "plans": plans,
        "evaluation_exclusion": {
            "safe373_plan_path": str(safe_plan_path),
            "safe373_plan_file_sha256": sha256_file(safe_plan_path),
            "safe373_plan_canonical_sha256": safe_plan[
                "plan_canonical_sha256"
            ],
        },
        "cache": {
            "status": "required_not_materialized",
            "schema_version": CACHE_SCHEMA,
            "purpose": "bounded_train_valid_only",
            "index_path": None,
            "index_sha256": None,
            "conformers_per_sequence": 10,
            "atom_cap_exclusive": ATOM_CAP_EXCLUSIVE,
            "required_peptide_sequences": required_sequences,
            "required_peptide_sequences_sha256": sequence_sha256(
                required_sequences
            ),
        },
    }
    manifest = {
        **core,
        "manifest_canonical_sha256": canonical_json_sha256(core),
    }
    manifest_path = output_dir / "full_heavy_adaptation_manifest.json"
    write_json(manifest_path, manifest)

    train_sequences_by_pair = {
        pair["interface_pair_id"]: pair["peptide_sequence"]
        for pair in formal_pairs if pair["split"] == "train"
    }
    valid_sequences_by_pair = {
        pair["interface_pair_id"]: pair["peptide_sequence"]
        for pair in formal_pairs if pair["split"] == "valid"
    }
    train_relations_by_pair = {
        pair["interface_pair_id"]: pair["biological_pair_id"]
        for pair in formal_pairs if pair["split"] == "train"
    }
    valid_relations_by_pair = {
        pair["interface_pair_id"]: pair["biological_pair_id"]
        for pair in formal_pairs if pair["split"] == "valid"
    }
    validation: dict[str, Any] | None = None
    if classification == "EXPLICIT_BOUNDED_PLAN_PASS":
        try:
            validation = validate_explicit_bounded_plan_contract(
                manifest_path,
                phase2_checkpoint=Path(args.phase2_checkpoint).resolve(),
                train_interface_pair_ids=train_sequences_by_pair,
                valid_interface_pair_ids=valid_sequences_by_pair,
                train_sequence_by_pair=train_sequences_by_pair,
                valid_sequence_by_pair=valid_sequences_by_pair,
                train_relation_by_pair=train_relations_by_pair,
                valid_relation_by_pair=valid_relations_by_pair,
            )
        except Exception as exc:
            classification = "PLAN_IMPLEMENTATION_FAIL"
            failure = f"{type(exc).__name__}:{exc}"

    safe_pair_ids = {
        str(value)
        for value in safe_plan.get("safe_query_interface_pair_ids", [])
    }
    safe_sequences = {
        str(value)
        for value in safe_plan.get("safe_peptide_candidate_ids", [])
    }
    safe_relations = {
        valid_relations_by_pair[pair_id]
        for pair_id in safe_pair_ids
        if pair_id in valid_relations_by_pair
    }
    selected_train_relations = {
        train_relations_by_pair[pair_id] for pair_id in train_pair_ids
    }
    report = {
        "schema_version": "phase3-v2-explicit-bounded-plan-validation-v1",
        "status": (
            "PASS" if classification == "EXPLICIT_BOUNDED_PLAN_PASS"
            else "FAIL"
        ),
        "classification": classification,
        "failure": failure,
        "formal_counts": {
            "train_pairs": len(train_rows),
            "valid_pairs": len(valid_rows),
            "train_unique_sequences": len({
                pair["peptide_sequence"]
                for pair in formal_pairs if pair["split"] == "train"
            }),
            "valid_unique_sequences": len({
                pair["peptide_sequence"]
                for pair in formal_pairs if pair["split"] == "valid"
            }),
        },
        "eligibility": {
            "sequence_count": len(registry),
            "eligible_sequence_count": sum(
                bool(row["eligible"]) for row in registry
            ),
            "ineligible_sequence_count": sum(
                not bool(row["eligible"]) for row in registry
            ),
            "eligible_pair_counts": {
                split: sum(
                    row["pair_count"] for row in registry
                    if row["split"] == split and row["eligible"]
                )
                for split in ("train", "valid")
            },
            "sequence_classification_counts": dict(sorted(Counter(
                row["chemistry_classification"] for row in registry
            ).items())),
            "exclusion_reason_counts": dict(sorted(Counter(
                reason
                for row in registry
                for reason in row["exclusion_reasons"]
            ).items())),
        },
        "plans": {
            "train_pair_count": len(train_pair_ids),
            "valid_pair_count": len(valid_pair_ids),
            "train_pair_sha256": sequence_sha256(train_pair_ids),
            "valid_pair_sha256": sequence_sha256(valid_pair_ids),
            "train_unique_sequence_count": len(train_sequences),
            "valid_unique_sequence_count": len(valid_sequences),
            "train_unique_sequence_sha256": sequence_sha256(train_sequences),
            "valid_unique_sequence_sha256": sequence_sha256(valid_sequences),
            "manifest_canonical_sha256": manifest[
                "manifest_canonical_sha256"
            ],
            "old_unfiltered_prefix_retained": False,
        },
        "leakage": {
            "train_valid_pair_intersection": len(
                set(train_pair_ids) & set(valid_pair_ids)
            ),
            "train_valid_relation_intersection": len(
                selected_train_relations & {
                    valid_relations_by_pair[pair_id]
                    for pair_id in valid_pair_ids
                }
            ),
            "train_valid_sequence_intersection": len(
                set(train_sequences) & set(valid_sequences)
            ),
            "train_safe373_pair_intersection": len(
                set(train_pair_ids) & safe_pair_ids
            ),
            "train_safe373_relation_intersection": len(
                selected_train_relations & safe_relations
            ),
            "train_safe373_sequence_intersection": len(
                set(train_sequences) & safe_sequences
            ),
        },
        "cache": {
            "status": "required_not_materialized",
            "generated": False,
            "required_unique_sequence_count": len(required_sequences),
            "required_conformer_count": len(required_sequences) * 10,
        },
        "validator_result": (
            None if validation is None else {
                key: value for key, value in validation.items()
                if key not in {"registry_by_sequence", "cache_contract"}
            }
        ),
        "forbidden_actions": {
            "faspr_invoked": False,
            "conformer_generated": False,
            "cache_generated": False,
            "optimizer_created": False,
            "backward_executed": False,
            "training_executed": False,
            "gpu_retrieval_executed": False,
        },
    }
    write_json(output_dir / "plan_validation_report.json", report)
    summary = (
        "# Phase-3 v2 explicit bounded full-heavy plan\n\n"
        f"- Classification: `{classification}`\n"
        f"- Full formal split audit: train {len(train_rows)}, "
        f"valid {len(valid_rows)} pairs.\n"
        f"- Eligible pairs: train "
        f"{report['eligibility']['eligible_pair_counts']['train']}, valid "
        f"{report['eligibility']['eligible_pair_counts']['valid']}.\n"
        f"- Selected plan: train {len(train_pair_ids)}, valid "
        f"{len(valid_pair_ids)}.\n"
        f"- Ordered pair SHA256: train "
        f"`{sequence_sha256(train_pair_ids)}`, valid "
        f"`{sequence_sha256(valid_pair_ids)}`.\n"
        f"- Required future cache: {len(required_sequences)} unique "
        f"sequences / {len(required_sequences) * 10} conformers.\n"
        "- Cache status: `required_not_materialized`; no conformers or cache "
        "were generated.\n"
        "- FASPR, optimizer, backward, training, and GPU retrieval were not "
        "run.\n"
    )
    if failure:
        summary += f"- Failure: `{failure}`\n"
    _atomic_text(output_dir / "summary.md", summary)
    print(json.dumps({
        "classification": classification,
        "output_dir": str(output_dir),
        "train_pair_count": len(train_pair_ids),
        "valid_pair_count": len(valid_pair_ids),
        "manifest_canonical_sha256": manifest[
            "manifest_canonical_sha256"
        ],
    }, sort_keys=True))
    if classification != "EXPLICIT_BOUNDED_PLAN_PASS":
        raise SystemExit(2)


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
            "histogram": {},
        }

    def percentile(fraction: float) -> int:
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "histogram": {
            str(key): value
            for key, value in sorted(Counter(ordered).items())
        },
    }


def _summarize_rows(
    pairs: list[dict[str, Any]],
    registry_by_sequence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sequences = {row["peptide_sequence"] for row in pairs}
    relations = {row["biological_pair_id"] for row in pairs}
    primary_sources = Counter(
        row["source_database"] or "unspecified" for row in pairs
    )
    all_sources = Counter(
        source
        for row in pairs
        for source in (
            row["source_databases"]
            or [row["source_database"] or "unspecified"]
        )
    )
    return {
        "pair_count": len(pairs),
        "unique_sequence_count": len(sequences),
        "biological_relation_count": len(relations),
        "primary_source_database_pair_counts": dict(
            sorted(primary_sources.items())
        ),
        "all_source_database_pair_memberships": dict(
            sorted(all_sources.items())
        ),
        "pair_weighted_peptide_length_distribution": _distribution(
            len(row["peptide_sequence"]) for row in pairs
        ),
        "unique_sequence_length_distribution": _distribution(
            len(sequence) for sequence in sequences
        ),
        "theoretical_heavy_atom_distribution": _distribution(
            registry_by_sequence[sequence]["theoretical_heavy_atom_count"]
            for sequence in sequences
        ),
    }


def _classification_report(
    formal_pairs: list[dict[str, Any]],
    registry_by_sequence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for classification in CHEMISTRY_PRECEDENCE:
        pairs = [
            pair for pair in formal_pairs
            if registry_by_sequence[pair["peptide_sequence"]][
                "chemistry_classification"
            ] == classification
        ]
        output[classification] = _summarize_rows(
            pairs, registry_by_sequence
        )
    return output


def _split_eligible_report(
    formal_pairs: list[dict[str, Any]],
    registry_by_sequence: dict[str, dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    original = [row for row in formal_pairs if row["split"] == split]
    eligible = [
        row for row in original
        if registry_by_sequence[row["peptide_sequence"]]["eligible"]
    ]
    summary = _summarize_rows(eligible, registry_by_sequence)
    summary["original_pair_count"] = len(original)
    summary["original_unique_sequence_count"] = len({
        row["peptide_sequence"] for row in original
    })
    summary["original_biological_relation_count"] = len({
        row["biological_pair_id"] for row in original
    })
    summary["pair_coverage_fraction"] = len(eligible) / len(original)
    summary["sequence_coverage_fraction"] = (
        summary["unique_sequence_count"]
        / summary["original_unique_sequence_count"]
    )
    summary["relation_coverage_fraction"] = (
        summary["biological_relation_count"]
        / summary["original_biological_relation_count"]
    )
    return summary


def _top_exclusion_concentrations(
    formal_pairs: list[dict[str, Any]],
    registry_by_sequence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    excluded = [
        row for row in formal_pairs
        if not registry_by_sequence[row["peptide_sequence"]]["eligible"]
    ]
    sequence_counts = Counter(row["peptide_sequence"] for row in excluded)
    source_counts = Counter(
        row["source_database"] or "unspecified" for row in excluded
    )
    split_counts = Counter(row["split"] for row in excluded)
    receptor_family_counts = Counter(
        row["receptor_family"] or "unspecified" for row in excluded
    )
    return {
        "note": (
            "No frozen peptide-family label exists. Exact peptide sequence "
            "multiplicity is the conservative peptide-family proxy; "
            "receptor_family is reported separately and is not relabeled as "
            "a peptide family."
        ),
        "top_exact_sequence_pair_counts": [
            {"peptide_sequence": key, "pair_count": value}
            for key, value in sequence_counts.most_common(20)
        ],
        "top_primary_source_pair_counts": [
            {"source_database": key, "pair_count": value}
            for key, value in source_counts.most_common()
        ],
        "split_pair_counts": dict(sorted(split_counts.items())),
        "top_receptor_family_pair_counts": [
            {"receptor_family": key, "pair_count": value}
            for key, value in receptor_family_counts.most_common(20)
        ],
        "excluded_pair_length_distribution": _distribution(
            len(row["peptide_sequence"]) for row in excluded
        ),
    }


def _covalent_evidence_summary(
    registry: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_rows = [
        evidence
        for row in registry
        if row["chemistry_classification"] == "receptor_covalent"
        for instance in row["evidence_instances"]
        for evidence in instance["evidence_instances"]
        if evidence["chemistry_classification"] == "receptor_covalent"
    ]
    explicit = sum(
        bool(row["peptide_receptor_explicit_connections"])
        for row in evidence_rows
    )
    geometry = sum(
        bool(row["peptide_other_covalent_geometry"])
        for row in evidence_rows
    )
    return {
        "structure_evidence_count": len(evidence_rows),
        "explicit_connection_record_count": explicit,
        "covalent_geometry_evidence_count": geometry,
        "all_have_explicit_or_geometry_evidence": all(
            bool(row["peptide_receptor_explicit_connections"])
            or bool(row["peptide_other_covalent_geometry"])
            for row in evidence_rows
        ),
        "task_boundary": (
            "Excluded from the current non-covalent peptide-receptor "
            "retrieval contract because the observed target relation includes "
            "a covalent peptide-receptor bond supported by an explicit "
            "connection record and/or covalent-distance geometry."
        ),
    }


def _coverage_scenarios(
    formal_pairs: list[dict[str, Any]],
    registry_by_sequence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    immediate = [
        row for row in formal_pairs
        if registry_by_sequence[row["peptide_sequence"]]["eligible"]
    ]
    disulfide_cyclic = [
        row for row in formal_pairs
        if (
            registry_by_sequence[row["peptide_sequence"]][
                "chemistry_classification"
            ] in {"known_disulfide", "cyclic_or_crosslinked"}
            and registry_by_sequence[row["peptide_sequence"]][
                "theoretical_heavy_atom_count"
            ] < ATOM_CAP_EXCLUSIVE
            and registry_by_sequence[row["peptide_sequence"]][
                "torsion_prior_covered"
            ]
        )
    ]
    metadata_insufficient = [
        row for row in formal_pairs
        if registry_by_sequence[row["peptide_sequence"]][
            "chemistry_classification"
        ] in {"chemistry_insufficient", "multiple_cys_unknown"}
    ]
    total = len(formal_pairs)
    return {
        "immediately_supported": {
            "pair_count": len(immediate),
            "fraction": len(immediate) / total,
        },
        "theoretically_supported_after_disulfide_cyclic_generators": {
            "pair_count": len(immediate) + len(disulfide_cyclic),
            "fraction": (len(immediate) + len(disulfide_cyclic)) / total,
            "incremental_known_disulfide_or_cyclic_pair_count": len(
                disulfide_cyclic
            ),
            "scope_note": (
                "Upper bound assuming explicit disulfide/cyclic chemistry "
                "can be parameterized; it does not include unknown multiple-"
                "Cys, modified residues, receptor-covalent targets, or atom-"
                "cap failures."
            ),
        },
        "still_blocked_by_insufficient_metadata": {
            "pair_count": len(metadata_insufficient),
            "fraction": len(metadata_insufficient) / total,
            "categories": [
                "chemistry_insufficient",
                "multiple_cys_unknown",
            ],
        },
    }


def build_audit_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit full formal train/valid sequence-level full-heavy "
            "chemistry applicability without selecting a bounded plan."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--candidate-evidence-jsonl", required=True)
    parser.add_argument("--expanded-evidence-jsonl", required=True)
    parser.add_argument("--mmcif-root", required=True)
    parser.add_argument("--qbiolip-root", required=True)
    parser.add_argument("--biolip-root", required=True)
    parser.add_argument("--torsion-prior-manifest", required=True)
    parser.add_argument("--torsion-prior-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_audit_argparser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output_dir_already_exists:{output_dir}")
    dataset_root = Path(args.dataset_root).resolve()
    train_rows = read_jsonl(
        dataset_root / "02_leakage_safe_split" / "train.jsonl"
    )
    valid_rows = read_jsonl(
        dataset_root / "02_leakage_safe_split" / "valid.jsonl"
    )
    if len(train_rows) != FORMAL_TRAIN_PAIR_COUNT:
        raise ValueError("formal_train_pair_count_mismatch")
    if len(valid_rows) != FORMAL_VALID_PAIR_COUNT:
        raise ValueError("formal_valid_pair_count_mismatch")
    relation_index = _biological_relation_index(
        read_jsonl(dataset_root / "dependencies" / "biological_pairs.jsonl")
    )
    formal_pairs = (
        [_formal_pair(row, "train", relation_index) for row in train_rows]
        + [_formal_pair(row, "valid", relation_index) for row in valid_rows]
    )
    evidence = _evidence_index([
        Path(args.candidate_evidence_jsonl).resolve(),
        Path(args.expanded_evidence_jsonl).resolve(),
    ])
    torsion_groups, torsion_manifest = load_torsion_prior(
        Path(args.torsion_prior_jsonl).resolve(),
        Path(args.torsion_prior_manifest).resolve(),
    )
    if (
        torsion_manifest["manifest_canonical_sha256"]
        != TORSION_PRIOR_MANIFEST_SHA256
        or sha256_file(args.torsion_prior_jsonl)
        != TORSION_PRIOR_JSONL_SHA256
    ):
        raise ValueError("torsion_prior_contract_mismatch")
    registry = build_sequence_eligibility_registry(
        formal_pairs,
        evidence=evidence,
        mmcif_root=Path(args.mmcif_root).resolve(),
        qbiolip_root=Path(args.qbiolip_root).resolve(),
        biolip_root=Path(args.biolip_root).resolve(),
        torsion_groups=torsion_groups,
    )
    registry_by_sequence = {
        row["peptide_sequence"]: row for row in registry
    }
    classification_reports = {
        split: _classification_report(
            [row for row in formal_pairs if row["split"] == split],
            registry_by_sequence,
        )
        for split in ("train", "valid")
    }
    classification_reports["combined"] = _classification_report(
        formal_pairs, registry_by_sequence
    )
    eligible = {
        split: _split_eligible_report(
            formal_pairs, registry_by_sequence, split
        )
        for split in ("train", "valid")
    }
    eligible["combined"] = _summarize_rows(
        [
            row for row in formal_pairs
            if registry_by_sequence[row["peptide_sequence"]]["eligible"]
        ],
        registry_by_sequence,
    )
    atom_cap_sequences = {
        row["peptide_sequence"] for row in registry
        if row["theoretical_heavy_atom_count"] >= ATOM_CAP_EXCLUSIVE
    }
    atom_cap_pairs = [
        row for row in formal_pairs
        if row["peptide_sequence"] in atom_cap_sequences
    ]
    atom_cap_relations = {
        row["biological_pair_id"] for row in atom_cap_pairs
    }
    max_atom_count = max(
        row["theoretical_heavy_atom_count"] for row in registry
    )
    atom_cap_report = {
        "atom_cap_exclusive": ATOM_CAP_EXCLUSIVE,
        "pair_count": len(atom_cap_pairs),
        "unique_sequence_count": len(atom_cap_sequences),
        "biological_relation_count": len(atom_cap_relations),
        "fraction_of_all_pairs": len(atom_cap_pairs) / len(formal_pairs),
        "maximum_theoretical_heavy_atom_count": max_atom_count,
        "maximum_atom_sequences": sorted(
            row["peptide_sequence"] for row in registry
            if row["theoretical_heavy_atom_count"] == max_atom_count
        ),
        "assessment": (
            "formal_atom_cap_change_required"
            if len(atom_cap_pairs) / len(formal_pairs) >= 0.01
            else "rare_outliers_can_be_excluded_from_first_bounded_contract"
        ),
    }
    immediate_fraction = (
        eligible["combined"]["pair_count"] / len(formal_pairs)
    )
    if (
        eligible["train"]["pair_count"] >= 4096
        and eligible["valid"]["pair_count"] >= 512
        and immediate_fraction >= 0.70
    ):
        classification = "CORE_LINEAR_SUBSET_SUFFICIENT"
    elif immediate_fraction >= 0.40:
        classification = "CORE_LINEAR_SUBSET_TOO_NARROW"
    else:
        classification = "FULL_DATA_CONTRACT_REDESIGN_REQUIRED"
    recovery_requirements = {
        "modified_or_nonstandard": (
            "Exact residue/component identities, covalent topology, protonation/"
            "terminal state, force-field parameters, and atom naming; a single-"
            "letter canonical sequence is insufficient."
        ),
        "known_disulfide": (
            "Explicit Cys-Cys pairing and disulfide connectivity, including "
            "bonded geometry and whether inter-chain/receptor-linked."
        ),
        "cyclic_or_crosslinked": (
            "Explicit closure/crosslink partners, bond atom names, topology, "
            "terminal chemistry, and ring-aware conformer generation."
        ),
        "multiple_cys_unknown": (
            "Experimental disulfide pairing/state or evidence that all Cys are "
            "reduced; sequence alone cannot choose connectivity."
        ),
        "chemistry_insufficient": (
            "Unambiguous residue identity, complete connectivity records, "
            "terminal state, and a structure/evidence record resolving whether "
            "the peptide is linear, cyclic, crosslinked, or covalent."
        ),
        "receptor_covalent": (
            "Not recoverable into the current non-covalent retrieval task by "
            "metadata completion; it requires a separately defined covalent-"
            "complex task and generation/scoring contract."
        ),
    }
    report = {
        "schema_version": AUDIT_SCHEMA,
        "status": "PASS",
        "classification": classification,
        "formal_split_counts": {
            "train_pairs": len(train_rows),
            "valid_pairs": len(valid_rows),
            "combined_pairs": len(formal_pairs),
            "train_unique_sequences": len({
                row["peptide_sequence"] for row in formal_pairs
                if row["split"] == "train"
            }),
            "valid_unique_sequences": len({
                row["peptide_sequence"] for row in formal_pairs
                if row["split"] == "valid"
            }),
        },
        "classification_statistics": classification_reports,
        "ordinary_linear_standard_atom_lt_192_and_prior_covered": eligible,
        "exclusion_concentration": _top_exclusion_concentrations(
            formal_pairs, registry_by_sequence
        ),
        "receptor_covalent_task_boundary": _covalent_evidence_summary(
            registry
        ),
        "chemistry_recovery_requirements": recovery_requirements,
        "atom_cap_audit": atom_cap_report,
        "coverage_scenarios": _coverage_scenarios(
            formal_pairs, registry_by_sequence
        ),
        "eligibility_policy": {
            "sequence_level_conservative": True,
            "all_structure_instances_must_be_ordinary_linear_standard": True,
            "atom_count_must_be_strictly_below": ATOM_CAP_EXCLUSIVE,
            "torsion_prior_must_cover_every_context": True,
            "bound_coordinates_used_for_generation": False,
            "plan_selected": False,
            "manifest_created": False,
            "cache_generated": False,
            "training_executed": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    registry_path = output_dir / "sequence_eligibility_registry.jsonl"
    write_jsonl(registry_path, registry)
    report["registry"] = {
        "path": registry_path.name,
        "sequence_count": len(registry),
        "file_sha256": sha256_file(registry_path),
        "canonical_sha256": canonical_json_sha256(registry),
    }
    write_json(output_dir / "full_split_eligibility_report.json", report)
    coverage = report["coverage_scenarios"]
    summary = (
        "# Phase-3 v2 full formal-split full-heavy eligibility audit\n\n"
        f"- Classification: `{classification}`\n"
        f"- Audited: train {len(train_rows):,} pairs; valid "
        f"{len(valid_rows):,} pairs.\n"
        f"- Immediate ordinary-linear support: "
        f"{eligible['combined']['pair_count']:,}/{len(formal_pairs):,} "
        f"pairs ({coverage['immediately_supported']['fraction']:.2%}).\n"
        f"- Train eligible: {eligible['train']['pair_count']:,} pairs / "
        f"{eligible['train']['unique_sequence_count']:,} sequences / "
        f"{eligible['train']['biological_relation_count']:,} relations.\n"
        f"- Valid eligible: {eligible['valid']['pair_count']:,} pairs / "
        f"{eligible['valid']['unique_sequence_count']:,} sequences / "
        f"{eligible['valid']['biological_relation_count']:,} relations.\n"
        f"- Atom-cap failures: {atom_cap_report['pair_count']:,} pairs / "
        f"{atom_cap_report['unique_sequence_count']:,} sequences; maximum "
        f"{max_atom_count} heavy atoms.\n"
        "- No 4096/512 plan, training/cache manifest, conformer cache, "
        "optimizer, training, or retrieval was created or run.\n"
    )
    _atomic_text(output_dir / "summary.md", summary)
    print(json.dumps({
        "classification": classification,
        "output_dir": str(output_dir),
        "registry_sequence_count": len(registry),
        "immediate_supported_pair_count": eligible["combined"]["pair_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

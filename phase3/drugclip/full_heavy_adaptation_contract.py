"""Strict Phase-3 v2 bounded full-heavy adaptation contracts.

This module does not generate conformers.  It validates a future bounded
train/validation cache and exposes it as a read-only view over the formal v3
interface-pair dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from phase2.pepclip.data import ATOM_NAME_TO_ID, ELEMENT_TO_ID, RESIDUE_NAME_TO_ID
from phase2.pepclip.train_concat_fusion import PepCLIPConcatFusionModel, set_frozen, set_trainable
from phase3.drugclip.full_atom_conformer_prototype import conformer_atoms
from phase3.drugclip.standard_residue_topology import canonical_peptide_graph


CONTRACT_SCHEMA = "phase3-v2-bounded-full-heavy-adaptation-manifest-v1"
PLAN_SCHEMA = "phase3-v2-bounded-full-heavy-plan-v1"
ELIGIBILITY_REGISTRY_SCHEMA = (
    "phase3-v2-full-formal-split-sequence-eligibility-v1"
)
CACHE_SCHEMA = "phase3-v2-bounded-full-heavy-cache-v1"
CACHE_MANIFEST_SCHEMA = "phase3-v2-bounded-full-heavy-cache-manifest-v1"
PLAN_SELECTION_ALGORITHM_VERSION = (
    "phase3-v2-bounded-full-heavy-sha256-order-v1"
)
SAFE373_PLAN_CANONICAL_SHA256 = (
    "A32FF671CFEA0D1B858C8EFC58AD0E30D6F3170C670089238127B637FCC64310"
)
FREEZE_CONTRACT_VERSION = "phase3-v2-peptide-3d-last1-plus-peptide-fusion-v1"
PHASE2_INITIALIZATION_SHA256 = (
    "9FB16C48BA715C6273341609D60725AE796AD4A78771744E19ECF2C13D38AE20"
)
TORSION_PRIOR_MANIFEST_SHA256 = (
    "E93B24E59D5C18D7CC4213BC82D38C789CB32A279A3078AED738477246E80F94"
)
TORSION_PRIOR_JSONL_SHA256 = (
    "BB86912B86388CB757467D610A3EA706BE03D69A98561FA362E95B71A5F7B57B"
)
CANONICAL_TOPOLOGY_CONTRACT = "standard-pdb-heavy-atom-bond-templates-v1"
ALLOWED_GENERATION_INPUTS = [
    "generator_version",
    "torsion_prior_manifest_sha256",
    "peptide_sequence",
    "conformer_index",
    "attempt_index",
]
FORBIDDEN_GENERATION_INPUTS = {
    "receptor",
    "contact",
    "interface",
    "evidence",
    "bound_coordinates",
    "bound_pose",
}
MAX_TRAIN_PAIRS = 4096
MAX_VALID_PAIRS = 512
CONFORMERS_PER_SEQUENCE = 10
ATOM_CAP_EXCLUSIVE = 192


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def sequence_sha256(values: Iterable[str]) -> str:
    return canonical_json_sha256([str(value) for value in values])


def bounded_plan_selection_key(split: str, interface_pair_id: str) -> str:
    if split not in {"train", "valid"}:
        raise ValueError(f"invalid_bounded_plan_split:{split}")
    payload = (
        PLAN_SCHEMA
        + "\0"
        + split
        + "\0"
        + str(interface_pair_id)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected_jsonl_object:{path}:{line_number}")
            rows.append(value)
    return rows


def _resolve(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _validate_manifest_core(
    manifest_path: Path,
    phase2_checkpoint: str | Path,
) -> tuple[dict[str, Any], str]:
    manifest = _read_json(manifest_path)
    recorded_sha = str(manifest.get("manifest_canonical_sha256") or "")
    core = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_canonical_sha256"
    }
    if canonical_json_sha256(core) != recorded_sha:
        raise ValueError("full_heavy_manifest_canonical_sha256_mismatch")
    if manifest.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("unsupported_full_heavy_adaptation_contract")
    initialization = manifest.get("initialization", {})
    if initialization.get("role") != "phase2_learned_concat_baseline":
        raise ValueError("full_heavy_initialization_must_be_phase2_learned_concat")
    checkpoint_sha = sha256_file(phase2_checkpoint)
    if (
        checkpoint_sha != PHASE2_INITIALIZATION_SHA256
        or initialization.get("checkpoint_sha256") != PHASE2_INITIALIZATION_SHA256
    ):
        raise ValueError("full_heavy_phase2_initialization_sha256_mismatch")
    policy = manifest.get("source_policy", {})
    required_policy = {
        "training_source_split": "formal_train_only",
        "chemistry_classification": "ordinary_linear_standard",
        "evaluation_cache_used_for_training": False,
        "target_bound_generation_inputs_used": False,
    }
    if any(policy.get(key) != value for key, value in required_policy.items()):
        raise ValueError("full_heavy_source_policy_mismatch")
    generation_inputs = list(policy.get("generation_seed_inputs", []))
    if generation_inputs != ALLOWED_GENERATION_INPUTS:
        raise ValueError("full_heavy_generation_seed_input_contract_mismatch")
    if FORBIDDEN_GENERATION_INPUTS.intersection(generation_inputs):
        raise ValueError("full_heavy_target_bound_generation_input")
    generator = manifest.get("generator", {})
    required_generator = {
        "torsion_prior_manifest_sha256": TORSION_PRIOR_MANIFEST_SHA256,
        "torsion_prior_jsonl_sha256": TORSION_PRIOR_JSONL_SHA256,
        "backbone_contract": "train-only-residue-context-trans-only-v1",
        "sidechain_packer": "FASPR-fixed-backbone",
        "canonical_topology_contract": CANONICAL_TOPOLOGY_CONTRACT,
        "max_attempts_per_slot": 25,
        "nonlocal_clash_threshold_angstrom": 0.75,
        "candidate_independent": True,
    }
    if any(generator.get(key) != value for key, value in required_generator.items()):
        raise ValueError("full_heavy_generator_contract_mismatch")
    for key in (
        "faspr_source_commit",
        "faspr_binary_sha256",
        "faspr_rotamer_library_sha256",
        "generator_version",
    ):
        if not str(generator.get(key) or ""):
            raise ValueError(f"full_heavy_generator_contract_missing:{key}")
    return manifest, checkpoint_sha


def validate_explicit_bounded_plan_contract(
    manifest_file: str | Path,
    *,
    phase2_checkpoint: str | Path,
    train_interface_pair_ids: Iterable[str],
    valid_interface_pair_ids: Iterable[str],
    train_sequence_by_pair: dict[str, str],
    valid_sequence_by_pair: dict[str, str],
    train_relation_by_pair: dict[str, str] | None = None,
    valid_relation_by_pair: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate a manifest-owned plan against complete formal split membership."""

    manifest_path = Path(manifest_file).resolve()
    manifest, checkpoint_sha = _validate_manifest_core(
        manifest_path, phase2_checkpoint
    )
    formal_train = [str(value) for value in train_interface_pair_ids]
    formal_valid = [str(value) for value in valid_interface_pair_ids]
    if len(formal_train) != len(set(formal_train)):
        raise ValueError("duplicate_formal_train_interface_pair_id")
    if len(formal_valid) != len(set(formal_valid)):
        raise ValueError("duplicate_formal_valid_interface_pair_id")
    if set(formal_train) & set(formal_valid):
        raise ValueError("formal_train_valid_interface_overlap")
    if set(train_sequence_by_pair) != set(formal_train):
        raise ValueError("formal_train_sequence_mapping_mismatch")
    if set(valid_sequence_by_pair) != set(formal_valid):
        raise ValueError("formal_valid_sequence_mapping_mismatch")

    registry_contract = manifest.get("eligibility_registry", {})
    if registry_contract.get("schema_version") != ELIGIBILITY_REGISTRY_SCHEMA:
        raise ValueError("full_heavy_eligibility_registry_schema_mismatch")
    registry_path = _resolve(
        manifest_path, str(registry_contract.get("path") or "")
    )
    if sha256_file(registry_path) != registry_contract.get("file_sha256"):
        raise ValueError("full_heavy_eligibility_registry_file_sha256_mismatch")
    registry_rows = _read_jsonl(registry_path)
    if (
        canonical_json_sha256(registry_rows)
        != registry_contract.get("canonical_sha256")
    ):
        raise ValueError("full_heavy_eligibility_registry_canonical_sha256_mismatch")
    registry_by_sequence: dict[str, dict[str, Any]] = {}
    all_pair_ids: set[str] = set()
    for row in registry_rows:
        sequence = str(row.get("peptide_sequence") or "")
        if not sequence or sequence in registry_by_sequence:
            raise ValueError(f"duplicate_or_empty_eligibility_sequence:{sequence}")
        split = str(row.get("split") or "")
        if split not in {"train", "valid"}:
            raise ValueError(f"eligibility_registry_split_invalid:{sequence}")
        pair_ids = [str(value) for value in row.get("interface_pair_ids", [])]
        expected_mapping = (
            train_sequence_by_pair if split == "train" else valid_sequence_by_pair
        )
        expected_pairs = sorted(
            pair_id
            for pair_id, mapped_sequence in expected_mapping.items()
            if mapped_sequence == sequence
        )
        if pair_ids != expected_pairs:
            raise ValueError(f"eligibility_registry_pair_coverage_mismatch:{sequence}")
        if all_pair_ids.intersection(pair_ids):
            raise ValueError(f"eligibility_registry_duplicate_pair:{sequence}")
        all_pair_ids.update(pair_ids)
        chemistry = str(row.get("chemistry_classification") or "")
        structure_classes = [
            str(value) for value in row.get("structure_instance_classifications", [])
        ]
        all_ordinary = bool(structure_classes) and all(
            value == "ordinary_linear_standard" for value in structure_classes
        )
        atom_count = int(row.get("theoretical_heavy_atom_count", -1))
        torsion_covered = bool(row.get("torsion_prior_covered", False))
        expected_eligible = (
            chemistry == "ordinary_linear_standard"
            and all_ordinary
            and atom_count < ATOM_CAP_EXCLUSIVE
            and torsion_covered
        )
        if bool(row.get("eligible", False)) != expected_eligible:
            raise ValueError(f"eligibility_registry_decision_mismatch:{sequence}")
        registry_by_sequence[sequence] = row
    expected_all_pairs = set(formal_train) | set(formal_valid)
    if all_pair_ids != expected_all_pairs:
        raise ValueError("eligibility_registry_does_not_cover_formal_splits")
    expected_all_sequences = (
        set(train_sequence_by_pair.values()) | set(valid_sequence_by_pair.values())
    )
    if set(registry_by_sequence) != expected_all_sequences:
        raise ValueError("eligibility_registry_sequence_set_mismatch")

    plans = manifest.get("plans", {})
    if plans.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("full_heavy_plan_schema_mismatch")
    train_plan = [str(value) for value in plans.get("train_interface_pair_ids", [])]
    valid_plan = [str(value) for value in plans.get("valid_interface_pair_ids", [])]
    if len(train_plan) != len(set(train_plan)) or len(valid_plan) != len(set(valid_plan)):
        raise ValueError("full_heavy_plan_duplicate_interface_pair")
    if (
        plans.get("target_train_pair_count") != MAX_TRAIN_PAIRS
        or len(train_plan) != MAX_TRAIN_PAIRS
    ):
        raise ValueError("full_heavy_bounded_train_plan_count_violation")
    if (
        plans.get("target_valid_pair_count") != MAX_VALID_PAIRS
        or len(valid_plan) != MAX_VALID_PAIRS
    ):
        raise ValueError("full_heavy_bounded_valid_plan_count_violation")
    wrong_train = sorted(set(train_plan) - set(formal_train))
    wrong_valid = sorted(set(valid_plan) - set(formal_valid))
    if wrong_train or wrong_valid:
        raise ValueError(
            f"full_heavy_plan_wrong_split_id:train={wrong_train[:3]}:"
            f"valid={wrong_valid[:3]}"
        )
    if plans.get("train_interface_pair_ids_sha256") != sequence_sha256(train_plan):
        raise ValueError("full_heavy_train_plan_sha256_mismatch")
    if plans.get("valid_interface_pair_ids_sha256") != sequence_sha256(valid_plan):
        raise ValueError("full_heavy_valid_plan_sha256_mismatch")
    if set(train_plan) & set(valid_plan):
        raise ValueError("full_heavy_train_valid_interface_overlap")
    train_sequences = sorted({train_sequence_by_pair[pair_id] for pair_id in train_plan})
    valid_sequences = sorted({valid_sequence_by_pair[pair_id] for pair_id in valid_plan})
    if train_sequences != list(plans.get("train_unique_peptide_sequences", [])):
        raise ValueError("full_heavy_train_unique_sequence_plan_mismatch")
    if valid_sequences != list(plans.get("valid_unique_peptide_sequences", [])):
        raise ValueError("full_heavy_valid_unique_sequence_plan_mismatch")
    if plans.get("train_unique_peptide_sequences_sha256") != sequence_sha256(
        train_sequences
    ):
        raise ValueError("full_heavy_train_unique_sequence_sha256_mismatch")
    if plans.get("valid_unique_peptide_sequences_sha256") != sequence_sha256(
        valid_sequences
    ):
        raise ValueError("full_heavy_valid_unique_sequence_sha256_mismatch")
    if set(train_sequences) & set(valid_sequences):
        raise ValueError("full_heavy_train_valid_sequence_leakage")
    for sequence in train_sequences + valid_sequences:
        if not bool(registry_by_sequence[sequence].get("eligible", False)):
            raise ValueError(f"full_heavy_plan_contains_ineligible_sequence:{sequence}")

    train_relation_by_pair = train_relation_by_pair or {}
    valid_relation_by_pair = valid_relation_by_pair or {}
    if train_relation_by_pair or valid_relation_by_pair:
        if set(train_relation_by_pair) != set(formal_train):
            raise ValueError("formal_train_relation_mapping_mismatch")
        if set(valid_relation_by_pair) != set(formal_valid):
            raise ValueError("formal_valid_relation_mapping_mismatch")
        train_relations = {train_relation_by_pair[pair_id] for pair_id in train_plan}
        valid_relations = {valid_relation_by_pair[pair_id] for pair_id in valid_plan}
        if train_relations & valid_relations:
            raise ValueError("full_heavy_train_valid_relation_leakage")

    evaluation = manifest.get("evaluation_exclusion", {})
    safe_plan_path = _resolve(
        manifest_path, str(evaluation.get("safe373_plan_path") or "")
    )
    if sha256_file(safe_plan_path) != evaluation.get("safe373_plan_file_sha256"):
        raise ValueError("full_heavy_safe373_plan_sha256_mismatch")
    safe_plan = _read_json(safe_plan_path)
    if (
        str(safe_plan.get("plan_canonical_sha256") or "")
        != evaluation.get("safe373_plan_canonical_sha256")
    ):
        raise ValueError("full_heavy_safe373_plan_canonical_sha256_mismatch")
    safe_pairs = {
        str(value) for value in safe_plan.get("safe_query_interface_pair_ids", [])
    }
    safe_sequences = {
        str(value) for value in safe_plan.get("safe_peptide_candidate_ids", [])
    }
    if set(train_plan) & safe_pairs:
        raise ValueError("full_heavy_train_safe373_pair_leakage")
    if set(train_sequences) & safe_sequences:
        raise ValueError("full_heavy_train_safe373_sequence_leakage")
    if train_relation_by_pair and valid_relation_by_pair:
        safe_relations = {
            valid_relation_by_pair[pair_id]
            for pair_id in safe_pairs
            if pair_id in valid_relation_by_pair
        }
        train_relations = {train_relation_by_pair[pair_id] for pair_id in train_plan}
        if train_relations & safe_relations:
            raise ValueError("full_heavy_train_safe373_relation_leakage")

    required_sequences = sorted(set(train_sequences) | set(valid_sequences))
    cache = manifest.get("cache", {})
    if (
        cache.get("schema_version") != CACHE_SCHEMA
        or cache.get("purpose") != "bounded_train_valid_only"
        or int(cache.get("conformers_per_sequence", -1))
        != CONFORMERS_PER_SEQUENCE
        or int(cache.get("atom_cap_exclusive", -1)) != ATOM_CAP_EXCLUSIVE
    ):
        raise ValueError("full_heavy_cache_contract_mismatch")
    if cache.get("status") not in {"required_not_materialized", "materialized"}:
        raise ValueError("full_heavy_cache_status_invalid")
    if list(cache.get("required_peptide_sequences", [])) != required_sequences:
        raise ValueError("full_heavy_cache_required_sequence_plan_mismatch")
    if cache.get("required_peptide_sequences_sha256") != sequence_sha256(
        required_sequences
    ):
        raise ValueError("full_heavy_cache_required_sequence_sha256_mismatch")
    index_value = str(cache.get("index_path") or "").lower()
    if "safe265" in index_value or "safe373" in index_value:
        raise ValueError("full_heavy_evaluation_cache_path_forbidden")
    if cache.get("status") == "required_not_materialized" and (
        cache.get("index_path") is not None or cache.get("index_sha256") is not None
    ):
        raise ValueError("full_heavy_unmaterialized_cache_claim_mismatch")

    return {
        "schema_version": CONTRACT_SCHEMA,
        "manifest_path": str(manifest_path),
        "manifest_canonical_sha256": str(
            manifest["manifest_canonical_sha256"]
        ),
        "phase2_initialization_sha256": checkpoint_sha,
        "train_interface_pair_ids": train_plan,
        "valid_interface_pair_ids": valid_plan,
        "train_interface_pair_ids_sha256": sequence_sha256(train_plan),
        "valid_interface_pair_ids_sha256": sequence_sha256(valid_plan),
        "train_unique_peptide_sequences": train_sequences,
        "valid_unique_peptide_sequences": valid_sequences,
        "eligibility_registry_path": str(registry_path),
        "eligibility_registry_file_sha256": registry_contract["file_sha256"],
        "eligibility_registry_canonical_sha256": registry_contract[
            "canonical_sha256"
        ],
        "registry_by_sequence": registry_by_sequence,
        "generator_contract": manifest["generator"],
        "source_policy": manifest["source_policy"],
        "cache_contract": cache,
    }


def _resolve_relative(contract_path: Path, value: Any, label: str) -> Path:
    raw = str(value or "")
    candidate = Path(raw)
    if (
        not raw
        or candidate.is_absolute()
        or (len(raw) >= 2 and raw[1] == ":")
        or raw.startswith(("/", "\\"))
    ):
        raise ValueError(f"{label}_must_be_relative")
    return (contract_path.parent / candidate).resolve()


def _validate_no_absolute_path_fields(value: Any, prefix: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            if str(key).endswith("_path") and item is not None:
                raw = str(item)
                if (
                    Path(raw).is_absolute()
                    or (len(raw) >= 2 and raw[1] == ":")
                    or raw.startswith(("/", "\\"))
                ):
                    raise ValueError(
                        f"bounded_plan_absolute_path_forbidden:{child}"
                    )
            _validate_no_absolute_path_fields(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_no_absolute_path_fields(item, f"{prefix}[{index}]")


def _descriptor_core(descriptor: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in descriptor.items()
        if key != "descriptor_canonical_sha256"
    }


def _read_plan_file(
    descriptor_path: Path,
    contract: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    path = _resolve_relative(
        descriptor_path,
        contract.get("path"),
        f"{split}_plan_path",
    )
    if sha256_file(path) != str(contract.get("file_sha256") or ""):
        raise ValueError(f"{split}_plan_file_sha256_mismatch")
    payload = _read_json(path)
    if payload.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(f"{split}_plan_schema_mismatch")
    if payload.get("split") != split:
        raise ValueError(f"{split}_plan_split_mismatch")
    return payload


def validate_bounded_plan_descriptor(
    descriptor_file: str | Path,
    *,
    train_interface_pair_ids: Iterable[str],
    valid_interface_pair_ids: Iterable[str],
    train_sequence_by_pair: dict[str, str],
    valid_sequence_by_pair: dict[str, str],
    train_relation_by_pair: dict[str, str],
    valid_relation_by_pair: dict[str, str],
) -> dict[str, Any]:
    """Validate the frozen descriptor without requiring a conformer cache."""

    descriptor_path = Path(descriptor_file).resolve()
    descriptor = _read_json(descriptor_path)
    if descriptor.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported_bounded_plan_descriptor_schema")
    recorded_sha = str(
        descriptor.get("descriptor_canonical_sha256") or ""
    )
    if canonical_json_sha256(_descriptor_core(descriptor)) != recorded_sha:
        raise ValueError("bounded_plan_descriptor_canonical_sha256_mismatch")
    _validate_no_absolute_path_fields(descriptor)

    formal_train = [str(value) for value in train_interface_pair_ids]
    formal_valid = [str(value) for value in valid_interface_pair_ids]
    formal_train_set = set(formal_train)
    formal_valid_set = set(formal_valid)
    if len(formal_train) != len(formal_train_set):
        raise ValueError("duplicate_formal_train_interface_pair_id")
    if len(formal_valid) != len(formal_valid_set):
        raise ValueError("duplicate_formal_valid_interface_pair_id")
    if formal_train_set & formal_valid_set:
        raise ValueError("formal_train_valid_interface_overlap")
    mappings = (
        (train_sequence_by_pair, formal_train_set, "train_sequence"),
        (valid_sequence_by_pair, formal_valid_set, "valid_sequence"),
        (train_relation_by_pair, formal_train_set, "train_relation"),
        (valid_relation_by_pair, formal_valid_set, "valid_relation"),
    )
    for mapping, expected, label in mappings:
        if set(mapping) != expected:
            raise ValueError(f"formal_{label}_mapping_mismatch")
    if set(train_sequence_by_pair.values()) & set(
        valid_sequence_by_pair.values()
    ):
        raise ValueError("formal_train_valid_sequence_leakage")
    if set(train_relation_by_pair.values()) & set(
        valid_relation_by_pair.values()
    ):
        raise ValueError("formal_train_valid_relation_leakage")

    inputs = descriptor.get("frozen_inputs", {})
    registry_contract = inputs.get("eligibility_registry", {})
    if (
        registry_contract.get("schema_version")
        != ELIGIBILITY_REGISTRY_SCHEMA
    ):
        raise ValueError("eligibility_registry_schema_mismatch")
    registry_path = _resolve_relative(
        descriptor_path,
        registry_contract.get("path"),
        "eligibility_registry_path",
    )
    if sha256_file(registry_path) != str(
        registry_contract.get("file_sha256") or ""
    ):
        raise ValueError("eligibility_registry_file_sha256_mismatch")
    registry_rows = _read_jsonl(registry_path)
    if canonical_json_sha256(registry_rows) != str(
        registry_contract.get("canonical_sha256") or ""
    ):
        raise ValueError("eligibility_registry_canonical_sha256_mismatch")
    if int(registry_contract.get("sequence_count", -1)) != len(
        registry_rows
    ):
        raise ValueError("eligibility_registry_sequence_count_mismatch")

    audit_contract = inputs.get("full_split_audit_report", {})
    audit_path = _resolve_relative(
        descriptor_path,
        audit_contract.get("path"),
        "full_split_audit_report_path",
    )
    if sha256_file(audit_path) != str(
        audit_contract.get("file_sha256") or ""
    ):
        raise ValueError("full_split_audit_report_sha256_mismatch")
    audit_report = _read_json(audit_path)
    if audit_report.get("classification") != "CORE_LINEAR_SUBSET_SUFFICIENT":
        raise ValueError("full_split_audit_report_classification_mismatch")

    formal_sources = inputs.get("formal_split_sources", {})
    for split, expected_count in (
        ("train", len(formal_train)),
        ("valid", len(formal_valid)),
    ):
        source_contract = formal_sources.get(split, {})
        source_path = _resolve_relative(
            descriptor_path,
            source_contract.get("path"),
            f"formal_{split}_source_path",
        )
        if sha256_file(source_path) != str(
            source_contract.get("file_sha256") or ""
        ):
            raise ValueError(f"formal_{split}_source_sha256_mismatch")
        if int(source_contract.get("pair_count", -1)) != expected_count:
            raise ValueError(f"formal_{split}_source_count_mismatch")
    biological_contract = formal_sources.get("biological_relations", {})
    biological_path = _resolve_relative(
        descriptor_path,
        biological_contract.get("path"),
        "formal_biological_relations_path",
    )
    if sha256_file(biological_path) != str(
        biological_contract.get("file_sha256") or ""
    ):
        raise ValueError("formal_biological_relations_sha256_mismatch")

    eligibility = descriptor.get("eligibility_contract", {})
    expected_eligibility = {
        "rule_version": ELIGIBILITY_REGISTRY_SCHEMA,
        "required_chemistry_classification": "ordinary_linear_standard",
        "theoretical_heavy_atom_count_operator": "<",
        "theoretical_heavy_atom_count_limit": ATOM_CAP_EXCLUSIVE,
        "torsion_prior_coverage_required": True,
        "sequence_level_all_structure_instances_required": True,
    }
    if any(
        eligibility.get(key) != value
        for key, value in expected_eligibility.items()
    ):
        raise ValueError("bounded_plan_eligibility_contract_mismatch")

    registry_by_sequence: dict[str, dict[str, Any]] = {}
    registry_pair_ids: set[str] = set()
    eligible_pairs_by_split: dict[str, list[str]] = {
        "train": [],
        "valid": [],
    }
    for row in registry_rows:
        sequence = str(row.get("peptide_sequence") or "")
        split = str(row.get("split") or "")
        if not sequence or sequence in registry_by_sequence:
            raise ValueError(
                f"eligibility_registry_duplicate_or_empty_sequence:{sequence}"
            )
        if split not in {"train", "valid"}:
            raise ValueError(f"eligibility_registry_invalid_split:{sequence}")
        pair_ids = [str(value) for value in row.get("interface_pair_ids", [])]
        expected_mapping = (
            train_sequence_by_pair
            if split == "train"
            else valid_sequence_by_pair
        )
        expected_pairs = sorted(
            pair_id
            for pair_id, mapped_sequence in expected_mapping.items()
            if mapped_sequence == sequence
        )
        if pair_ids != expected_pairs:
            raise ValueError(
                f"eligibility_registry_pair_coverage_mismatch:{sequence}"
            )
        if registry_pair_ids.intersection(pair_ids):
            raise ValueError(
                f"eligibility_registry_duplicate_pair:{sequence}"
            )
        registry_pair_ids.update(pair_ids)
        structure_classes = [
            str(value)
            for value in row.get(
                "structure_instance_classifications", []
            )
        ]
        atom_count = int(row.get("theoretical_heavy_atom_count", -1))
        torsion_covered = bool(row.get("torsion_prior_covered", False))
        expected_eligible = (
            str(row.get("chemistry_classification") or "")
            == "ordinary_linear_standard"
            and bool(structure_classes)
            and all(
                value == "ordinary_linear_standard"
                for value in structure_classes
            )
            and atom_count < ATOM_CAP_EXCLUSIVE
            and torsion_covered
        )
        if bool(row.get("eligible", False)) != expected_eligible:
            raise ValueError(
                f"eligibility_registry_decision_mismatch:{sequence}"
            )
        if expected_eligible:
            eligible_pairs_by_split[split].extend(pair_ids)
        registry_by_sequence[sequence] = row
    if registry_pair_ids != formal_train_set | formal_valid_set:
        raise ValueError("eligibility_registry_formal_pair_coverage_mismatch")

    selection = descriptor.get("selection_contract", {})
    if (
        selection.get("algorithm_version")
        != PLAN_SELECTION_ALGORITHM_VERSION
        or selection.get("namespace") != PLAN_SCHEMA
        or selection.get("primary_key")
        != (
            'SHA256("phase3-v2-bounded-full-heavy-plan-v1" + "\\0" '
            '+ split + "\\0" + interface_pair_id)'
        )
        or selection.get("tie_breaker") != "interface_pair_id"
    ):
        raise ValueError("bounded_plan_selection_contract_mismatch")

    plan_files = descriptor.get("plan_files", {})
    inline_plans = descriptor.get("plans", {})
    plan_payloads: dict[str, dict[str, Any]] = {}
    for split, target_count, formal_set, sequence_by_pair in (
        (
            "train",
            MAX_TRAIN_PAIRS,
            formal_train_set,
            train_sequence_by_pair,
        ),
        (
            "valid",
            MAX_VALID_PAIRS,
            formal_valid_set,
            valid_sequence_by_pair,
        ),
    ):
        plan = _read_plan_file(
            descriptor_path, plan_files.get(split, {}), split
        )
        if plan != inline_plans.get(split):
            raise ValueError(f"{split}_plan_inline_file_mismatch")
        pair_ids = [
            str(value) for value in plan.get("interface_pair_ids", [])
        ]
        if len(pair_ids) != target_count or len(set(pair_ids)) != target_count:
            raise ValueError(f"{split}_plan_count_mismatch")
        if set(pair_ids) - formal_set:
            raise ValueError(f"{split}_plan_wrong_split_id")
        if plan.get("interface_pair_ids_sha256") != sequence_sha256(pair_ids):
            raise ValueError(f"{split}_plan_ordered_sha256_mismatch")
        expected_pair_ids = sorted(
            eligible_pairs_by_split[split],
            key=lambda pair_id: (
                bounded_plan_selection_key(split, pair_id),
                pair_id,
            ),
        )[:target_count]
        if len(expected_pair_ids) != target_count:
            raise ValueError(f"{split}_eligible_plan_capacity_fail")
        if pair_ids != expected_pair_ids:
            raise ValueError(f"{split}_plan_selection_mismatch")
        sequences = sorted({sequence_by_pair[pair_id] for pair_id in pair_ids})
        if plan.get("unique_peptide_sequences") != sequences:
            raise ValueError(f"{split}_plan_unique_sequence_mismatch")
        if plan.get("unique_peptide_sequences_sha256") != sequence_sha256(
            sequences
        ):
            raise ValueError(f"{split}_plan_unique_sequence_sha256_mismatch")
        records = list(plan.get("sequence_records", []))
        if [row.get("peptide_sequence") for row in records] != sequences:
            raise ValueError(f"{split}_plan_sequence_record_order_mismatch")
        pair_counts: dict[str, int] = {}
        for pair_id in pair_ids:
            sequence = sequence_by_pair[pair_id]
            pair_counts[sequence] = pair_counts.get(sequence, 0) + 1
        for record in records:
            sequence = str(record.get("peptide_sequence") or "")
            registry_row = registry_by_sequence.get(sequence)
            if not registry_row or not bool(registry_row.get("eligible", False)):
                raise ValueError(
                    f"{split}_plan_contains_ineligible_sequence:{sequence}"
                )
            atom_count = int(
                registry_row.get("theoretical_heavy_atom_count", -1)
            )
            if atom_count >= ATOM_CAP_EXCLUSIVE:
                raise ValueError(
                    f"{split}_plan_atom_cap_violation:{sequence}"
                )
            if not bool(registry_row.get("torsion_prior_covered", False)):
                raise ValueError(
                    f"{split}_plan_torsion_prior_uncovered:{sequence}"
                )
            if (
                int(record.get("theoretical_heavy_atom_count", -1))
                != atom_count
                or int(record.get("selected_pair_count", -1))
                != pair_counts[sequence]
            ):
                raise ValueError(
                    f"{split}_plan_sequence_record_mismatch:{sequence}"
                )
        expected_conformers = len(sequences) * CONFORMERS_PER_SEQUENCE
        if int(plan.get("future_required_conformer_count", -1)) != (
            expected_conformers
        ):
            raise ValueError(f"{split}_plan_future_conformer_count_mismatch")
        plan_payloads[split] = plan

    train_plan = plan_payloads["train"]["interface_pair_ids"]
    valid_plan = plan_payloads["valid"]["interface_pair_ids"]
    train_sequences = set(
        plan_payloads["train"]["unique_peptide_sequences"]
    )
    valid_sequences = set(
        plan_payloads["valid"]["unique_peptide_sequences"]
    )
    train_relations = {
        train_relation_by_pair[pair_id] for pair_id in train_plan
    }
    valid_relations = {
        valid_relation_by_pair[pair_id] for pair_id in valid_plan
    }
    if set(train_plan) & set(valid_plan):
        raise ValueError("bounded_plan_train_valid_pair_leakage")
    if train_sequences & valid_sequences:
        raise ValueError("bounded_plan_train_valid_sequence_leakage")
    if train_relations & valid_relations:
        raise ValueError("bounded_plan_train_valid_relation_leakage")

    evaluation = descriptor.get("safe373_evaluation_exclusion", {})
    safe_plan_path = _resolve_relative(
        descriptor_path,
        evaluation.get("plan_path"),
        "safe373_plan_path",
    )
    if sha256_file(safe_plan_path) != str(
        evaluation.get("plan_file_sha256") or ""
    ):
        raise ValueError("safe373_plan_file_sha256_mismatch")
    safe_plan = _read_json(safe_plan_path)
    if (
        str(safe_plan.get("plan_canonical_sha256") or "")
        != SAFE373_PLAN_CANONICAL_SHA256
        or evaluation.get("plan_canonical_sha256")
        != SAFE373_PLAN_CANONICAL_SHA256
    ):
        raise ValueError("safe373_plan_canonical_sha256_mismatch")
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
    train_safe_pair = set(train_plan) & safe_pairs
    train_safe_sequence = train_sequences & safe_sequences
    train_safe_relation = train_relations & safe_relations
    if train_safe_pair:
        raise ValueError("bounded_plan_train_safe373_pair_leakage")
    if train_safe_sequence:
        raise ValueError("bounded_plan_train_safe373_sequence_leakage")
    if train_safe_relation:
        raise ValueError("bounded_plan_train_safe373_relation_leakage")

    valid_overlap = {
        "query_pair_ids": sorted(set(valid_plan) & safe_pairs),
        "peptide_sequences": sorted(valid_sequences & safe_sequences),
        "biological_relation_ids": sorted(valid_relations & safe_relations),
    }
    recorded_overlap = evaluation.get("valid_overlap_report", {})
    for key, values in valid_overlap.items():
        if recorded_overlap.get(key) != values:
            raise ValueError(f"safe373_valid_overlap_report_mismatch:{key}")
        if recorded_overlap.get(f"{key}_sha256") != sequence_sha256(values):
            raise ValueError(f"safe373_valid_overlap_sha256_mismatch:{key}")

    cache = descriptor.get("future_cache_requirement", {})
    required_sequences = sorted(train_sequences | valid_sequences)
    if (
        cache.get("generation_status") != "NOT_BUILT"
        or cache.get("cache_status") != "NOT_BUILT"
        or cache.get("cache_manifest_path") is not None
        or cache.get("cache_manifest_sha256") is not None
        or int(cache.get("conformers_per_sequence", -1))
        != CONFORMERS_PER_SEQUENCE
        or int(cache.get("future_required_conformer_count", -1))
        != len(required_sequences) * CONFORMERS_PER_SEQUENCE
        or cache.get("required_peptide_sequences") != required_sequences
        or cache.get("required_peptide_sequences_sha256")
        != sequence_sha256(required_sequences)
    ):
        raise ValueError("bounded_plan_future_cache_requirement_mismatch")

    return {
        "schema_version": PLAN_SCHEMA,
        "descriptor_path": str(descriptor_path),
        "descriptor_file_sha256": sha256_file(descriptor_path),
        "descriptor_canonical_sha256": recorded_sha,
        "train_interface_pair_ids": train_plan,
        "valid_interface_pair_ids": valid_plan,
        "train_unique_peptide_sequences": sorted(train_sequences),
        "valid_unique_peptide_sequences": sorted(valid_sequences),
        "required_peptide_sequences": required_sequences,
        "registry_by_sequence": registry_by_sequence,
        "train_valid_overlap": {
            "interface_pair_ids": [],
            "peptide_sequences": [],
            "biological_relation_ids": [],
        },
        "train_safe373_overlap": {
            "query_pair_ids": sorted(train_safe_pair),
            "peptide_sequences": sorted(train_safe_sequence),
            "biological_relation_ids": sorted(train_safe_relation),
        },
        "valid_safe373_overlap": valid_overlap,
        "future_cache_requirement": cache,
    }


def configure_bounded_full_heavy_trainable(
    model: PepCLIPConcatFusionModel,
) -> dict[str, Any]:
    """Freeze everything except peptide EGNN last block/norm/project and fusion."""

    set_frozen(model)
    peptide_encoder = model.model_3d.peptide_encoder
    layers = getattr(peptide_encoder, "layers", None)
    final_norm = getattr(peptide_encoder, "final_norm", None)
    project = getattr(peptide_encoder, "project", None)
    if model.model_3d.encoder_type != "egnn" or not layers or final_norm is None or project is None:
        raise ValueError("full_heavy_adaptation_requires_peptide_egnn_encoder")
    set_trainable(layers[-1])
    set_trainable(final_norm)
    set_trainable(project)
    set_trainable(model.peptide_fusion)

    trainable_names = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    allowed_prefixes = (
        f"model_3d.peptide_encoder.layers.{len(layers) - 1}.",
        "model_3d.peptide_encoder.final_norm.",
        "model_3d.peptide_encoder.project.",
        "peptide_fusion.",
    )
    disallowed = [
        name for name in trainable_names if not name.startswith(allowed_prefixes)
    ]
    if disallowed:
        raise AssertionError(f"forbidden_trainable_parameters:{disallowed}")
    if not trainable_names or not any(
        name.startswith("peptide_fusion.") for name in trainable_names
    ):
        raise AssertionError("incomplete_full_heavy_trainable_parameter_set")
    return {
        "contract_version": FREEZE_CONTRACT_VERSION,
        "receptor_1d_encoder_frozen": True,
        "receptor_3d_encoder_frozen": True,
        "receptor_fusion_frozen": True,
        "peptide_1d_encoder_frozen": True,
        "peptide_3d_trainable_components": [
            f"layers.{len(layers) - 1}",
            "final_norm",
            "project",
        ],
        "peptide_fusion_trainable": True,
        "temperature_frozen": True,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_names_sha256": sequence_sha256(trainable_names),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def build_bounded_optimizer_groups(
    model: PepCLIPConcatFusionModel,
    *,
    fusion_lr: float,
    tower_lr: float,
) -> list[dict[str, Any]]:
    named = dict(model.named_parameters())
    fusion = [
        parameter
        for name, parameter in named.items()
        if parameter.requires_grad and name.startswith("peptide_fusion.")
    ]
    peptide_3d = [
        parameter
        for name, parameter in named.items()
        if parameter.requires_grad
        and name.startswith("model_3d.peptide_encoder.")
    ]
    selected = {id(parameter) for parameter in (*fusion, *peptide_3d)}
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if selected != expected or not fusion or not peptide_3d:
        raise AssertionError("optimizer_parameter_set_mismatch")
    return [
        {"params": fusion, "lr": float(fusion_lr), "group_name": "peptide_fusion"},
        {"params": peptide_3d, "lr": float(tower_lr), "group_name": "peptide_3d_last1"},
    ]


def parameter_state_sha256(model: torch.nn.Module) -> dict[str, str]:
    return {
        name: hashlib.sha256(
            parameter.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest().upper()
        for name, parameter in model.named_parameters()
    }


def assert_parameter_change_contract(
    before: dict[str, str],
    after: dict[str, str],
    freeze_contract: dict[str, Any],
) -> dict[str, list[str]]:
    if set(before) != set(after):
        raise AssertionError("parameter_identity_set_changed")
    allowed = set(freeze_contract["trainable_parameter_names"])
    changed = sorted(name for name in before if before[name] != after[name])
    forbidden = sorted(set(changed) - allowed)
    if forbidden:
        raise AssertionError(f"forbidden_parameter_changed:{forbidden}")
    return {
        "changed_allowed_parameters": changed,
        "unchanged_frozen_parameters": sorted(set(before) - set(changed)),
    }


def _validate_full_heavy_payload(payload: dict[str, Any], sequence: str) -> None:
    if payload.get("peptide_sequence") != sequence:
        raise ValueError(f"cache_payload_sequence_mismatch:{sequence}")
    identities = payload.get("atom_identity")
    conformers = payload.get("conformers")
    if not isinstance(identities, list) or not identities:
        raise ValueError(f"cache_atom_identity_missing:{sequence}")
    if payload.get("atom_identity_sha256") != canonical_json_sha256(identities):
        raise ValueError(f"cache_atom_identity_sha256_mismatch:{sequence}")
    if len(identities) >= ATOM_CAP_EXCLUSIVE:
        raise ValueError(f"cache_atom_cap_violation:{sequence}")
    canonical_peptide_graph(identities)
    for identity in identities:
        if str(identity.get("element")) not in ELEMENT_TO_ID:
            raise ValueError(f"cache_element_unk:{sequence}:{identity.get('element')}")
        if str(identity.get("atom_name")) not in ATOM_NAME_TO_ID:
            raise ValueError(f"cache_atom_name_unk:{sequence}:{identity.get('atom_name')}")
        if str(identity.get("residue_name")) not in RESIDUE_NAME_TO_ID:
            raise ValueError(
                f"cache_residue_name_unk:{sequence}:{identity.get('residue_name')}"
            )
    if not isinstance(conformers, list) or len(conformers) != CONFORMERS_PER_SEQUENCE:
        raise ValueError(f"cache_requires_exactly_10_conformers:{sequence}")
    indices = [int(row.get("conformer_index", -1)) for row in conformers]
    if indices != list(range(CONFORMERS_PER_SEQUENCE)):
        raise ValueError(f"cache_conformer_order_mismatch:{sequence}")
    coordinate_hashes: set[str] = set()
    for conformer in conformers:
        attempt_index = int(conformer.get("attempt_index", -1))
        if not 0 <= attempt_index < 25:
            raise ValueError(f"cache_attempt_index_contract_mismatch:{sequence}")
        faspr = conformer.get("faspr", {})
        if (
            int(faspr.get("exit_code", -1)) != 0
            or bool(faspr.get("timed_out", False))
        ):
            raise ValueError(f"cache_faspr_contract_failure:{sequence}")
        geometry = conformer.get("geometry_audit", {})
        if (
            geometry.get("status") != "PASS"
            or geometry.get("topology_contract") != CANONICAL_TOPOLOGY_CONTRACT
        ):
            raise ValueError(f"cache_canonical_geometry_qc_failure:{sequence}")
        attempt_qc = conformer.get("attempt_qc", {})
        structural_qc = attempt_qc.get("structural_qc", {})
        cpu_forward = attempt_qc.get("cpu_egnn_forward", {})
        if (
            attempt_qc.get("status") != "PASS"
            or structural_qc.get("status") != "PASS"
            or bool(structural_qc.get("target_bound_inputs_used", True))
            or cpu_forward.get("status") != "PASS"
            or not bool(cpu_forward.get("embedding_finite", False))
            or int(cpu_forward.get("tensorization_unk_count", -1)) != 0
        ):
            raise ValueError(f"cache_accepted_qc_contract_failure:{sequence}")
        coordinates = conformer.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) != len(identities):
            raise ValueError(f"cache_coordinate_count_mismatch:{sequence}")
        if not all(
            isinstance(xyz, list)
            and len(xyz) == 3
            and all(math.isfinite(float(value)) for value in xyz)
            for xyz in coordinates
        ):
            raise ValueError(f"cache_nonfinite_coordinate:{sequence}")
        coordinate_hash = canonical_json_sha256(coordinates)
        if coordinate_hash in coordinate_hashes:
            raise ValueError(f"cache_duplicate_conformer_coordinates:{sequence}")
        coordinate_hashes.add(coordinate_hash)


def validate_bounded_full_heavy_contract(
    adaptation_manifest_file: str | Path,
    *,
    plan_descriptor_file: str | Path,
    cache_manifest_file: str | Path,
    phase2_checkpoint: str | Path,
    train_interface_pair_ids: Iterable[str],
    valid_interface_pair_ids: Iterable[str],
    train_sequence_by_pair: dict[str, str],
    valid_sequence_by_pair: dict[str, str],
    train_relation_by_pair: dict[str, str],
    valid_relation_by_pair: dict[str, str],
    freeze_contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate separately bound plan, materialized cache, and adaptation."""

    plan_contract = validate_bounded_plan_descriptor(
        plan_descriptor_file,
        train_interface_pair_ids=train_interface_pair_ids,
        valid_interface_pair_ids=valid_interface_pair_ids,
        train_sequence_by_pair=train_sequence_by_pair,
        valid_sequence_by_pair=valid_sequence_by_pair,
        train_relation_by_pair=train_relation_by_pair,
        valid_relation_by_pair=valid_relation_by_pair,
    )
    descriptor_path = Path(plan_descriptor_file).resolve()
    cache_manifest_path = Path(cache_manifest_file).resolve()
    cache_manifest = _read_json(cache_manifest_path)
    if cache_manifest.get("schema_version") != CACHE_MANIFEST_SCHEMA:
        raise ValueError("unsupported_full_heavy_cache_manifest")
    cache_recorded_sha = str(
        cache_manifest.get("manifest_canonical_sha256") or ""
    )
    cache_core = {
        key: value
        for key, value in cache_manifest.items()
        if key != "manifest_canonical_sha256"
    }
    if canonical_json_sha256(cache_core) != cache_recorded_sha:
        raise ValueError("full_heavy_cache_manifest_canonical_sha256_mismatch")
    _validate_no_absolute_path_fields(cache_manifest)
    if (
        cache_manifest.get("status") != "MATERIALIZED"
        or cache_manifest.get("purpose") != "bounded_train_valid_only"
        or cache_manifest.get("plan_descriptor_file_sha256")
        != sha256_file(descriptor_path)
        or cache_manifest.get("plan_descriptor_canonical_sha256")
        != plan_contract["descriptor_canonical_sha256"]
        or int(cache_manifest.get("conformers_per_sequence", -1))
        != CONFORMERS_PER_SEQUENCE
        or int(cache_manifest.get("atom_cap_exclusive", -1))
        != ATOM_CAP_EXCLUSIVE
    ):
        raise ValueError("full_heavy_cache_manifest_contract_mismatch")

    required_sequences = set(plan_contract["required_peptide_sequences"])
    if cache_manifest.get("required_peptide_sequences") != sorted(
        required_sequences
    ):
        raise ValueError("full_heavy_cache_required_sequence_mismatch")
    if cache_manifest.get(
        "required_peptide_sequences_sha256"
    ) != sequence_sha256(sorted(required_sequences)):
        raise ValueError("full_heavy_cache_required_sequence_sha256_mismatch")
    index_path = _resolve_relative(
        cache_manifest_path,
        cache_manifest.get("index_path"),
        "full_heavy_cache_index_path",
    )
    index_text = str(cache_manifest.get("index_path") or "").lower()
    if "safe265" in index_text or "safe373" in index_text:
        raise ValueError("full_heavy_evaluation_cache_path_forbidden")
    if sha256_file(index_path) != cache_manifest.get("index_sha256"):
        raise ValueError("full_heavy_cache_index_sha256_mismatch")
    rows = _read_jsonl(index_path)
    if len(rows) != len(required_sequences):
        raise ValueError("full_heavy_cache_sequence_count_mismatch")

    train_plan = plan_contract["train_interface_pair_ids"]
    valid_plan = plan_contract["valid_interface_pair_ids"]
    train_sequences = set(plan_contract["train_unique_peptide_sequences"])
    valid_sequences = set(plan_contract["valid_unique_peptide_sequences"])
    payload_by_sequence: dict[str, dict[str, Any]] = {}
    for row in rows:
        sequence = str(row.get("peptide_sequence") or "")
        if sequence in payload_by_sequence:
            raise ValueError(f"duplicate_full_heavy_cache_sequence:{sequence}")
        if row.get("chemistry_classification") != "ordinary_linear_standard":
            raise ValueError(f"full_heavy_nonordinary_sequence:{sequence}")
        roles = set(row.get("split_roles", []))
        expected_roles = (
            ({"train"} if sequence in train_sequences else set())
            | ({"valid"} if sequence in valid_sequences else set())
        )
        if roles != expected_roles:
            raise ValueError(f"full_heavy_cache_split_role_mismatch:{sequence}")
        payload_path = _resolve_relative(
            cache_manifest_path,
            row.get("cache_path"),
            f"full_heavy_payload_path:{sequence}",
        )
        if sha256_file(payload_path) != row.get("cache_file_sha256"):
            raise ValueError(f"full_heavy_cache_file_sha256_mismatch:{sequence}")
        payload = _read_json(payload_path)
        _validate_full_heavy_payload(payload, sequence)
        payload_by_sequence[sequence] = payload
    if set(payload_by_sequence) != required_sequences:
        raise ValueError("full_heavy_cache_sequence_set_mismatch")

    adaptation_path = Path(adaptation_manifest_file).resolve()
    adaptation = _read_json(adaptation_path)
    if adaptation.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("unsupported_full_heavy_adaptation_manifest")
    adaptation_recorded_sha = str(
        adaptation.get("manifest_canonical_sha256") or ""
    )
    adaptation_core = {
        key: value
        for key, value in adaptation.items()
        if key != "manifest_canonical_sha256"
    }
    if canonical_json_sha256(adaptation_core) != adaptation_recorded_sha:
        raise ValueError(
            "full_heavy_adaptation_manifest_canonical_sha256_mismatch"
        )
    _validate_no_absolute_path_fields(adaptation)
    checkpoint_sha = sha256_file(phase2_checkpoint)
    expected_freeze_sha = str(
        freeze_contract.get("trainable_parameter_names_sha256") or ""
    )
    if (
        checkpoint_sha != PHASE2_INITIALIZATION_SHA256
        or adaptation.get("phase2_checkpoint_sha256")
        != PHASE2_INITIALIZATION_SHA256
        or adaptation.get("plan_descriptor_file_sha256")
        != sha256_file(descriptor_path)
        or adaptation.get("plan_descriptor_canonical_sha256")
        != plan_contract["descriptor_canonical_sha256"]
        or adaptation.get("cache_manifest_file_sha256")
        != sha256_file(cache_manifest_path)
        or adaptation.get("cache_manifest_canonical_sha256")
        != cache_recorded_sha
        or adaptation.get("freeze_contract_version")
        != FREEZE_CONTRACT_VERSION
        or adaptation.get("trainable_parameter_names_sha256")
        != expected_freeze_sha
    ):
        raise ValueError("full_heavy_adaptation_binding_mismatch")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "adaptation_manifest_path": str(adaptation_path),
        "adaptation_manifest_canonical_sha256": adaptation_recorded_sha,
        "plan_descriptor_path": str(descriptor_path),
        "plan_descriptor_file_sha256": sha256_file(descriptor_path),
        "plan_descriptor_canonical_sha256": plan_contract[
            "descriptor_canonical_sha256"
        ],
        "cache_manifest_path": str(cache_manifest_path),
        "cache_manifest_file_sha256": sha256_file(cache_manifest_path),
        "cache_manifest_canonical_sha256": cache_recorded_sha,
        "phase2_initialization_sha256": checkpoint_sha,
        "freeze_contract_version": FREEZE_CONTRACT_VERSION,
        "trainable_parameter_names_sha256": expected_freeze_sha,
        "train_interface_pair_ids": train_plan,
        "valid_interface_pair_ids": valid_plan,
        "train_unique_peptide_count": len(train_sequences),
        "valid_unique_peptide_count": len(valid_sequences),
        "cache_sequence_count": len(payload_by_sequence),
        "cache_index_path": str(index_path),
        "cache_index_sha256": cache_manifest["index_sha256"],
        "payload_by_sequence": payload_by_sequence,
    }


class FullHeavyDatasetView:
    """Replace only peptide coordinates in a formal v3 dataset plan."""

    def __init__(self, base: Any, payload_by_sequence: dict[str, dict[str, Any]]) -> None:
        self.base = base
        self.payload_by_sequence = payload_by_sequence
        self.split = base.split
        self.mode = base.mode
        self.global_seed = base.global_seed
        self.interface_pair_ids = base.interface_pair_ids
        self.interface_row_count = base.interface_row_count

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def epoch_plan(self) -> list[dict[str, Any]]:
        return self.base.epoch_plan()

    def peptide_sequence_for_index(self, index: int) -> str:
        return self.base.peptide_sequence_for_index(index)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base[index])
        sequence = str(item["peptide_sequence"])
        payload = self.payload_by_sequence[sequence]
        conformer_index = int(item["conformer_index"])
        item["peptide_atoms"] = conformer_atoms(payload, conformer_index)
        item["conformer_cluster_id"] = (
            f"fullheavy:{sequence}:{conformer_index}:"
            f"{payload['atom_identity_sha256']}"
        )
        item["conformer_source_kind"] = "train_only_full_heavy_sequence_conformer"
        return item

"""Finalize the immutable Phase-3 v2 bounded full-heavy adaptation manifest.

This entry point only binds already-materialized, read-only artifacts.  It does
not create conformers, load a model, create an optimizer, or write checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from phase3.drugclip.full_heavy_adaptation_contract import (
    ATOM_CAP_EXCLUSIVE,
    CACHE_MANIFEST_SCHEMA,
    CANONICAL_TOPOLOGY_CONTRACT,
    CONFORMERS_PER_SEQUENCE,
    CONTRACT_SCHEMA,
    FREEZE_CONTRACT_VERSION,
    MAX_TRAIN_PAIRS,
    MAX_VALID_PAIRS,
    PHASE2_INITIALIZATION_SHA256,
    SAFE373_PLAN_CANONICAL_SHA256,
    canonical_json_sha256,
    sha256_file,
)


EXPECTED_PLAN_FILE_SHA256 = (
    "1894F635E352D127AC79DF226E4F50A7451B8E47C43D6388239A23752721957D"
)
EXPECTED_PLAN_CANONICAL_SHA256 = (
    "2F8FF55185DE5E87861687CA564EC4851E186C16C4C8158B9C1168D8E32D8DE0"
)
EXPECTED_CACHE_FILE_SHA256 = (
    "8FB8BB574D72925445D4C13B930F26273691868A9DC351EBB6CDD2B76E5FB992"
)
EXPECTED_CACHE_CANONICAL_SHA256 = (
    "AC189E317FF454A199C8A6C3F8FFD4EDB1C74DF681F3B10B678B0793547A67ED"
)
EXPECTED_TRAINABLE_PARAMETER_NAMES_SHA256 = (
    "246F0A44F6D3E39FA64F0EA2C04E416316375D9BBBA66700EB566D0DB505745D"
)
EXPECTED_TRAINABLE_TENSOR_COUNT = 22
EXPECTED_TRAINABLE_PARAMETER_COUNT = 2_580_096
FINALIZER_VERSION = "phase3-v2-bounded-full-heavy-adaptation-finalizer-v1"

SPECIAL_CHEMISTRY_EXCLUSIONS = [
    "receptor_covalent",
    "modified_or_nonstandard",
    "chemistry_insufficient",
    "known_disulfide",
    "multiple_cys_unknown",
    "cyclic_or_crosslinked",
]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def _relative_path(target: Path, parent: Path) -> str:
    value = os.path.relpath(target.resolve(), parent.resolve()).replace("\\", "/")
    if Path(value).is_absolute() or ":" in value:
        raise ValueError(f"manifest_path_not_relative:{value}")
    return value


def _canonical_core(value: dict[str, Any], sha_field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != sha_field}


def _validate_embedded_sha(
    value: dict[str, Any], sha_field: str, error: str
) -> str:
    recorded = str(value.get(sha_field) or "").upper()
    if canonical_json_sha256(_canonical_core(value, sha_field)) != recorded:
        raise ValueError(error)
    return recorded


def build_final_adaptation_manifest(
    *,
    output_file: str | Path,
    plan_descriptor_file: str | Path,
    cache_manifest_file: str | Path,
    phase2_checkpoint: str | Path,
    random_conformer_v3_manifest: str | Path,
    safe373_evaluation_plan: str | Path,
) -> dict[str, Any]:
    """Build and fully validate a deterministic final binding in memory."""

    output_path = Path(output_file).resolve()
    plan_path = Path(plan_descriptor_file).resolve()
    cache_path = Path(cache_manifest_file).resolve()
    checkpoint_path = Path(phase2_checkpoint).resolve()
    random_manifest_path = Path(random_conformer_v3_manifest).resolve()
    safe_plan_path = Path(safe373_evaluation_plan).resolve()
    for path in (
        plan_path,
        cache_path,
        checkpoint_path,
        random_manifest_path,
        safe_plan_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    plan = _read_json(plan_path)
    cache = _read_json(cache_path)
    cache_contract_path = cache_path.parent / "cache_contract.json"
    cache_contract = _read_json(cache_contract_path)
    random_manifest = _read_json(random_manifest_path)
    safe_plan = _read_json(safe_plan_path)

    plan_file_sha = sha256_file(plan_path)
    plan_canonical_sha = _validate_embedded_sha(
        plan, "descriptor_canonical_sha256", "plan_descriptor_canonical_sha256_mismatch"
    )
    cache_file_sha = sha256_file(cache_path)
    cache_canonical_sha = _validate_embedded_sha(
        cache, "manifest_canonical_sha256", "cache_manifest_canonical_sha256_mismatch"
    )
    cache_contract_canonical_sha = _validate_embedded_sha(
        cache_contract,
        "contract_canonical_sha256",
        "cache_contract_canonical_sha256_mismatch",
    )
    safe_plan_canonical_sha = _validate_embedded_sha(
        safe_plan, "plan_canonical_sha256", "safe373_plan_canonical_sha256_mismatch"
    )
    if plan_file_sha != EXPECTED_PLAN_FILE_SHA256:
        raise ValueError("plan_descriptor_file_sha256_mismatch")
    if plan_canonical_sha != EXPECTED_PLAN_CANONICAL_SHA256:
        raise ValueError("plan_descriptor_registered_canonical_sha256_mismatch")
    if cache_file_sha != EXPECTED_CACHE_FILE_SHA256:
        raise ValueError("cache_manifest_file_sha256_mismatch")
    if cache_canonical_sha != EXPECTED_CACHE_CANONICAL_SHA256:
        raise ValueError("cache_manifest_registered_canonical_sha256_mismatch")
    if sha256_file(checkpoint_path) != PHASE2_INITIALIZATION_SHA256:
        raise ValueError("phase2_checkpoint_sha256_mismatch")
    if safe_plan_canonical_sha != SAFE373_PLAN_CANONICAL_SHA256:
        raise ValueError("safe373_plan_registered_canonical_sha256_mismatch")
    if cache.get("schema_version") != CACHE_MANIFEST_SCHEMA:
        raise ValueError("unsupported_cache_manifest_schema")
    if cache.get("cache_contract_canonical_sha256") != cache_contract_canonical_sha:
        raise ValueError("cache_contract_binding_mismatch")
    if (
        cache.get("plan_descriptor_file_sha256") != plan_file_sha
        or cache.get("plan_descriptor_canonical_sha256") != plan_canonical_sha
    ):
        raise ValueError("cache_plan_binding_mismatch")
    train_plan = plan.get("plans", {}).get("train", {})
    valid_plan = plan.get("plans", {}).get("valid", {})
    if (
        int(train_plan.get("pair_count", -1)) != MAX_TRAIN_PAIRS
        or int(valid_plan.get("pair_count", -1)) != MAX_VALID_PAIRS
    ):
        raise ValueError("bounded_plan_pair_count_mismatch")
    if (
        int(cache.get("sequence_count", -1)) != 2085
        or int(cache.get("conformer_count", -1)) != 20850
        or int(cache.get("conformers_per_sequence", -1))
        != CONFORMERS_PER_SEQUENCE
        or int(cache.get("atom_cap_exclusive", -1)) != ATOM_CAP_EXCLUSIVE
    ):
        raise ValueError("bounded_cache_count_contract_mismatch")
    if random_manifest.get("dataset_version") != "random_conformer_v3":
        raise ValueError("formal_random_conformer_v3_manifest_mismatch")
    if cache_contract.get("canonical_topology_contract") != CANONICAL_TOPOLOGY_CONTRACT:
        raise ValueError("canonical_topology_contract_mismatch")

    parent = output_path.parent
    generation = {
        key: cache_contract[key]
        for key in (
            "generator_version",
            "torsion_prior_manifest_file_sha256",
            "torsion_prior_manifest_canonical_sha256",
            "torsion_prior_jsonl_sha256",
            "faspr_source_commit",
            "faspr_binary_sha256",
            "faspr_rotamer_library_sha256",
            "canonical_topology_contract",
            "maximum_attempts_per_logical_conformer",
            "nonlocal_heavy_atom_clash_threshold_angstrom",
            "atom_cap_exclusive",
            "generation_input_contract",
        )
    }
    core: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA,
        "finalizer_version": FINALIZER_VERSION,
        "status": "FINALIZED_NOT_TRAINED",
        "purpose": "bounded_full_heavy_train_valid_adaptation",
        # Flat fields are retained as the runtime validator's fail-closed API.
        "phase2_checkpoint_sha256": PHASE2_INITIALIZATION_SHA256,
        "plan_descriptor_file_sha256": plan_file_sha,
        "plan_descriptor_canonical_sha256": plan_canonical_sha,
        "cache_manifest_file_sha256": cache_file_sha,
        "cache_manifest_canonical_sha256": cache_canonical_sha,
        "freeze_contract_version": FREEZE_CONTRACT_VERSION,
        "trainable_parameter_names_sha256": (
            EXPECTED_TRAINABLE_PARAMETER_NAMES_SHA256
        ),
        "plan": {
            "path": _relative_path(plan_path, parent),
            "file_sha256": plan_file_sha,
            "canonical_sha256": plan_canonical_sha,
            "train_pair_count": MAX_TRAIN_PAIRS,
            "valid_pair_count": MAX_VALID_PAIRS,
            "train_pair_ids_sha256": train_plan["interface_pair_ids_sha256"],
            "valid_pair_ids_sha256": valid_plan["interface_pair_ids_sha256"],
        },
        "cache": {
            "path": _relative_path(cache_path, parent),
            "file_sha256": cache_file_sha,
            "canonical_sha256": cache_canonical_sha,
            "cache_contract_path": _relative_path(cache_contract_path, parent),
            "cache_contract_canonical_sha256": cache_contract_canonical_sha,
            "sequence_count": 2085,
            "conformer_count": 20850,
            "conformers_per_sequence": CONFORMERS_PER_SEQUENCE,
        },
        "initialization": {
            "role": "phase2_learned_concat_baseline",
            "checkpoint_path": _relative_path(checkpoint_path, parent),
            "checkpoint_sha256": PHASE2_INITIALIZATION_SHA256,
        },
        "freeze_contract": {
            "version": FREEZE_CONTRACT_VERSION,
            "trainable_parameter_names_sha256": (
                EXPECTED_TRAINABLE_PARAMETER_NAMES_SHA256
            ),
            "trainable_tensor_count": EXPECTED_TRAINABLE_TENSOR_COUNT,
            "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETER_COUNT,
            "trainable_scope": [
                "model_3d.peptide_encoder.layers.2.edge_mlp",
                "model_3d.peptide_encoder.layers.2.node_mlp",
                "model_3d.peptide_encoder.layers.2.norm",
                "model_3d.peptide_encoder.final_norm",
                "model_3d.peptide_encoder.project",
                "peptide_fusion",
            ],
            "frozen_scope": [
                "model_3d.peptide_encoder.layers.2.coord_mlp",
            ],
        },
        "formal_random_conformer_v3": {
            "manifest_path": _relative_path(random_manifest_path, parent),
            "manifest_file_sha256": sha256_file(random_manifest_path),
            "dataset_version": "random_conformer_v3",
            "generator_id": random_manifest.get("generator_id"),
            "cache_schema": random_manifest.get("cache_schema"),
        },
        "safe373_evaluation_plan": {
            "path": _relative_path(safe_plan_path, parent),
            "file_sha256": sha256_file(safe_plan_path),
            "canonical_sha256": safe_plan_canonical_sha,
            "role": "evaluation_exclusion_reference_only",
            "cache_used_for_training": False,
        },
        "generation_contract": generation,
        "source_policy": {
            "training_source_split": "formal_train_and_valid_plan_only",
            "chemistry_classification": "ordinary_linear_standard",
            "special_chemistry_in_scope": False,
            "excluded_chemistry_classifications": SPECIAL_CHEMISTRY_EXCLUSIONS,
            "evaluation_cache_used_for_training": False,
            "target_bound_generation_inputs_used": False,
        },
        "counts": {
            "train_interface_pairs": MAX_TRAIN_PAIRS,
            "valid_interface_pairs": MAX_VALID_PAIRS,
            "cache_sequences": 2085,
            "cache_conformers": 20850,
        },
        "execution_state": {
            "optimizer_created": False,
            "optimizer_step_executed": False,
            "backward_executed": False,
            "checkpoint_written": False,
            "training_executed": False,
            "retrieval_executed": False,
        },
    }
    return {
        **core,
        "manifest_canonical_sha256": canonical_json_sha256(core),
    }


def manifest_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale_temporary_file:{temporary}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize the read-only Phase-3 v2 bounded full-heavy adaptation manifest."
    )
    parser.add_argument("--plan-descriptor", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--phase2-checkpoint", required=True)
    parser.add_argument("--random-conformer-v3-manifest", required=True)
    parser.add_argument("--safe373-evaluation-plan", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output_dir_already_exists:{output_dir}")
    output_file = output_dir / "final_adaptation_manifest.json"
    first = build_final_adaptation_manifest(
        output_file=output_file,
        plan_descriptor_file=args.plan_descriptor,
        cache_manifest_file=args.cache_manifest,
        phase2_checkpoint=args.phase2_checkpoint,
        random_conformer_v3_manifest=args.random_conformer_v3_manifest,
        safe373_evaluation_plan=args.safe373_evaluation_plan,
    )
    second = build_final_adaptation_manifest(
        output_file=output_file,
        plan_descriptor_file=args.plan_descriptor,
        cache_manifest_file=args.cache_manifest,
        phase2_checkpoint=args.phase2_checkpoint,
        random_conformer_v3_manifest=args.random_conformer_v3_manifest,
        safe373_evaluation_plan=args.safe373_evaluation_plan,
    )
    first_bytes = manifest_bytes(first)
    if first_bytes != manifest_bytes(second):
        raise RuntimeError("adaptation_manifest_not_byte_deterministic")
    write_atomic(output_file, first_bytes)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output_file),
                "file_sha256": sha256_file(output_file),
                "canonical_sha256": first["manifest_canonical_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

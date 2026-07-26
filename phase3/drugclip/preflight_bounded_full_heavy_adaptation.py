"""Real-model, zero-step preflight for bounded full-heavy adaptation.

The module intentionally has no optimizer, scheduler, backward, AMP-step, or
checkpoint-writing path.  It performs read-only contract validation and two
inference-only batch forwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import torch

from phase3.drugclip.batching import UniquePeptideBatchSampler, collate_phase3
from phase3.drugclip.finalize_bounded_full_heavy_adaptation import (
    EXPECTED_TRAINABLE_PARAMETER_COUNT,
    EXPECTED_TRAINABLE_PARAMETER_NAMES_SHA256,
    EXPECTED_TRAINABLE_TENSOR_COUNT,
    manifest_bytes,
    write_atomic,
)
from phase3.drugclip.forward import forward_and_known_positive_loss
from phase3.drugclip.full_heavy_adaptation_contract import (
    FullHeavyDatasetView,
    canonical_json_sha256,
    configure_bounded_full_heavy_trainable,
    sequence_sha256,
    sha256_file,
    validate_bounded_full_heavy_contract,
)
from phase3.drugclip.random_augmentation_dataset import (
    InterfacePairSubsetDataset,
    Phase3RandomConformerDataset,
)
from phase3.drugclip.train import (
    load_phase2_fusion_model,
    load_source_configs,
)


PREFLIGHT_SCHEMA = "phase3-v2-bounded-full-heavy-real-zero-step-preflight-v1"
EXPECTED_MODEL_STATE_TENSOR_COUNT = 352
EXPECTED_MODEL_STATE_ELEMENT_COUNT = 28_575_002


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest().upper()


def optimizer_parameter_group_description(
    model: torch.nn.Module,
) -> list[dict[str, Any]]:
    """Describe the two registered groups without constructing an optimizer."""

    groups = [
        (
            "peptide_fusion",
            [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and name.startswith("peptide_fusion.")
            ],
        ),
        (
            "peptide_3d_last1",
            [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and name.startswith("model_3d.peptide_encoder.")
            ],
        ),
    ]
    described: list[dict[str, Any]] = []
    selected: set[str] = set()
    for group_name, rows in groups:
        if not rows:
            raise AssertionError(f"empty_optimizer_parameter_group:{group_name}")
        names = sorted(name for name, _ in rows)
        selected.update(names)
        described.append(
            {
                "group_name": group_name,
                "parameter_names": names,
                "parameter_names_sha256": sequence_sha256(names),
                "tensor_count": len(names),
                "parameter_count": sum(parameter.numel() for _, parameter in rows),
            }
        )
    expected = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if selected != expected:
        raise AssertionError("optimizer_parameter_description_set_mismatch")
    return described


def _batch_audit(
    model: torch.nn.Module,
    dataset: InterfacePairSubsetDataset,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    split: str,
) -> dict[str, Any]:
    sampler = UniquePeptideBatchSampler(dataset, batch_size, seed=seed, epoch=0)
    indices = next(iter(sampler))
    batch = collate_phase3([dataset[index] for index in indices])
    sequences = [str(value) for value in batch["one_d"]["peptide_sequence"]]
    if len(sequences) != len(set(sequences)):
        raise ValueError(f"{split}_batch_peptide_not_unique")
    with torch.inference_mode():
        result = forward_and_known_positive_loss(model, batch, device)
    tensors = {
        "receptor_embedding": result["receptor_embedding"],
        "peptide_embedding": result["peptide_embedding"],
        "logits_receptor_to_peptide": result["similarity_matrix"],
        "logits_peptide_to_receptor": result["similarity_matrix_transpose"],
        "loss_total": result["loss_total"],
        "loss_receptor_to_peptide": result["loss_receptor_to_peptide"],
        "loss_peptide_to_receptor": result["loss_peptide_to_receptor"],
    }
    finite = {
        key: bool(torch.isfinite(value).all().item()) for key, value in tensors.items()
    }
    if not all(finite.values()):
        raise FloatingPointError(f"{split}_zero_step_nonfinite")
    return {
        "split": split,
        "batch_size": len(indices),
        "dataset_indices": indices,
        "interface_pair_ids": list(result["batch_interface_pair_id"]),
        "peptide_sequences": sequences,
        "peptide_unique": True,
        "conformer_indices": list(batch["one_d"]["conformer_index"]),
        "conformer_source_kinds": list(batch["one_d"]["conformer_source_kind"]),
        "tensor_shapes": {
            key: list(value.shape) for key, value in tensors.items()
        },
        "finite": finite,
        "loss_total": float(result["loss_total"].item()),
        "loss_receptor_to_peptide": float(
            result["loss_receptor_to_peptide"].item()
        ),
        "loss_peptide_to_receptor": float(
            result["loss_peptide_to_receptor"].item()
        ),
    }


def _formal_paths(dataset_root: Path) -> dict[str, Path]:
    return {
        "random_pairs": dataset_root
        / "04_training_input"
        / "random_conformer_pairs.jsonl",
        "random_cache": dataset_root
        / "03_random_conformer_cache"
        / "random_conformer_cache.jsonl",
        "biological_pairs": dataset_root / "dependencies" / "biological_pairs.jsonl",
        "pair_splits": dataset_root
        / "02_leakage_safe_split"
        / "pair_splits.jsonl",
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.adaptation_manifest).resolve()
    plan_path = Path(args.plan_descriptor).resolve()
    cache_path = Path(args.cache_manifest).resolve()
    checkpoint_path = Path(args.phase2_checkpoint).resolve()
    source_configs_path = Path(args.source_model_configs).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    random_manifest_path = dataset_root / "DATA_MANIFEST.json"
    safe_plan_path = Path(args.safe373_evaluation_plan).resolve()
    esm_path = Path(args.esm_model).resolve()
    if output_dir != manifest_path.parent:
        raise ValueError("adaptation_manifest_must_be_in_output_dir")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not esm_path.is_dir():
        raise FileNotFoundError(esm_path)

    file_paths = {
        "adaptation_manifest": manifest_path,
        "plan_descriptor": plan_path,
        "cache_manifest": cache_path,
        "phase2_checkpoint": checkpoint_path,
        "random_conformer_v3_manifest": random_manifest_path,
        "safe373_evaluation_plan": safe_plan_path,
    }
    before_file_sha = {key: sha256_file(path) for key, path in file_paths.items()}
    plan = _read_json(plan_path)
    formal = _formal_paths(dataset_root)

    # Make all transformer resolution fail closed to the restored local asset.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    device = torch.device("cpu")
    source_configs = load_source_configs(checkpoint_path, str(source_configs_path), repo_root)
    model_args = SimpleNamespace(
        hf_model_name_or_path_1d=str(esm_path),
        fusion_hidden_dim=None,
        fusion_output_dim=None,
        dropout=None,
        temperature=None,
    )
    model = load_phase2_fusion_model(
        checkpoint_path, source_configs, device, model_args, repo_root
    )
    model.eval()
    model_state = model.state_dict()
    model_state_tensor_count = len(model_state)
    model_state_element_count = sum(value.numel() for value in model_state.values())
    model_state_all_finite = all(
        bool(torch.isfinite(value).all().item())
        for value in model_state.values()
        if value.is_floating_point() or value.is_complex()
    )
    if (
        model_state_tensor_count != EXPECTED_MODEL_STATE_TENSOR_COUNT
        or model_state_element_count != EXPECTED_MODEL_STATE_ELEMENT_COUNT
        or not model_state_all_finite
    ):
        raise ValueError("real_model_state_contract_mismatch")
    freeze_contract = configure_bounded_full_heavy_trainable(model)
    trainable_names = freeze_contract["trainable_parameter_names"]
    coord_mlp_frozen = not any(
        parameter.requires_grad
        for parameter in model.model_3d.peptide_encoder.layers[-1].coord_mlp.parameters()
    )
    feature_path_only = all(
        ".coord_mlp." not in name
        and name.startswith(
            (
                "model_3d.peptide_encoder.layers.2.edge_mlp.",
                "model_3d.peptide_encoder.layers.2.node_mlp.",
                "model_3d.peptide_encoder.layers.2.norm.",
                "model_3d.peptide_encoder.final_norm.",
                "model_3d.peptide_encoder.project.",
                "peptide_fusion.",
            )
        )
        for name in trainable_names
    )
    if (
        len(trainable_names)
        != EXPECTED_TRAINABLE_TENSOR_COUNT
        or int(freeze_contract["trainable_parameter_count"])
        != EXPECTED_TRAINABLE_PARAMETER_COUNT
        or freeze_contract["trainable_parameter_names_sha256"]
        != EXPECTED_TRAINABLE_PARAMETER_NAMES_SHA256
        or not coord_mlp_frozen
        or not feature_path_only
    ):
        raise ValueError("real_model_freeze_contract_mismatch")
    parameter_groups = optimizer_parameter_group_description(model)

    dataset_kwargs = {
        "random_pairs_jsonl": formal["random_pairs"],
        "random_conformer_cache_jsonl": formal["random_cache"],
        "biological_pairs_jsonl": formal["biological_pairs"],
        "pair_splits_jsonl": formal["pair_splits"],
        "data_version": "v3",
        "dataset_root": dataset_root,
        "expected_manifest_sha256": before_file_sha[
            "random_conformer_v3_manifest"
        ],
    }
    train_base = Phase3RandomConformerDataset(
        **dataset_kwargs,
        split="train",
        mode="train_random",
        global_seed=int(args.seed),
    )
    valid_base = Phase3RandomConformerDataset(
        **dataset_kwargs,
        split="valid",
        mode="fixed",
        global_seed=int(args.seed) + 17,
        fixed_conformer_index=0,
    )
    runtime = validate_bounded_full_heavy_contract(
        manifest_path,
        plan_descriptor_file=plan_path,
        cache_manifest_file=cache_path,
        phase2_checkpoint=checkpoint_path,
        train_interface_pair_ids=train_base.interface_pair_ids,
        valid_interface_pair_ids=valid_base.interface_pair_ids,
        train_sequence_by_pair={
            pair_id: str(row["pair"]["peptide_sequence"])
            for pair_id, row in train_base.interface_pair_rows.items()
        },
        valid_sequence_by_pair={
            pair_id: str(row["pair"]["peptide_sequence"])
            for pair_id, row in valid_base.interface_pair_rows.items()
        },
        train_relation_by_pair={
            pair_id: str(row["_biological_pair_id"])
            for pair_id, row in train_base.interface_pair_rows.items()
        },
        valid_relation_by_pair={
            pair_id: str(row["_biological_pair_id"])
            for pair_id, row in valid_base.interface_pair_rows.items()
        },
        freeze_contract=freeze_contract,
    )
    payloads = runtime.pop("payload_by_sequence")
    train_ids = runtime["train_interface_pair_ids"]
    valid_ids = runtime["valid_interface_pair_ids"]
    train_dataset = InterfacePairSubsetDataset(
        FullHeavyDatasetView(train_base, payloads), train_ids
    )
    valid_dataset = InterfacePairSubsetDataset(
        FullHeavyDatasetView(valid_base, payloads), valid_ids
    )
    descriptor_train = plan["plans"]["train"]
    descriptor_valid = plan["plans"]["valid"]
    if (
        train_dataset.interface_pair_ids != descriptor_train["interface_pair_ids"]
        or valid_dataset.interface_pair_ids != descriptor_valid["interface_pair_ids"]
        or sequence_sha256(train_dataset.interface_pair_ids)
        != descriptor_train["interface_pair_ids_sha256"]
        or sequence_sha256(valid_dataset.interface_pair_ids)
        != descriptor_valid["interface_pair_ids_sha256"]
    ):
        raise ValueError("real_dataset_plan_order_or_sha_mismatch")

    frozen_modules = {
        "receptor_1d_encoder": model.model_1d.receptor_encoder,
        "peptide_1d_encoder": model.model_1d.peptide_encoder,
        "receptor_3d_encoder": model.model_3d.receptor_encoder,
        "receptor_fusion": model.receptor_fusion,
    }
    before_state = {
        key: module_state_sha256(module) for key, module in frozen_modules.items()
    }
    train_batch = _batch_audit(
        model,
        train_dataset,
        device=device,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        split="train",
    )
    valid_batch = _batch_audit(
        model,
        valid_dataset,
        device=device,
        batch_size=int(args.batch_size),
        seed=int(args.seed) + 17,
        split="valid",
    )
    after_state = {
        key: module_state_sha256(module) for key, module in frozen_modules.items()
    }
    unchanged = {key: before_state[key] == after_state[key] for key in before_state}
    if not all(unchanged.values()):
        raise AssertionError("frozen_module_state_changed_during_zero_step")
    gradients = {
        name: parameter.grad is not None for name, parameter in model.named_parameters()
    }
    if any(gradients.values()):
        raise AssertionError("gradient_created_during_zero_step")
    after_file_sha = {key: sha256_file(path) for key, path in file_paths.items()}
    if before_file_sha != after_file_sha:
        raise AssertionError("bound_input_file_changed_during_zero_step")

    batch_audit = {
        "schema_version": PREFLIGHT_SCHEMA,
        "train": train_batch,
        "valid": valid_batch,
    }
    validation = {
        key: value
        for key, value in runtime.items()
        if not key.endswith("_path") or isinstance(value, (str, int, float, bool))
    }
    validation.update(
        {
            "status": "PASS",
            "train_interface_pair_count": len(train_ids),
            "valid_interface_pair_count": len(valid_ids),
            "train_interface_pair_ids_sha256": sequence_sha256(train_ids),
            "valid_interface_pair_ids_sha256": sequence_sha256(valid_ids),
        }
    )
    report = {
        "schema_version": PREFLIGHT_SCHEMA,
        "classification": "FULL_HEAVY_ADAPTATION_ZERO_STEP_PASS",
        "device": str(device),
        "offline_environment": {
            "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
            "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
            "HF_DATASETS_OFFLINE": os.environ["HF_DATASETS_OFFLINE"],
            "esm_model_path": str(esm_path),
        },
        "model_contract": {
            "checkpoint_sha256": before_file_sha["phase2_checkpoint"],
            "state_tensor_count": model_state_tensor_count,
            "state_element_count": model_state_element_count,
            "state_all_finite": model_state_all_finite,
            "trainable_tensor_count": EXPECTED_TRAINABLE_TENSOR_COUNT,
            "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETER_COUNT,
            "trainable_parameter_names_sha256": (
                EXPECTED_TRAINABLE_PARAMETER_NAMES_SHA256
            ),
            "freeze_contract": freeze_contract,
            "last_egnn_coord_mlp_frozen": coord_mlp_frozen,
            "trainable_parameters_all_on_feature_embedding_path": feature_path_only,
            "optimizer_created": False,
            "optimizer_parameter_group_description": parameter_groups,
        },
        "dataset_contract": {
            "train_interface_pair_count": len(train_ids),
            "valid_interface_pair_count": len(valid_ids),
            "train_interface_pair_ids_sha256": sequence_sha256(train_ids),
            "valid_interface_pair_ids_sha256": sequence_sha256(valid_ids),
            "cache_sequence_count": runtime["cache_sequence_count"],
            "cache_conformer_count": 20850,
        },
        "frozen_module_state_sha256_before": before_state,
        "frozen_module_state_sha256_after": after_state,
        "frozen_module_state_unchanged": unchanged,
        "gradients_created": False,
        "input_file_sha256_before": before_file_sha,
        "input_file_sha256_after": after_file_sha,
        "input_files_unchanged": True,
        "optimizer_step_executed": False,
        "backward_executed": False,
        "scheduler_step_executed": False,
        "amp_step_executed": False,
        "checkpoint_written": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(
        output_dir / "adaptation_manifest_validation.json",
        manifest_bytes(validation),
    )
    write_atomic(output_dir / "batch_contract_audit.json", manifest_bytes(batch_audit))
    write_atomic(
        output_dir / "real_model_zero_step_preflight.json", manifest_bytes(report)
    )
    summary = (
        "# Phase-3 v2 bounded full-heavy adaptation zero-step preflight\n\n"
        "- Classification: `FULL_HEAVY_ADAPTATION_ZERO_STEP_PASS`\n"
        f"- Plan: {len(train_ids)} train / {len(valid_ids)} valid interface pairs.\n"
        "- Cache: 2,085 sequences / 20,850 conformers.\n"
        f"- Freeze contract: {EXPECTED_TRAINABLE_TENSOR_COUNT} tensors / "
        f"{EXPECTED_TRAINABLE_PARAMETER_COUNT:,} parameters.\n"
        "- Execution: offline CPU inference only; no optimizer was created, "
        "no gradients/backward/step/checkpoint write occurred.\n"
        "- Both real train and valid batch forward/loss audits were finite.\n"
        "- Training and GPU retrieval remain unauthorized and were not run.\n"
    ).encode("utf-8")
    write_atomic(output_dir / "summary.md", summary)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real-model, optimizer-free bounded full-heavy zero-step preflight."
    )
    parser.add_argument("--adaptation-manifest", required=True)
    parser.add_argument("--plan-descriptor", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--phase2-checkpoint", required=True)
    parser.add_argument("--source-model-configs", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--safe373-evaluation-plan", required=True)
    parser.add_argument("--esm-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    report = run_preflight(parse_args())
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

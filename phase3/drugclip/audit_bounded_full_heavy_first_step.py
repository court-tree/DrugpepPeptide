"""Exactly-one-step audit for Phase-3 v2 bounded full-heavy adaptation.

This is deliberately separate from both the formal runner and historical
diagnostic scripts.  It permits one successful optimizer update, writes one
``step_001.pt``, verifies strict restore, and has no loop capable of reaching
step 2.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Iterable

import torch

from phase3.drugclip.batching import UniquePeptideBatchSampler, collate_phase3
from phase3.drugclip.forward import forward_and_known_positive_loss
from phase3.drugclip.full_heavy_adaptation_contract import (
    FullHeavyDatasetView,
    build_bounded_optimizer_groups,
    configure_bounded_full_heavy_trainable,
    sequence_sha256,
    sha256_file,
    validate_bounded_full_heavy_contract,
)
from phase3.drugclip.losses import build_known_positive_masks
from phase3.drugclip.preflight_bounded_full_heavy_adaptation import (
    EXPECTED_MODEL_STATE_ELEMENT_COUNT,
    EXPECTED_MODEL_STATE_TENSOR_COUNT,
    EXPECTED_TRAINABLE_PARAMETER_COUNT,
    EXPECTED_TRAINABLE_PARAMETER_NAMES_SHA256,
    EXPECTED_TRAINABLE_TENSOR_COUNT,
    module_state_sha256,
)
from phase3.drugclip.random_augmentation_dataset import (
    InterfacePairSubsetDataset,
    Phase3RandomConformerDataset,
)
from phase3.drugclip.train import (
    _build_warmup_scheduler,
    _plan_hash,
    load_phase2_fusion_model,
    load_source_configs,
)
from phase3.drugclip.training_state import (
    amp_is_enabled,
    autocast_context,
    load_training_checkpoint,
    make_grad_scaler,
    save_training_checkpoint,
)


AUDIT_SCHEMA = "phase3-v2-bounded-full-heavy-first-step-audit-v1"
EXPECTED_HEAD = "7db276b96b3fc923a3c291b990928752c964c5e9"
BATCH_SIZE = 16
FUSION_LR = 1e-6
TOWER_LR = 2e-7
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
WARMUP_FRACTION = 0.0
GLOBAL_SEED = 1


class ExactlyOneStepAdamW(torch.optim.AdamW):
    """AdamW with a fail-closed process-local one-step budget."""

    def __init__(self, params: Iterable[Any], **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.successful_step_calls = 0

    def step(self, closure: Any = None) -> Any:
        if self.successful_step_calls >= 1:
            raise RuntimeError("second_optimizer_step_forbidden")
        result = super().step(closure)
        self.successful_step_calls += 1
        return result


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.numpy().tobytes())
    return digest.hexdigest().upper()


def named_parameter_hashes(model: torch.nn.Module) -> dict[str, str]:
    return {
        name: tensor_sha256(parameter)
        for name, parameter in sorted(model.named_parameters())
    }


def named_buffer_hashes(model: torch.nn.Module) -> dict[str, str]:
    return {
        name: tensor_sha256(buffer)
        for name, buffer in sorted(model.named_buffers())
    }


def nested_state_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            digest.update(b"T")
            digest.update(tensor_sha256(item).encode("ascii"))
        elif isinstance(item, dict):
            digest.update(b"D")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"L")
            for child in item:
                visit(child)
        else:
            digest.update(b"V")
            digest.update(repr(item).encode("utf-8"))
        digest.update(b"\0")

    visit(value)
    return digest.hexdigest().upper()


def analyze_parameter_changes(
    before_hashes: dict[str, str],
    after_hashes: dict[str, str],
    allowed_names: Iterable[str],
) -> dict[str, list[str]]:
    if set(before_hashes) != set(after_hashes):
        raise AssertionError("named_parameter_set_changed")
    allowed = set(allowed_names)
    changed = sorted(
        name for name in before_hashes if before_hashes[name] != after_hashes[name]
    )
    forbidden = sorted(set(changed) - allowed)
    if forbidden:
        raise AssertionError(f"forbidden_parameter_changed:{forbidden}")
    return {
        "changed_allowed": sorted(set(changed) & allowed),
        "unchanged_allowed": sorted(allowed - set(changed)),
        "changed_forbidden": forbidden,
    }


def validate_optimizer_state(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    allowed_names: Iterable[str],
) -> dict[str, Any]:
    allowed = set(allowed_names)
    name_by_parameter = {id(parameter): name for name, parameter in model.named_parameters()}
    state_names = sorted(name_by_parameter[id(parameter)] for parameter in optimizer.state)
    if set(state_names) != allowed:
        raise AssertionError("optimizer_state_parameter_scope_mismatch")
    steps: dict[str, int] = {}
    for parameter, state in optimizer.state.items():
        name = name_by_parameter[id(parameter)]
        raw_step = state.get("step")
        step = int(raw_step.item()) if torch.is_tensor(raw_step) else int(raw_step)
        if step != 1:
            raise AssertionError(f"optimizer_state_step_mismatch:{name}:{step}")
        steps[name] = step
    return {
        "state_tensor_count": len(state_names),
        "parameter_names": state_names,
        "parameter_names_sha256": sequence_sha256(state_names),
        "allowed_parameter_names_without_state": [],
        "steps": steps,
        "all_steps_equal_one": True,
    }


def validate_trainable_gradients(model: torch.nn.Module) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    missing: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise AssertionError(f"frozen_parameter_received_gradient:{name}")
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise FloatingPointError(f"nonfinite_gradient:{name}")
        audit[name] = {
            "present": True,
            "finite": True,
            "maximum_absolute_gradient": float(parameter.grad.detach().abs().max().item()),
            "mean_absolute_gradient": float(parameter.grad.detach().abs().mean().item()),
        }
    if missing:
        raise AssertionError(f"trainable_parameter_missing_gradient:{missing}")
    return audit


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale_temporary_output:{temporary}")
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _formal_paths(dataset_root: Path) -> dict[str, Path]:
    return {
        "random_pairs": dataset_root / "04_training_input" / "random_conformer_pairs.jsonl",
        "random_cache": dataset_root
        / "03_random_conformer_cache"
        / "random_conformer_cache.jsonl",
        "biological_pairs": dataset_root / "dependencies" / "biological_pairs.jsonl",
        "pair_splits": dataset_root
        / "02_leakage_safe_split"
        / "pair_splits.jsonl",
        "manifest": dataset_root / "DATA_MANIFEST.json",
    }


def _load_model(
    checkpoint: Path,
    source_configs_path: Path,
    esm_path: Path,
    device: torch.device,
    repo_root: Path,
) -> torch.nn.Module:
    configs = load_source_configs(checkpoint, str(source_configs_path), repo_root)
    model_args = SimpleNamespace(
        hf_model_name_or_path_1d=str(esm_path),
        fusion_hidden_dim=None,
        fusion_output_dim=None,
        dropout=None,
        temperature=None,
    )
    return load_phase2_fusion_model(checkpoint, configs, device, model_args, repo_root)


def _component_embedding_hashes(
    model: torch.nn.Module, forward: dict[str, Any]
) -> dict[str, str]:
    moved = forward["moved_batch"]
    one_d = moved["one_d"]
    three_d = moved["three_d"]
    with torch.inference_mode():
        receptor_1d = model.model_1d.encode_receptor(
            receptor_tokens=one_d["receptor_tokens"],
            receptor_sequences=one_d["receptor_sequence"],
        )
        peptide_1d = model.model_1d.encode_peptide(
            peptide_tokens=one_d["peptide_tokens"],
            peptide_sequences=one_d["peptide_sequence"],
        )
        receptor_3d = model.model_3d.encode_receptor(
            receptor_coords=three_d["receptor_coords"],
            receptor_elements=three_d["receptor_elements"],
            receptor_mask=three_d["receptor_mask"],
            receptor_atom_names=three_d["receptor_atom_names"],
            receptor_residue_names=three_d["receptor_residue_names"],
        )
    return {
        "receptor_1d": tensor_sha256(receptor_1d),
        "receptor_3d": tensor_sha256(receptor_3d),
        "receptor_fused": tensor_sha256(forward["receptor_embedding"]),
        "peptide_1d": tensor_sha256(peptide_1d),
        "peptide_fused": tensor_sha256(forward["peptide_embedding"]),
        "logits": tensor_sha256(forward["similarity_matrix"]),
    }


def _finite_forward_summary(forward: dict[str, Any]) -> dict[str, Any]:
    tensors = {
        "receptor_embedding": forward["receptor_embedding"],
        "peptide_embedding": forward["peptide_embedding"],
        "logits_receptor_to_peptide": forward["similarity_matrix"],
        "logits_peptide_to_receptor": forward["similarity_matrix_transpose"],
        "loss_total": forward["loss_total"],
        "loss_receptor_to_peptide": forward["loss_receptor_to_peptide"],
        "loss_peptide_to_receptor": forward["loss_peptide_to_receptor"],
    }
    finite = {key: bool(torch.isfinite(value).all().item()) for key, value in tensors.items()}
    if not all(finite.values()):
        raise FloatingPointError("nonfinite_forward_or_loss")
    return {
        "finite": finite,
        "shapes": {key: list(value.shape) for key, value in tensors.items()},
        "loss_total": float(forward["loss_total"].item()),
        "loss_receptor_to_peptide": float(
            forward["loss_receptor_to_peptide"].item()
        ),
        "loss_peptide_to_receptor": float(
            forward["loss_peptide_to_receptor"].item()
        ),
    }


def _parameter_deltas(
    before: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, dict[str, float | bool]]:
    current = dict(model.named_parameters())
    result: dict[str, dict[str, float | bool]] = {}
    for name, old in sorted(before.items()):
        new = current[name].detach().cpu()
        delta = (new - old).abs().float()
        result[name] = {
            "changed": not torch.equal(old, new),
            "maximum_absolute_change": float(delta.max().item()),
            "mean_absolute_change": float(delta.mean().item()),
            "finite_after": bool(torch.isfinite(new).all().item()),
        }
    return result


def _build_run_config(
    *,
    train_base: Phase3RandomConformerDataset,
    valid_base: Phase3RandomConformerDataset,
    train_dataset: InterfacePairSubsetDataset,
    valid_dataset: InterfacePairSubsetDataset,
    train_sampler: UniquePeptideBatchSampler,
    freeze_contract: dict[str, Any],
    runtime_contract: dict[str, Any],
    input_paths: dict[str, Path],
    validation_plan_sha256: str,
) -> dict[str, Any]:
    return {
        "command": "phase3.drugclip.audit_bounded_full_heavy_first_step",
        "phase2_checkpoint": str(input_paths["phase2_checkpoint"]),
        "data_version": train_base.data_contract.data_version,
        "dataset_version": train_base.data_contract.dataset_version,
        "dataset_root": str(train_base.data_contract.dataset_root),
        "data_manifest_path": str(train_base.data_contract.manifest_path),
        "data_manifest_sha256": train_base.data_contract.manifest_sha256,
        "database_contract": train_base.data_contract.database_contract,
        "cache_schema": train_base.data_contract.cache_schema,
        "generator_id": train_base.data_contract.generator_id,
        "qc_id": train_base.data_contract.qc_id,
        "random_pairs_sha256": sha256_file(train_base.random_pairs_jsonl),
        "random_conformer_cache_sha256": sha256_file(
            train_base.random_conformer_cache_jsonl
        ),
        "pair_splits_sha256": sha256_file(train_base.pair_splits_jsonl),
        "random_pairs_jsonl": str(train_base.random_pairs_jsonl),
        "valid_random_pairs_jsonl": str(valid_base.random_pairs_jsonl),
        "random_conformer_cache_jsonl": str(train_base.random_conformer_cache_jsonl),
        "biological_pairs_jsonl": str(train_base.biological_pairs_jsonl),
        "biological_pairs_sha256": train_base.mapping_summary()[
            "biological_pairs_sha256"
        ],
        "pair_splits_jsonl": str(train_base.pair_splits_jsonl),
        "relation_schema": train_base.data_contract.relation_schema,
        "sampling_unit": "interface_pair",
        "train_interface_pair_ids": train_dataset.interface_pair_ids,
        "valid_interface_pair_ids": valid_dataset.interface_pair_ids,
        "train_interface_pair_ids_sha256": sequence_sha256(
            train_dataset.interface_pair_ids
        ),
        "valid_interface_pair_ids_sha256": sequence_sha256(
            valid_dataset.interface_pair_ids
        ),
        "fixed_validation_plan_sha256": validation_plan_sha256,
        "total_train_steps": 1,
        "warmup_fraction": WARMUP_FRACTION,
        "warmup_steps": 0,
        "scheduler_kind": "linear_warmup_constant",
        "global_seed": GLOBAL_SEED,
        "freeze_configuration": {
            "freeze_all_towers": False,
            "unfreeze_1d": {"requested_layers": 0, "trainable_parameters": 0},
            "unfreeze_3d": {
                "requested_layers": 1,
                "scope": "peptide_encoder_only",
            },
            "bounded_full_heavy": freeze_contract,
        },
        "full_heavy_data_contract": runtime_contract,
        "amp_enabled": True,
        "first_step_audit_contract": {
            "schema_version": AUDIT_SCHEMA,
            "maximum_successful_optimizer_steps": 1,
            "batch_size": BATCH_SIZE,
            "fusion_lr": FUSION_LR,
            "tower_lr": TOWER_LR,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "plan_descriptor_sha256": sha256_file(input_paths["plan_descriptor"]),
            "cache_manifest_sha256": sha256_file(input_paths["cache_manifest"]),
            "adaptation_manifest_sha256": sha256_file(
                input_paths["adaptation_manifest"]
            ),
        },
        "train_sampler_summary": train_sampler.summary(),
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[2]
    actual_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if actual_head != EXPECTED_HEAD:
        raise ValueError(f"git_head_mismatch:{actual_head}")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output_dir_already_exists:{output_dir}")
    output_dir.mkdir(parents=True)
    input_paths = {
        "phase2_checkpoint": Path(args.phase2_checkpoint).resolve(),
        "adaptation_manifest": Path(args.adaptation_manifest).resolve(),
        "plan_descriptor": Path(args.plan_descriptor).resolve(),
        "cache_manifest": Path(args.cache_manifest).resolve(),
        "source_model_configs": Path(args.source_model_configs).resolve(),
        "safe373_evaluation_plan": Path(args.safe373_evaluation_plan).resolve(),
    }
    dataset_root = Path(args.dataset_root).resolve()
    formal = _formal_paths(dataset_root)
    input_paths["random_conformer_v3_manifest"] = formal["manifest"]
    for path in (*input_paths.values(), dataset_root, Path(args.esm_model).resolve()):
        if not path.exists():
            raise FileNotFoundError(path)
    before_input_sha = {key: sha256_file(path) for key, path in input_paths.items()}

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)
    try:
        import numpy as np

        np.random.seed(GLOBAL_SEED)
    except ImportError:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = amp_is_enabled(device, True)
    model = _load_model(
        input_paths["phase2_checkpoint"],
        input_paths["source_model_configs"],
        Path(args.esm_model).resolve(),
        device,
        repo_root,
    )
    model_state = model.state_dict()
    if (
        len(model_state) != EXPECTED_MODEL_STATE_TENSOR_COUNT
        or sum(value.numel() for value in model_state.values())
        != EXPECTED_MODEL_STATE_ELEMENT_COUNT
        or not all(
            bool(torch.isfinite(value).all().item())
            for value in model_state.values()
            if value.is_floating_point() or value.is_complex()
        )
    ):
        raise ValueError("real_model_state_contract_mismatch")
    freeze_contract = configure_bounded_full_heavy_trainable(model)
    if (
        len(freeze_contract["trainable_parameter_names"])
        != EXPECTED_TRAINABLE_TENSOR_COUNT
        or freeze_contract["trainable_parameter_count"]
        != EXPECTED_TRAINABLE_PARAMETER_COUNT
        or freeze_contract["trainable_parameter_names_sha256"]
        != EXPECTED_TRAINABLE_PARAMETER_NAMES_SHA256
    ):
        raise ValueError("freeze_contract_mismatch")

    dataset_kwargs = {
        "random_pairs_jsonl": formal["random_pairs"],
        "random_conformer_cache_jsonl": formal["random_cache"],
        "biological_pairs_jsonl": formal["biological_pairs"],
        "pair_splits_jsonl": formal["pair_splits"],
        "data_version": "v3",
        "dataset_root": dataset_root,
        "expected_manifest_sha256": before_input_sha[
            "random_conformer_v3_manifest"
        ],
    }
    train_base = Phase3RandomConformerDataset(
        **dataset_kwargs,
        split="train",
        mode="train_random",
        global_seed=GLOBAL_SEED,
    )
    valid_base = Phase3RandomConformerDataset(
        **dataset_kwargs,
        split="valid",
        mode="fixed",
        global_seed=GLOBAL_SEED + 17,
        fixed_conformer_index=0,
    )
    runtime = validate_bounded_full_heavy_contract(
        input_paths["adaptation_manifest"],
        plan_descriptor_file=input_paths["plan_descriptor"],
        cache_manifest_file=input_paths["cache_manifest"],
        phase2_checkpoint=input_paths["phase2_checkpoint"],
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
    train_dataset = InterfacePairSubsetDataset(
        FullHeavyDatasetView(train_base, payloads),
        runtime["train_interface_pair_ids"],
    )
    valid_dataset = InterfacePairSubsetDataset(
        FullHeavyDatasetView(valid_base, payloads),
        runtime["valid_interface_pair_ids"],
    )
    train_sampler = UniquePeptideBatchSampler(
        train_dataset, BATCH_SIZE, seed=GLOBAL_SEED, epoch=0
    )
    valid_dataset.set_epoch(0)
    train_plan_hash = _plan_hash(train_dataset.epoch_plan())
    validation_plan_sha = _plan_hash(valid_dataset.epoch_plan())
    first_indices = next(iter(train_sampler))
    if len(first_indices) != BATCH_SIZE:
        raise ValueError("first_train_batch_size_mismatch")
    batch = collate_phase3([train_dataset[index] for index in first_indices])
    sequences = [str(value) for value in batch["one_d"]["peptide_sequence"]]
    if len(sequences) != len(set(sequences)):
        raise ValueError("first_train_batch_peptide_uniqueness_failure")
    r2p_mask, p2r_mask = build_known_positive_masks(
        list(batch["one_d"]["receptor_interface_id"]),
        sequences,
        list(batch["one_d"]["known_positive_group"]),
        device,
    )
    known_positive_audit = {
        "status": "PASS",
        "receptor_to_peptide_masked_count": int(r2p_mask.sum().item()),
        "peptide_to_receptor_masked_count": int(p2r_mask.sum().item()),
        "diagonal_masked_count": int(
            (
                r2p_mask.diagonal().sum() + p2r_mask.diagonal().sum()
            ).item()
        ),
    }
    if known_positive_audit["diagonal_masked_count"] != 0:
        raise ValueError("known_positive_diagonal_mask_failure")
    batch_audit = {
        "schema_version": AUDIT_SCHEMA,
        "interface_pair_ids": list(batch["one_d"]["interface_pair_id"]),
        "peptide_sequences": sequences,
        "conformer_indices": list(batch["one_d"]["conformer_index"]),
        "peptide_unique": True,
        "known_positive_mask": known_positive_audit,
        "train_plan_sha256": train_plan_hash,
        "batch_size": BATCH_SIZE,
    }
    _atomic_json(output_dir / "batch_contract_audit.json", batch_audit)
    _atomic_json(
        output_dir / "execution_state.json",
        {
            "schema_version": AUDIT_SCHEMA,
            "phase": "pre_step_validated",
            "optimizer_step_count": 0,
            "scheduler_step_count": 0,
        },
    )

    before_parameter_hash = named_parameter_hashes(model)
    before_buffer_hash = named_buffer_hashes(model)
    allowed_names = freeze_contract["trainable_parameter_names"]
    before_allowed = {
        name: dict(model.named_parameters())[name].detach().cpu().clone()
        for name in allowed_names
    }
    frozen_modules = {
        "receptor_1d": model.model_1d.receptor_encoder,
        "receptor_3d": model.model_3d.receptor_encoder,
        "receptor_fusion": model.receptor_fusion,
        "peptide_1d": model.model_1d.peptide_encoder,
        "peptide_3d_last_coord_mlp": (
            model.model_3d.peptide_encoder.layers[-1].coord_mlp
        ),
    }
    before_module_hash = {
        key: module_state_sha256(module) for key, module in frozen_modules.items()
    }
    temperature_before = float(model.temperature)
    model.eval()
    with torch.inference_mode():
        before_forward = forward_and_known_positive_loss(model, batch, device)
        before_forward_summary = _finite_forward_summary(before_forward)
        before_embedding_hash = _component_embedding_hashes(model, before_forward)
        receptor_embedding_before = before_forward["receptor_embedding"].detach().cpu().clone()
        peptide_embedding_before = before_forward["peptide_embedding"].detach().cpu().clone()

    parameter_groups = build_bounded_optimizer_groups(
        model, fusion_lr=FUSION_LR, tower_lr=TOWER_LR
    )
    optimizer = ExactlyOneStepAdamW(parameter_groups, weight_decay=WEIGHT_DECAY)
    selected_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if selected_ids != expected_ids or len(selected_ids) != EXPECTED_TRAINABLE_TENSOR_COUNT:
        raise AssertionError("optimizer_parameter_scope_mismatch")
    scheduler, warmup_steps = _build_warmup_scheduler(
        optimizer, total_train_steps=1, warmup_fraction=WARMUP_FRACTION
    )
    if warmup_steps != 0:
        raise AssertionError("unexpected_warmup_steps")
    scaler = make_grad_scaler(device, use_amp)

    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, use_amp):
        update_forward = forward_and_known_positive_loss(model, batch, device)
        update_loss = update_forward["loss_total"]
    if not bool(torch.isfinite(update_loss).item()):
        raise FloatingPointError("nonfinite_update_loss")
    scale_before = float(scaler.get_scale())
    scaler.scale(update_loss).backward()
    scaler.unscale_(optimizer)
    gradient_audit = validate_trainable_gradients(model)
    pre_clip_norm = float(
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            MAX_GRAD_NORM,
        ).item()
    )
    if not math.isfinite(pre_clip_norm):
        raise FloatingPointError("nonfinite_gradient_norm")
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    if optimizer.successful_step_calls != 1 or scale_after < scale_before:
        raise FloatingPointError("amp_skip_or_optimizer_step_failure")
    scheduler.step()
    if scheduler.last_epoch != 1:
        raise AssertionError("scheduler_step_count_mismatch")
    second_step_hard_gate_rejected = False
    try:
        optimizer.step()
    except RuntimeError as error:
        if "second_optimizer_step_forbidden" not in str(error):
            raise
        second_step_hard_gate_rejected = True
    if not second_step_hard_gate_rejected:
        raise AssertionError("second_optimizer_step_hard_gate_failed")
    global_step = 1
    _atomic_json(
        output_dir / "execution_state.json",
        {
            "schema_version": AUDIT_SCHEMA,
            "phase": "single_step_completed",
            "optimizer_step_count": 1,
            "scheduler_step_count": 1,
        },
    )
    optimizer.zero_grad(set_to_none=True)

    after_parameter_hash = named_parameter_hashes(model)
    after_buffer_hash = named_buffer_hashes(model)
    changes = analyze_parameter_changes(
        before_parameter_hash, after_parameter_hash, allowed_names
    )
    if after_buffer_hash != before_buffer_hash:
        changed_buffers = sorted(
            name
            for name in before_buffer_hash
            if before_buffer_hash[name] != after_buffer_hash.get(name)
        )
        raise AssertionError(f"buffer_changed:{changed_buffers}")
    deltas = _parameter_deltas(before_allowed, model)
    changed_allowed_count = sum(bool(value["changed"]) for value in deltas.values())
    if changed_allowed_count < 1:
        raise AssertionError("optimizer_updated_no_allowed_parameters")
    changed_allowed_names = {
        name for name, value in deltas.items() if bool(value["changed"])
    }
    if not any(
        name.startswith("model_3d.peptide_encoder.")
        for name in changed_allowed_names
    ):
        raise AssertionError("peptide_feature_path_parameter_did_not_change")
    if not any(name.startswith("peptide_fusion.") for name in changed_allowed_names):
        raise AssertionError("peptide_fusion_parameter_did_not_change")
    optimizer_audit = validate_optimizer_state(optimizer, model, allowed_names)

    after_module_hash = {
        key: module_state_sha256(module) for key, module in frozen_modules.items()
    }
    if before_module_hash != after_module_hash:
        raise AssertionError("frozen_directional_module_state_changed")
    if float(model.temperature) != temperature_before:
        raise AssertionError("temperature_changed")
    model.eval()
    with torch.inference_mode():
        after_forward = forward_and_known_positive_loss(model, batch, device)
        after_forward_summary = _finite_forward_summary(after_forward)
        after_embedding_hash = _component_embedding_hashes(model, after_forward)
        receptor_embedding_after = after_forward["receptor_embedding"].detach().cpu()
        peptide_embedding_after = after_forward["peptide_embedding"].detach().cpu()
    if not torch.equal(receptor_embedding_before, receptor_embedding_after):
        raise AssertionError("receptor_embedding_changed")
    if torch.equal(peptide_embedding_before, peptide_embedding_after):
        raise AssertionError("peptide_full_heavy_embedding_did_not_change")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("gradient_not_cleared_after_step")

    runtime_checkpoint_contract = copy.deepcopy(runtime)
    run_config = _build_run_config(
        train_base=train_base,
        valid_base=valid_base,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        train_sampler=train_sampler,
        freeze_contract=freeze_contract,
        runtime_contract=runtime_checkpoint_contract,
        input_paths=input_paths,
        validation_plan_sha256=validation_plan_sha,
    )
    sampler_state = {
        "completed_epoch": None,
        "current_epoch": 0,
        "batches_completed_in_epoch": 1,
        "train_plan_hash": train_plan_hash,
        "next_epoch": 0,
        "next_train_plan_hash": train_plan_hash,
        "validation_epoch": 0,
        "validation_plan_hash": validation_plan_sha,
        "best_validation_epoch": None,
        "summary": train_sampler.summary(),
    }
    checkpoint_path = output_dir / "step_001.pt"
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=0,
        global_step=global_step,
        global_seed=GLOBAL_SEED,
        best_validation_loss=float(before_forward_summary["loss_total"]),
        run_config=run_config,
        sampler_state=sampler_state,
        history=[
            {
                "audit": AUDIT_SCHEMA,
                "global_step": 1,
                "batches_completed_in_epoch": 1,
                "loss_total": float(update_loss.detach().float().item()),
            }
        ],
    )
    if sorted(path.name for path in output_dir.glob("*.pt")) != ["step_001.pt"]:
        raise AssertionError("unexpected_checkpoint_set")
    checkpoint_sha = sha256_file(checkpoint_path)
    saved_model_sha = nested_state_sha256(model.state_dict())
    saved_optimizer_sha = nested_state_sha256(optimizer.state_dict())
    saved_scheduler_sha = nested_state_sha256(scheduler.state_dict())
    saved_scaler_sha = nested_state_sha256(scaler.state_dict())

    del payloads, before_forward, update_forward, after_forward
    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    restore_device = device
    restored_model = _load_model(
        input_paths["phase2_checkpoint"],
        input_paths["source_model_configs"],
        Path(args.esm_model).resolve(),
        restore_device,
        repo_root,
    )
    restored_freeze = configure_bounded_full_heavy_trainable(restored_model)
    restored_groups = build_bounded_optimizer_groups(
        restored_model, fusion_lr=FUSION_LR, tower_lr=TOWER_LR
    )
    restored_optimizer = ExactlyOneStepAdamW(
        restored_groups, weight_decay=WEIGHT_DECAY
    )
    restored_scheduler, _ = _build_warmup_scheduler(
        restored_optimizer, total_train_steps=1, warmup_fraction=WARMUP_FRACTION
    )
    restored_scaler = make_grad_scaler(restore_device, use_amp)
    restored = load_training_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        expected_run_config=run_config,
        device=restore_device,
    )
    if (
        int(restored["global_step"]) != 1
        or int(restored["sampler_state"]["batches_completed_in_epoch"]) != 1
        or restored["sampler_state"]["next_train_plan_hash"] != train_plan_hash
        or restored_freeze["trainable_parameter_names_sha256"]
        != EXPECTED_TRAINABLE_PARAMETER_NAMES_SHA256
        or nested_state_sha256(restored_model.state_dict()) != saved_model_sha
        or nested_state_sha256(restored_optimizer.state_dict())
        != saved_optimizer_sha
        or nested_state_sha256(restored_scheduler.state_dict())
        != saved_scheduler_sha
        or nested_state_sha256(restored_scaler.state_dict()) != saved_scaler_sha
    ):
        raise AssertionError("checkpoint_restore_state_mismatch")
    mismatch_config = copy.deepcopy(run_config)
    mismatch_config["full_heavy_data_contract"] = {
        **mismatch_config["full_heavy_data_contract"],
        "adaptation_manifest_canonical_sha256": "0" * 64,
    }
    mismatch_rejected = False
    try:
        load_training_checkpoint(
            checkpoint_path,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            scaler=restored_scaler,
            expected_run_config=mismatch_config,
            device=restore_device,
        )
    except ValueError as error:
        if "full_heavy_data_contract" not in str(error):
            raise
        mismatch_rejected = True
    if not mismatch_rejected:
        raise AssertionError("checkpoint_contract_mismatch_not_rejected")
    if restored_optimizer.successful_step_calls != 0:
        raise AssertionError("restore_executed_optimizer_step")

    after_input_sha = {key: sha256_file(path) for key, path in input_paths.items()}
    if before_input_sha != after_input_sha:
        raise AssertionError("bound_input_file_changed")
    parameter_audit = {
        "schema_version": AUDIT_SCHEMA,
        "trainable_tensor_count": len(allowed_names),
        "trainable_parameter_count": freeze_contract["trainable_parameter_count"],
        "trainable_parameter_names_sha256": freeze_contract[
            "trainable_parameter_names_sha256"
        ],
        "before_named_parameter_sha256": before_parameter_hash,
        "after_named_parameter_sha256": after_parameter_hash,
        "before_named_buffer_sha256": before_buffer_hash,
        "after_named_buffer_sha256": after_buffer_hash,
        "changes": changes,
        "allowed_parameter_deltas": deltas,
        "changed_allowed_tensor_count": changed_allowed_count,
        "forbidden_changed_tensor_count": len(changes["changed_forbidden"]),
        "gradient_audit": gradient_audit,
        "pre_clip_gradient_norm": pre_clip_norm,
        "max_grad_norm": MAX_GRAD_NORM,
        "optimizer_state": optimizer_audit,
    }
    report = {
        "schema_version": AUDIT_SCHEMA,
        "classification": "FULL_HEAVY_FIRST_STEP_PASS",
        "git_head": EXPECTED_HEAD,
        "device": str(device),
        "amp_enabled": use_amp,
        "global_step": 1,
        "successful_optimizer_step_count": 1,
        "scheduler_step_count": 1,
        "checkpoint": {
            "path": "step_001.pt",
            "file_sha256": checkpoint_sha,
            "global_step": 1,
            "restore_state_match": True,
            "next_batch_offset": 1,
            "contract_mismatch_rejected": True,
            "second_step_executed": False,
            "second_step_hard_gate_rejected": second_step_hard_gate_rejected,
        },
        "model_contract": {
            "state_tensor_count": EXPECTED_MODEL_STATE_TENSOR_COUNT,
            "state_element_count": EXPECTED_MODEL_STATE_ELEMENT_COUNT,
            "trainable_tensor_count": EXPECTED_TRAINABLE_TENSOR_COUNT,
            "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETER_COUNT,
            "trainable_parameter_names_sha256": (
                EXPECTED_TRAINABLE_PARAMETER_NAMES_SHA256
            ),
        },
        "forward_before": before_forward_summary,
        "forward_update": {
            "loss_total": float(update_loss.detach().float().item()),
            "loss_finite": True,
        },
        "forward_after": after_forward_summary,
        "embedding_sha256_before": before_embedding_hash,
        "embedding_sha256_after": after_embedding_hash,
        "frozen_embedding_hashes_unchanged": {
            key: before_embedding_hash[key] == after_embedding_hash[key]
            for key in ("receptor_1d", "receptor_3d", "receptor_fused", "peptide_1d")
        },
        "receptor_embedding_bitwise_unchanged": True,
        "peptide_full_heavy_embedding_changed": True,
        "peptide_feature_path_parameter_changed": True,
        "peptide_fusion_parameter_changed": True,
        "coord_mlp_bitwise_unchanged": (
            before_module_hash["peptide_3d_last_coord_mlp"]
            == after_module_hash["peptide_3d_last_coord_mlp"]
        ),
        "frozen_module_state_sha256_before": before_module_hash,
        "frozen_module_state_sha256_after": after_module_hash,
        "temperature_before": temperature_before,
        "temperature_after": float(restored_model.temperature),
        "input_file_sha256_before": before_input_sha,
        "input_file_sha256_after": after_input_sha,
        "input_files_unchanged": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(output_dir / "parameter_update_audit.json", parameter_audit)
    _atomic_json(output_dir / "first_step_audit_report.json", report)
    summary = (
        "# Phase-3 v2 bounded full-heavy first optimizer-step audit\n\n"
        "- Classification: `FULL_HEAVY_FIRST_STEP_PASS`\n"
        "- Exactly one successful optimizer step and one scheduler step.\n"
        f"- Allowed tensors changed: {changed_allowed_count}/{len(allowed_names)}; "
        "forbidden tensors changed: 0.\n"
        "- Receptor embedding remained bitwise identical; peptide full-heavy "
        "embedding changed and remained finite.\n"
        "- `step_001.pt` restored exactly with batch offset 1; a mismatched "
        "full-heavy contract was rejected.\n"
        "- No step 2, step 32, retrieval, or further training was run.\n"
    )
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exactly one Phase-3 v2 bounded full-heavy optimizer-step audit."
    )
    parser.add_argument("--phase2-checkpoint", required=True)
    parser.add_argument("--adaptation-manifest", required=True)
    parser.add_argument("--plan-descriptor", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--source-model-configs", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--safe373-evaluation-plan", required=True)
    parser.add_argument("--esm-model", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_audit(args)
    except Exception as error:
        message = str(error)
        if any(
            token in message
            for token in (
                "optimizer_parameter",
                "optimizer_state_parameter_scope",
                "freeze_contract",
            )
        ):
            classification = "OPTIMIZER_SCOPE_FAIL"
        elif any(
            token in message
            for token in (
                "nonfinite",
                "amp_skip",
                "did_not_change",
                "received_gradient",
                "gradient",
            )
        ):
            classification = "NUMERICAL_UPDATE_FAIL"
        elif any(
            token in message
            for token in ("checkpoint", "restore_state", "contract_mismatch")
        ):
            classification = "CHECKPOINT_RESUME_CONTRACT_FAIL"
        else:
            classification = "OPERATIONAL_FAILURE"
        output_dir = Path(args.output_dir).resolve()
        if output_dir.exists():
            _atomic_json(
                output_dir / "failure.json",
                {
                    "schema_version": AUDIT_SCHEMA,
                    "classification": classification,
                    "exception_type": type(error).__name__,
                    "exception_text": str(error),
                },
            )
        raise
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "global_step": result["global_step"],
                "checkpoint_sha256": result["checkpoint"]["file_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

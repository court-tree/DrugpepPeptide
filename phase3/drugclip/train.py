"""DrugCLIP-style Phase-3 training with sequence-only random conformers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shlex
import sys
from pathlib import Path
from typing import Any, Callable
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from phase2.pepclip.data import (
    AA_TO_ID,
    ATOM_NAME_TO_ID,
    ELEMENT_TO_ID,
    RESIDUE_NAME_TO_ID,
)
from phase2.pepclip.model import PepCLIPModel
from phase2.pepclip.model_3d import PepCLIP3DModel
from phase2.pepclip.train_concat_fusion import (
    PepCLIPConcatFusionModel,
    set_frozen,
    unfreeze_1d_last_layers,
    unfreeze_3d_last_layers,
)
from phase3.drugclip.io_utils import write_jsonl
from phase3.drugclip.random_augmentation_dataset import (
    Phase3RandomConformerDataset,
    InterfacePairSubsetDataset,
    sha256_file,
)
from phase3.drugclip.batching import UniquePeptideBatchSampler, collate_phase3
from phase3.drugclip.config import DEFAULT_RUN_ROOT
from phase3.drugclip.forward import forward_and_known_positive_loss
from phase3.drugclip.full_heavy_adaptation_contract import (
    FullHeavyDatasetView,
    build_bounded_optimizer_groups,
    configure_bounded_full_heavy_trainable,
    validate_bounded_full_heavy_contract,
)
from phase3.drugclip.training_state import (
    amp_is_enabled,
    autocast_context,
    load_training_checkpoint,
    make_grad_scaler,
    save_training_checkpoint,
)


DEFAULT_PHASE2_FINAL_CHECKPOINT = (
    "phase2/runs/v9_concat_fusion_partial_unfreeze_1d3d_last1_from3d_e40_v1/checkpoint_best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Phase-2 fusion on interface positives with random peptide conformers.")
    parser.add_argument("--train_random_conformer_pairs", required=True)
    parser.add_argument("--valid_random_conformer_pairs", required=True)
    parser.add_argument("--random_conformer_cache", required=True)
    parser.add_argument("--biological_pairs_jsonl", required=True)
    parser.add_argument("--pair_splits_jsonl", required=True)
    parser.add_argument("--data-version", choices=("v2", "v3"), default="v2")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--expected_manifest_sha256", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--phase2_checkpoint", default=DEFAULT_PHASE2_FINAL_CHECKPOINT)
    parser.add_argument("--source_model_configs", default=None)
    parser.add_argument("--hf_model_name_or_path_1d", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--tower_lr", type=float, default=2e-7)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--fusion_hidden_dim", type=int, default=None)
    parser.add_argument("--fusion_output_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--unfreeze_1d_last_n_layers", type=int, default=1)
    parser.add_argument("--unfreeze_3d_last_n_layers", type=int, default=1)
    parser.add_argument("--freeze_all_towers", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--scheduler_gamma", type=float, default=1.0)
    parser.add_argument("--warmup_fraction", type=float, default=0.0)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, nargs="+", default=None)
    parser.add_argument("--stop_after_epoch", type=int, default=None)
    parser.add_argument("--train_interface_pair_limit", type=int, default=None)
    parser.add_argument("--valid_interface_pair_limit", type=int, default=None)
    parser.add_argument(
        "--full-heavy-adaptation-manifest",
        default=None,
        help=(
            "Final Phase-3 v2 adaptation manifest binding the frozen plan, "
            "materialized cache, freeze contract, and Phase-2 checkpoint."
        ),
    )
    parser.add_argument(
        "--full-heavy-plan-descriptor",
        default=None,
        help=(
            "Frozen Phase-3 v2 explicit bounded plan descriptor. This alone "
            "cannot start training while its cache status is NOT_BUILT."
        ),
    )
    parser.add_argument(
        "--full-heavy-cache-manifest",
        default=None,
        help=(
            "Materialized bounded train/valid full-heavy cache manifest; "
            "safe265/safe373 evaluation caches are forbidden."
        ),
    )
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def resolve_hf_path(config: dict[str, Any], override: str | None, repo_root: Path) -> None:
    if override:
        config["hf_model_name_or_path"] = override
        return
    value = str(config.get("hf_model_name_or_path") or "")
    if not value or Path(value).exists():
        return
    local = repo_root / "models" / Path(value).name
    if local.exists():
        config["hf_model_name_or_path"] = str(local)


def load_source_configs(phase2_checkpoint: Path, explicit_path: str | None, repo_root: Path) -> dict[str, Any]:
    config_path = Path(explicit_path) if explicit_path else phase2_checkpoint.parent / "source_model_configs.json"
    config_path = resolve_path(config_path, repo_root)
    if not config_path.exists():
        raise FileNotFoundError(f"source_model_configs.json not found: {config_path}")
    return read_json(config_path)


def build_1d_model_from_config(config: dict[str, Any], device: torch.device) -> PepCLIPModel:
    return PepCLIPModel(
        vocab_size=max(AA_TO_ID.values()) + 1,
        encoder_type=config.get("encoder_type", "mean_pool"),
        hf_model_name_or_path=config.get("hf_model_name_or_path"),
        freeze_hf_backbone=bool(config.get("freeze_hf_backbone", True)),
        hf_max_length=int(config.get("hf_max_length", 512)),
        embed_dim=int(config.get("embed_dim", 128)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        output_dim=int(config.get("output_dim", 128)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)


def build_3d_model_from_config(config: dict[str, Any], device: torch.device) -> PepCLIP3DModel:
    return PepCLIP3DModel(
        num_elements=max(ELEMENT_TO_ID.values()) + 1,
        num_atom_names=max(ATOM_NAME_TO_ID.values()) + 1,
        num_residue_names=max(RESIDUE_NAME_TO_ID.values()) + 1,
        encoder_type=config.get("encoder_type", "radial"),
        element_dim=int(config.get("element_dim", 32)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        output_dim=int(config.get("output_dim", 128)),
        dropout=float(config.get("dropout", 0.1)),
        coord_scale=float(config.get("coord_scale", 10.0)),
        num_layers=int(config.get("num_layers", 4)),
        num_heads=int(config.get("num_heads", 8)),
        num_rbf=int(config.get("num_rbf", 32)),
        distance_cutoff=float(config.get("distance_cutoff", 20.0)),
        num_neighbors=int(config.get("num_neighbors", 32)),
    ).to(device)


def load_tower_state_compatible(module: nn.Module, state_dict: dict[str, Any], label: str) -> None:
    """Load tower state while tolerating ESM rotary-buffer naming drift."""

    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    allowed_missing = [
        key
        for key in missing
        if key.endswith("attention.self.rotary_embeddings.inv_freq")
    ]
    allowed_unexpected = [
        key
        for key in unexpected
        if key.endswith("backbone.rotary_embeddings.inv_freq")
    ]
    hard_missing = [key for key in missing if key not in allowed_missing]
    hard_unexpected = [key for key in unexpected if key not in allowed_unexpected]
    if hard_missing or hard_unexpected:
        raise RuntimeError(
            f"{label} tower checkpoint did not load cleanly: "
            f"missing={hard_missing[:20]} unexpected={hard_unexpected[:20]}"
        )


def load_phase2_fusion_model(
    checkpoint_path: Path,
    source_configs: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    repo_root: Path,
) -> PepCLIPConcatFusionModel:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Phase-2 final checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    config_1d = dict(source_configs["config_1d"])
    config_3d = dict(source_configs["config_3d"])
    resolve_hf_path(config_1d, args.hf_model_name_or_path_1d, repo_root)

    model_1d = build_1d_model_from_config(config_1d, device)
    model_3d = build_3d_model_from_config(config_3d, device)
    tower_state = state.get("tower_state_dict")
    if not tower_state:
        raise ValueError("Phase-3 requires a learned concat fusion checkpoint with tower_state_dict")
    load_tower_state_compatible(model_1d, tower_state["model_1d"], "1d")
    load_tower_state_compatible(model_3d, tower_state["model_3d"], "3d")
    set_frozen(model_1d)
    set_frozen(model_3d)

    checkpoint_args = state.get("args", {})
    dim_1d = int(config_1d.get("output_dim", 128))
    dim_3d = int(config_3d.get("output_dim", 128))
    model = PepCLIPConcatFusionModel(
        model_1d=model_1d,
        model_3d=model_3d,
        concat_dim=dim_1d + dim_3d,
        hidden_dim=int(args.fusion_hidden_dim or checkpoint_args.get("fusion_hidden_dim", 512)),
        output_dim=int(args.fusion_output_dim or checkpoint_args.get("fusion_output_dim", 512)),
        dropout=float(args.dropout if args.dropout is not None else checkpoint_args.get("dropout", 0.1)),
        temperature=float(args.temperature if args.temperature is not None else checkpoint_args.get("temperature", 1.0 / 14.0)),
    ).to(device)
    fusion_state = state.get("fusion_state_dict")
    if not fusion_state:
        raise ValueError("Phase-2 checkpoint has no fusion_state_dict")
    model.receptor_fusion.load_state_dict(fusion_state["receptor_fusion"], strict=True)
    model.peptide_fusion.load_state_dict(fusion_state["peptide_fusion"], strict=True)
    return model


def run_epoch(
    model: PepCLIPConcatFusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    train: bool,
    use_amp: bool,
    max_grad_norm: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    max_steps: int | None = None,
    skip_batches: int = 0,
    on_step: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], int]:
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if skip_batches < 0:
        raise ValueError("skip_batches must be non-negative")
    model.train(train)
    total_loss = 0.0
    total_loss_r2p = 0.0
    total_loss_p2r = 0.0
    total_items = 0
    start_time = time.perf_counter()
    global_step_increment = 0
    peak_memory_before = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    processed_batches = 0
    last_batch_index = skip_batches
    for batch_index, batch in enumerate(loader):
        if batch_index < skip_batches:
            continue
        if train and max_steps is not None and global_step_increment >= max_steps:
            break
        with torch.set_grad_enabled(train):
            with autocast_context(device, use_amp):
                forward = forward_and_known_positive_loss(model, batch, device)
                loss = forward["loss_total"]
            if train:
                if optimizer is None:
                    raise ValueError("optimizer is required for train=True")
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_grad_norm,
                )
                scale_before = float(scaler.get_scale()) if scaler.is_enabled() else None
                scaler.step(optimizer)
                scaler.update()
                step_succeeded = not scaler.is_enabled() or float(scaler.get_scale()) >= float(scale_before)
                if not step_succeeded:
                    continue
                if scheduler is not None:
                    scheduler.step()
                global_step_increment += 1
                if on_step is not None:
                    on_step(global_step_increment, batch_index + 1)
        batch_size = len(batch["one_d"]["sample_id"])
        total_loss += float(loss.item()) * batch_size
        total_loss_r2p += float(forward["loss_receptor_to_peptide"].item()) * batch_size
        total_loss_p2r += float(forward["loss_peptide_to_receptor"].item()) * batch_size
        total_items += batch_size
        processed_batches += 1
        last_batch_index = batch_index + 1
    metrics: dict[str, Any] = {
        "loss_total": total_loss / max(total_items, 1),
        "loss_receptor_to_peptide": total_loss_r2p / max(total_items, 1),
        "loss_peptide_to_receptor": total_loss_p2r / max(total_items, 1),
        "items": total_items,
        "batches": processed_batches,
        "batches_completed_in_epoch": last_batch_index,
        "elapsed_seconds": time.perf_counter() - start_time,
        "gpu_peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) - peak_memory_before if device.type == "cuda" else 0
        ),
        "amp_scaler": float(scaler.get_scale()) if scaler.is_enabled() else None,
    }
    return metrics, global_step_increment


def _plan_hash(plan: list[dict[str, Any]]) -> str:
    encoded = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _sequence_hash(values: list[str]) -> str:
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _build_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_train_steps: int,
    warmup_fraction: float,
) -> tuple[torch.optim.lr_scheduler.LRScheduler, int]:
    if total_train_steps < 1:
        raise ValueError("total_train_steps must be positive")
    if not 0.0 <= warmup_fraction <= 1.0:
        raise ValueError("warmup_fraction must be in [0, 1]")
    warmup_steps = int(math.ceil(total_train_steps * warmup_fraction))
    if warmup_steps == 0:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0), 0

    def scale(step: int) -> float:
        return min(float(step + 1) / float(warmup_steps), 1.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale), warmup_steps


def _interface_pair_subset(
    base: Phase3RandomConformerDataset, limit: int | None
) -> InterfacePairSubsetDataset:
    interface_pair_ids = sorted(base.interface_pair_ids)
    if limit is not None:
        if limit < 1:
            raise ValueError("interface-pair limit must be positive")
        interface_pair_ids = interface_pair_ids[:limit]
    return InterfacePairSubsetDataset(base, interface_pair_ids)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _step_checkpoint_path(output_dir: Path, global_step: int) -> Path:
    if global_step < 1:
        raise ValueError("global_step must be positive for a step checkpoint")
    return output_dir / f"step_{global_step:03d}.pt"


def _validate_full_heavy_cli_contract(args: argparse.Namespace) -> bool:
    values = {
        "adaptation_manifest": args.full_heavy_adaptation_manifest,
        "plan_descriptor": args.full_heavy_plan_descriptor,
        "cache_manifest": args.full_heavy_cache_manifest,
    }
    requested = any(values.values())
    if requested and not all(values.values()):
        if (
            args.full_heavy_plan_descriptor
            and not args.full_heavy_cache_manifest
        ):
            raise ValueError(
                "full_heavy_cache_not_built:"
                "plan_descriptor_cannot_start_training"
            )
        raise ValueError(
            "full_heavy_runtime_contract_incomplete:"
            "adaptation_manifest_plan_descriptor_cache_manifest_required"
        )
    return requested


def main() -> None:
    args = parse_args()
    full_heavy_requested = _validate_full_heavy_cli_contract(args)
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if args.max_steps is not None and args.stop_after_epoch is not None:
        raise ValueError("max_steps and stop_after_epoch cannot be combined")
    save_steps = set(args.save_steps or [])
    if any(step < 1 for step in save_steps):
        raise ValueError("save_steps must contain positive integers")
    if args.max_steps is not None:
        save_steps.add(args.max_steps)
        if any(step > args.max_steps for step in save_steps):
            raise ValueError("save_steps cannot exceed max_steps")
    if args.stop_after_epoch is not None and not 0 <= args.stop_after_epoch < args.epochs:
        raise ValueError("stop_after_epoch must be in [0, epochs)")
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = resolve_path(args.phase2_checkpoint, repo_root).resolve()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except ImportError:
        pass
    device = torch.device(args.device)
    output_dir = resolve_path(args.output_dir, repo_root)
    allowed_root = DEFAULT_RUN_ROOT.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output != allowed_root and allowed_root not in resolved_output.parents:
        raise ValueError(f"DrugCLIP outputs must stay under {allowed_root}: {resolved_output}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_configs = load_source_configs(checkpoint_path, args.source_model_configs, repo_root)
    model = load_phase2_fusion_model(checkpoint_path, source_configs, device, args, repo_root)
    full_heavy_freeze_contract: dict[str, Any] | None = None
    if full_heavy_requested:
        if args.freeze_all_towers or args.train_interface_pair_limit or args.valid_interface_pair_limit:
            raise ValueError(
                "full-heavy adaptation uses its manifest plans and fixed freeze contract"
            )
        full_heavy_freeze_contract = configure_bounded_full_heavy_trainable(model)
        unfreeze_1d = {"requested_layers": 0, "trainable_parameters": 0}
        unfreeze_3d = {
            "requested_layers": 1,
            "scope": "peptide_encoder_only",
            "trainable_parameters": sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and name.startswith("model_3d.peptide_encoder.")
            ),
        }
    elif args.freeze_all_towers:
        unfreeze_1d = {"requested_layers": 0, "trainable_parameters": 0}
        unfreeze_3d = {"requested_layers": 0, "trainable_parameters": 0}
    else:
        unfreeze_1d = unfreeze_1d_last_layers(model.model_1d, args.unfreeze_1d_last_n_layers)
        unfreeze_3d = unfreeze_3d_last_layers(model.model_3d, args.unfreeze_3d_last_n_layers)

    train_base = Phase3RandomConformerDataset(
        args.train_random_conformer_pairs, args.random_conformer_cache,
        args.biological_pairs_jsonl, args.pair_splits_jsonl, split="train",
        mode="train_random", global_seed=args.seed, data_version=args.data_version,
        dataset_root=args.dataset_root, expected_manifest_sha256=args.expected_manifest_sha256,
    )
    valid_base = Phase3RandomConformerDataset(
        args.valid_random_conformer_pairs, args.random_conformer_cache,
        args.biological_pairs_jsonl, args.pair_splits_jsonl, split="valid", mode="fixed",
        global_seed=args.seed + 17, fixed_conformer_index=0, data_version=args.data_version,
        dataset_root=args.dataset_root, expected_manifest_sha256=args.expected_manifest_sha256,
    )
    if train_base.data_contract != valid_base.data_contract:
        raise ValueError("train_valid_data_contract_mismatch")
    train_base.write_mapping_summary(output_dir / "train_interface_pair_mapping_summary.json")
    valid_base.write_mapping_summary(output_dir / "valid_interface_pair_mapping_summary.json")
    full_heavy_data_contract: dict[str, Any] | None = None
    if full_heavy_requested:
        full_heavy_data_contract = validate_bounded_full_heavy_contract(
            resolve_path(args.full_heavy_adaptation_manifest, repo_root),
            plan_descriptor_file=resolve_path(
                args.full_heavy_plan_descriptor, repo_root
            ),
            cache_manifest_file=resolve_path(
                args.full_heavy_cache_manifest, repo_root
            ),
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
            freeze_contract=full_heavy_freeze_contract,
        )
        payloads = full_heavy_data_contract.pop("payload_by_sequence")
        train_view = FullHeavyDatasetView(train_base, payloads)
        valid_view = FullHeavyDatasetView(valid_base, payloads)
        train_dataset = InterfacePairSubsetDataset(
            train_view, full_heavy_data_contract["train_interface_pair_ids"]
        )
        valid_dataset = InterfacePairSubsetDataset(
            valid_view, full_heavy_data_contract["valid_interface_pair_ids"]
        )
    else:
        train_dataset = _interface_pair_subset(train_base, args.train_interface_pair_limit)
        valid_dataset = _interface_pair_subset(valid_base, args.valid_interface_pair_limit)
    train_pair_sha256 = _sequence_hash(train_dataset.interface_pair_ids)
    valid_pair_sha256 = _sequence_hash(valid_dataset.interface_pair_ids)
    subset_manifest = {
        "sampling_unit": "interface_pair",
        "train_interface_pair_ids": train_dataset.interface_pair_ids,
        "train_interface_pair_ids_sha256": train_pair_sha256,
        "valid_interface_pair_ids": valid_dataset.interface_pair_ids,
        "valid_interface_pair_ids_sha256": valid_pair_sha256,
        "train_mode": train_dataset.mode,
        "valid_mode": valid_dataset.mode,
    }
    _write_json(output_dir / "interface_pair_subsets.json", subset_manifest)

    train_sampler = UniquePeptideBatchSampler(train_dataset, args.batch_size, seed=args.seed, epoch=0)
    valid_sampler = UniquePeptideBatchSampler(valid_dataset, args.batch_size, seed=args.seed + 17, epoch=0)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=args.num_workers, collate_fn=collate_phase3)
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=args.num_workers, collate_fn=collate_phase3)
    if full_heavy_requested:
        parameter_groups = build_bounded_optimizer_groups(
            model, fusion_lr=args.lr, tower_lr=args.tower_lr
        )
    else:
        fusion_parameters = [p for module in (model.receptor_fusion, model.peptide_fusion) for p in module.parameters() if p.requires_grad]
        tower_parameters = [p for module in (model.model_1d, model.model_3d) for p in module.parameters() if p.requires_grad]
        parameter_groups = [{"params": fusion_parameters, "lr": args.lr, "group_name": "fusion"}]
        if tower_parameters:
            parameter_groups.append({"params": tower_parameters, "lr": args.tower_lr, "group_name": "tower"})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    epoch_train_steps = len(train_sampler) * args.epochs
    if args.max_steps is not None and args.max_steps > epoch_train_steps:
        raise ValueError("max_steps exceeds the available optimizer steps across all epochs")
    if any(step > epoch_train_steps for step in save_steps):
        raise ValueError("save_steps exceeds the available optimizer steps across all epochs")
    total_train_steps = min(epoch_train_steps, args.max_steps) if args.max_steps is not None else epoch_train_steps
    scheduler, warmup_steps = _build_warmup_scheduler(
        optimizer, total_train_steps, args.warmup_fraction
    )
    use_amp = amp_is_enabled(device, args.amp)
    scaler = make_grad_scaler(device, use_amp)
    freeze_configuration = {
        "freeze_all_towers": args.freeze_all_towers,
        "unfreeze_1d": unfreeze_1d,
        "unfreeze_3d": unfreeze_3d,
        "bounded_full_heavy": full_heavy_freeze_contract,
    }
    valid_sampler.set_epoch(0)
    validation_plan = valid_dataset.epoch_plan()
    validation_plan_path = output_dir / "validation_sampling_plan.jsonl"
    write_jsonl(validation_plan_path, validation_plan)
    validation_plan_sha256 = _plan_hash(validation_plan)
    run_config = {
        "command": shlex.join([sys.executable, "-m", "phase3.drugclip.train", *sys.argv[1:]]),
        "phase2_checkpoint": str(checkpoint_path),
        "data_version": train_base.data_contract.data_version,
        "dataset_version": train_base.data_contract.dataset_version,
        "dataset_root": str(train_base.data_contract.dataset_root) if train_base.data_contract.dataset_root else None,
        "data_manifest_path": str(train_base.data_contract.manifest_path) if train_base.data_contract.manifest_path else None,
        "data_manifest_sha256": train_base.data_contract.manifest_sha256,
        "database_contract": train_base.data_contract.database_contract,
        "cache_schema": train_base.data_contract.cache_schema,
        "generator_id": train_base.data_contract.generator_id,
        "qc_id": train_base.data_contract.qc_id,
        "random_pairs_sha256": sha256_file(train_base.random_pairs_jsonl),
        "random_conformer_cache_sha256": sha256_file(train_base.random_conformer_cache_jsonl),
        "pair_splits_sha256": sha256_file(train_base.pair_splits_jsonl),
        "random_pairs_jsonl": str(Path(args.train_random_conformer_pairs).resolve()),
        "valid_random_pairs_jsonl": str(Path(args.valid_random_conformer_pairs).resolve()),
        "random_conformer_cache_jsonl": str(Path(args.random_conformer_cache).resolve()),
        "biological_pairs_jsonl": str(train_base.biological_pairs_jsonl),
        "biological_pairs_sha256": train_base.mapping_summary()["biological_pairs_sha256"],
        "pair_splits_jsonl": str(Path(args.pair_splits_jsonl).resolve()),
        "relation_schema": train_base.data_contract.relation_schema,
        "sampling_unit": "interface_pair",
        "train_interface_pair_ids_sha256": train_pair_sha256,
        "valid_interface_pair_ids_sha256": valid_pair_sha256,
        "fixed_validation_plan_sha256": validation_plan_sha256,
        "total_train_steps": total_train_steps,
        "max_steps": args.max_steps,
        "save_steps": sorted(save_steps),
        "warmup_fraction": args.warmup_fraction,
        "warmup_steps": warmup_steps,
        "scheduler_kind": "linear_warmup_constant",
        "global_seed": args.seed,
        "freeze_configuration": freeze_configuration,
        "full_heavy_data_contract": full_heavy_data_contract,
        "train_interface_pair_ids": train_dataset.interface_pair_ids,
        "valid_interface_pair_ids": valid_dataset.interface_pair_ids,
        "amp_enabled": use_amp,
        "args": vars(args),
    }
    _write_json(output_dir / "config.json", run_config)

    history: list[dict[str, Any]] = []
    global_step = 0
    best_validation_loss = float("inf")
    best_validation_epoch: int | None = None
    start_epoch = 0
    resume_skip_batches = 0
    if args.resume_checkpoint:
        restored = load_training_checkpoint(
            resolve_path(args.resume_checkpoint, repo_root), model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, expected_run_config=run_config, device=device,
        )
        restored_epoch = int(restored["epoch"])
        restored_sampler_state = restored.get("sampler_state", {})
        completed_epoch = restored_sampler_state.get("completed_epoch")
        if completed_epoch == restored_epoch:
            start_epoch = restored_epoch + 1
        else:
            start_epoch = max(restored_epoch, 0)
            resume_skip_batches = int(restored_sampler_state.get("batches_completed_in_epoch", 0))
        global_step = int(restored["global_step"])
        best_validation_loss = float(restored["best_validation_loss"])
        best_validation_epoch = restored.get("sampler_state", {}).get("best_validation_epoch")
        history = list(restored.get("history", []))
        train_sampler.set_epoch(start_epoch)
        expected_hash = restored.get("sampler_state", {}).get("next_train_plan_hash")
        if expected_hash and _plan_hash(train_dataset.epoch_plan()) != expected_hash:
            raise RuntimeError("resume regenerated a different next-epoch sampling plan")
    else:
        valid_metrics, _ = run_epoch(
            model, valid_loader, None, scaler, device, False, use_amp, args.max_grad_norm
        )
        best_validation_loss = float(valid_metrics["loss_total"])
        _write_json(output_dir / "initial_validation_metrics.json", {
            "epoch": None,
            "global_step": 0,
            "metrics": valid_metrics,
            "fixed_validation_plan_sha256": validation_plan_sha256,
        })
        train_sampler.set_epoch(0)
        baseline_sampler_state = {
            "completed_epoch": None,
            "train_plan_hash": None,
            "next_epoch": 0,
            "next_train_plan_hash": _plan_hash(train_dataset.epoch_plan()),
            "validation_epoch": 0,
            "validation_plan_hash": validation_plan_sha256,
            "best_validation_epoch": None,
            "summary": train_sampler.summary(),
        }
        save_training_checkpoint(
            output_dir / "checkpoint_best.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=-1,
            global_step=0,
            global_seed=args.seed,
            best_validation_loss=best_validation_loss,
            run_config=run_config,
            sampler_state=baseline_sampler_state,
            history=history,
        )

    previous_plan: dict[str, dict[str, Any]] | None = None
    if start_epoch > 0:
        # Preserve change counters across a resumed process boundary.
        train_sampler.set_epoch(start_epoch - 1)
        previous_plan = {
            str(item["interface_pair_id"]): item for item in train_dataset.epoch_plan()
        }
    log_path = output_dir / "train_log.jsonl"
    for epoch in range(start_epoch, args.epochs):
        if args.max_steps is not None and global_step >= args.max_steps:
            break
        train_sampler.set_epoch(epoch)
        plan = train_dataset.epoch_plan()
        plan_hash = _plan_hash(plan)
        write_jsonl(output_dir / "sampling_plans" / f"epoch_{epoch:03d}.jsonl", plan)
        current_plan = {str(item["interface_pair_id"]): item for item in plan}
        conformer_changes = 0
        if previous_plan is not None:
            for interface_pair_id, item in current_plan.items():
                previous = previous_plan[interface_pair_id]
                conformer_changes += int(item["conformer_index"] != previous["conformer_index"])
        learning_rates_start = [float(group["lr"]) for group in optimizer.param_groups]
        remaining_steps = None if args.max_steps is None else args.max_steps - global_step

        def save_requested_step(step_increment: int, batches_completed: int) -> None:
            checkpoint_step = global_step + step_increment
            if checkpoint_step not in save_steps:
                return
            step_sampler_state = {
                "completed_epoch": None,
                "current_epoch": epoch,
                "batches_completed_in_epoch": batches_completed,
                "train_plan_hash": plan_hash,
                "next_epoch": epoch,
                "next_train_plan_hash": plan_hash,
                "validation_epoch": 0,
                "validation_plan_hash": validation_plan_sha256,
                "best_validation_epoch": best_validation_epoch,
                "summary": train_sampler.summary(),
            }
            save_training_checkpoint(
                _step_checkpoint_path(output_dir, checkpoint_step),
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch, global_step=checkpoint_step, global_seed=args.seed,
                best_validation_loss=best_validation_loss, run_config=run_config,
                sampler_state=step_sampler_state, history=history,
            )

        train_metrics, step_increment = run_epoch(
            model, train_loader, optimizer, scaler, device, True, use_amp, args.max_grad_norm,
            scheduler=scheduler,
            max_steps=remaining_steps,
            skip_batches=resume_skip_batches if epoch == start_epoch else 0,
            on_step=save_requested_step,
        )
        global_step += step_increment
        valid_sampler.set_epoch(0)
        valid_metrics, _ = run_epoch(
            model, valid_loader, None, scaler, device, False, use_amp, args.max_grad_norm
        )
        learning_rates_end = [float(group["lr"]) for group in optimizer.param_groups]
        sampler_summary = train_sampler.summary()
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rates_start": learning_rates_start,
            "learning_rates_end": learning_rates_end,
            "train": train_metrics,
            "valid": valid_metrics,
            "sampling": {
                **sampler_summary,
                "plan_sha256": plan_hash,
                "conformer_changed_interface_pairs": conformer_changes,
            },
        }
        history.append(row)
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        batches_completed_in_epoch = int(train_metrics["batches_completed_in_epoch"])
        epoch_completed = batches_completed_in_epoch >= len(train_loader)
        next_epoch = epoch + 1 if epoch_completed else epoch
        train_sampler.set_epoch(next_epoch)
        sampler_state = {
            "completed_epoch": epoch if epoch_completed else None,
            "current_epoch": next_epoch,
            "train_plan_hash": plan_hash,
            "next_epoch": next_epoch,
            "next_train_plan_hash": _plan_hash(train_dataset.epoch_plan()),
            "batches_completed_in_epoch": 0 if epoch_completed else batches_completed_in_epoch,
            "validation_epoch": 0,
            "validation_plan_hash": _plan_hash(valid_dataset.epoch_plan()),
            "summary": sampler_summary,
        }
        improved = float(valid_metrics["loss_total"]) < best_validation_loss
        if improved:
            best_validation_loss = float(valid_metrics["loss_total"])
            best_validation_epoch = epoch
        common = dict(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch,
            global_step=global_step, global_seed=args.seed, best_validation_loss=best_validation_loss,
            run_config=run_config, sampler_state=sampler_state, history=history,
        )
        sampler_state["best_validation_epoch"] = best_validation_epoch
        common["sampler_state"] = sampler_state
        save_training_checkpoint(output_dir / "checkpoint_last.pt", **common)
        if improved:
            save_training_checkpoint(output_dir / "checkpoint_best.pt", **common)
        _write_json(output_dir / "summary.json", {
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_validation_epoch,
            "best_is_initial_validation": best_validation_epoch is None,
            "latest": row,
            "completed_epochs": len(history),
            "latest_epoch_completed": epoch_completed,
        })
        print(json.dumps(row, ensure_ascii=False), flush=True)
        previous_plan = current_plan
        resume_skip_batches = 0
        if args.max_steps is not None and global_step >= args.max_steps:
            break
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            break
    if args.max_steps is not None and global_step != args.max_steps:
        raise RuntimeError(
            f"bounded training ended after {global_step} successful optimizer steps; expected {args.max_steps}"
        )


if __name__ == "__main__":
    main()

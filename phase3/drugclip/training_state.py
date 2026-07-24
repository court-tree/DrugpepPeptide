"""AMP, RNG, and strict checkpoint helpers for Phase-3 training."""

from __future__ import annotations

import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_SCHEMA = "pepclip-phase3-drugclip-training-v1"


def amp_is_enabled(device: torch.device, requested: bool) -> bool:
    return bool(requested and device.type == "cuda" and torch.cuda.is_available())


def autocast_context(device: torch.device, enabled: bool):
    if enabled:
        # EGNN uses pairwise squared distances.  BF16 retains AMP acceleration
        # while avoiding FP16 distance overflow on large receptor coordinates.
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def make_grad_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device="cuda", enabled=amp_is_enabled(device, enabled))


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("numpy") is not None:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError as error:
            raise RuntimeError("checkpoint contains NumPy RNG state but NumPy is unavailable") from error
    if state.get("torch_cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    global_seed: int,
    best_validation_loss: float,
    run_config: dict[str, Any],
    sampler_state: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data_contract_keys = (
        "data_version", "dataset_version", "dataset_root", "data_manifest_path",
        "data_manifest_sha256", "database_contract", "cache_schema", "generator_id", "qc_id",
        "full_heavy_data_contract",
    )
    data_contract = {key: run_config.get(key) for key in data_contract_keys}
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "global_seed": int(global_seed),
            "best_validation_loss": float(best_validation_loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "sampler_state": sampler_state,
            "run_config": run_config,
            "data_contract": data_contract,
            "history": history,
        },
        output,
    )


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    expected_run_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    # RNG states must remain CPU ByteTensors; model/optimizer loading then moves
    # parameter state onto the already-created target-device parameters.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported Phase-3 training checkpoint schema")
    stored_config = checkpoint.get("run_config")
    if not isinstance(stored_config, dict):
        raise ValueError("checkpoint lacks run_config")
    version_contract_keys = (
        "data_version",
        "dataset_version",
        "dataset_root",
        "data_manifest_path",
        "data_manifest_sha256",
        "database_contract",
        "cache_schema",
        "generator_id",
        "qc_id",
        "random_pairs_sha256",
        "random_conformer_cache_sha256",
        "pair_splits_sha256",
    )
    stored_data_contract = checkpoint.get("data_contract")
    if stored_data_contract is not None:
        for key, stored_value in stored_data_contract.items():
            if stored_value != expected_run_config.get(key):
                raise ValueError(f"checkpoint data_contract mismatch for {key}")
    required_keys = (
        "phase2_checkpoint",
        "relation_schema",
        "random_pairs_jsonl",
        "valid_random_pairs_jsonl",
        "random_conformer_cache_jsonl",
        "biological_pairs_jsonl",
        "biological_pairs_sha256",
        "pair_splits_jsonl",
        "freeze_configuration",
        "global_seed",
        "sampling_unit",
        "train_interface_pair_ids",
        "valid_interface_pair_ids",
        "train_interface_pair_ids_sha256",
        "valid_interface_pair_ids_sha256",
        "fixed_validation_plan_sha256",
        "total_train_steps",
        "warmup_fraction",
        "warmup_steps",
        "scheduler_kind",
    )
    for key in (*version_contract_keys, *required_keys):
        if key in version_contract_keys and key not in stored_config:
            if expected_run_config.get("data_version") in (None, "v2"):
                continue
            raise ValueError(f"checkpoint run_config mismatch for {key}")
        if stored_config.get(key) != expected_run_config.get(key):
            raise ValueError(f"checkpoint run_config mismatch for {key}")
    if (
        stored_config.get("full_heavy_data_contract")
        != expected_run_config.get("full_heavy_data_contract")
    ):
        raise ValueError(
            "checkpoint run_config mismatch for full_heavy_data_contract"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint

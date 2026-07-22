"""Read-only v3 opt-in smoke and v2/v3 conformer-difference audit."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from phase2.pepclip.train_concat_fusion import unfreeze_1d_last_layers, unfreeze_3d_last_layers
from phase3.drugclip.batching import UniquePeptideBatchSampler, collate_phase3
from phase3.drugclip.forward import forward_and_known_positive_loss
from phase3.drugclip.io_utils import read_jsonl
from phase3.drugclip.random_augmentation_dataset import (
    Phase3RandomConformerDataset,
    materialize_random_conformer,
)
from phase3.drugclip.random_conformer_v3 import clash15_details, coordinate_sha256
from phase3.drugclip.train import DEFAULT_PHASE2_FINAL_CHECKPOINT, load_phase2_fusion_model, load_source_configs, resolve_path


FORMAL_KNOWN_POSITIVE_PAIR_IDS = [
    "interface_pair:0116a09ddeb4d49524a1",
    "interface_pair:05b9a7663ec39d485b26",
]


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _tensor_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.detach().float().cpu() - b.detach().float().cpu()).abs()
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean())}


def _cache_conformer(dataset: Phase3RandomConformerDataset, pair_id: str, conformer_index: int) -> dict[str, Any]:
    row = dataset.interface_pair_rows[pair_id]
    conformer = next(
        item for item in row["_cache"]["conformers"] if int(item["conformer_index"]) == int(conformer_index)
    )
    return materialize_random_conformer(row, conformer, str(row["_biological_pair_id"]))


def _select_examples(v3_root: Path, split: str, limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    replacements = [
        row for row in read_jsonl(v3_root / "conformer_replacement_audit.jsonl")
        if str(row.get("split")) == split
    ]
    cache_to_pairs: dict[str, list[str]] = {}
    for row in read_jsonl(v3_root / "04_training_input/random_conformer_pairs.jsonl"):
        if row.get("split") == split:
            cache_to_pairs.setdefault(str(row["random_conformer_cache_id"]), []).append(str(row["pair"]["pair_id"]))

    replaced: list[dict[str, Any]] = []
    seen_pair_peptide: set[tuple[str, str]] = set()
    for row in replacements:
        candidates = cache_to_pairs.get(str(row["cache_id"]), [])
        if not candidates:
            continue
        pair_id = candidates[0]
        key = (pair_id, str(row["peptide_sequence"]))
        if key in seen_pair_peptide:
            continue
        seen_pair_peptide.add(key)
        replaced.append({**row, "interface_pair_id": pair_id})
        if len(replaced) >= limit:
            break

    replaced_keys = {(str(row["cache_id"]), int(row["conformer_index"])) for row in replacements}
    unchanged: list[dict[str, Any]] = []
    for row in read_jsonl(v3_root / "03_random_conformer_cache/random_conformer_cache.jsonl"):
        if row.get("split") != split:
            continue
        pair_ids = cache_to_pairs.get(str(row["cache_id"]), [])
        if not pair_ids:
            continue
        for conformer in row["conformers"]:
            key = (str(row["cache_id"]), int(conformer["conformer_index"]))
            if key in replaced_keys:
                continue
            unchanged.append(
                {
                    "cache_id": str(row["cache_id"]),
                    "interface_pair_id": pair_ids[0],
                    "peptide_sequence": str(row["peptide_sequence"]),
                    "conformer_index": int(conformer["conformer_index"]),
                }
            )
            break
        if len(unchanged) >= limit:
            break
    if len(replaced) < limit or len(unchanged) < limit:
        raise RuntimeError("unable_to_select_required_v3_comparison_examples")
    return unchanged, replaced


def _audit_sampling(dataset: Phase3RandomConformerDataset, batch_size: int, seed: int) -> dict[str, Any]:
    sampler0 = UniquePeptideBatchSampler(dataset, batch_size=batch_size, seed=seed, epoch=0)
    plan0 = dataset.epoch_plan()
    batches0 = list(sampler0)
    peptide_by_index0 = {
        index: str(row["peptide_sequence"]) for index, row in enumerate(plan0)
    }
    sampler0_repeat = UniquePeptideBatchSampler(dataset, batch_size=batch_size, seed=seed, epoch=0)
    plan0_repeat = dataset.epoch_plan()
    sampler1 = UniquePeptideBatchSampler(dataset, batch_size=batch_size, seed=seed, epoch=1)
    plan1 = dataset.epoch_plan()
    flattened0 = [index for batch in batches0 for index in batch]
    peptide_violations = 0
    for batch in batches0:
        peptides = [peptide_by_index0[index] for index in batch]
        peptide_violations += int(len(peptides) != len(set(peptides)))
    plan1_by_pair = {row["interface_pair_id"]: row for row in plan1}
    conformer_changes = sum(
        int(row["conformer_index"] != plan1_by_pair[row["interface_pair_id"]]["conformer_index"])
        for row in plan0
    )
    interface_changes = sum(
        int(row["receptor_interface_id"] != plan1_by_pair[row["interface_pair_id"]]["receptor_interface_id"])
        for row in plan0
    )
    return {
        "train_pairs": len(dataset),
        "pairs_visited": len(flattened0),
        "pair_loss": len(dataset) - len(set(flattened0)),
        "pair_duplicates": len(flattened0) - len(set(flattened0)),
        "batch_peptide_duplicates": peptide_violations,
        "conformer_index_histogram": dict(sorted(Counter(int(row["conformer_index"]) for row in plan0).items())),
        "same_seed_epoch_reproducible": plan0 == plan0_repeat,
        "different_epoch_changed_conformers": conformer_changes,
        "interface_changes_across_epoch": interface_changes,
        "batches": sampler0.summary()["batches"],
    }


def _run_forward(model: torch.nn.Module, batch: dict[str, Any], device: torch.device, backward: bool, amp: bool) -> dict[str, Any]:
    model = model.to(device)
    model.train(backward)
    if backward:
        unfreeze_1d_last_layers(model.model_1d, 1)
        unfreeze_3d_last_layers(model.model_3d, 1)
        model.zero_grad(set_to_none=True)
    context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if amp else torch.no_grad()
    if backward and not amp:
        context = torch.enable_grad()
    with context:
        result = forward_and_known_positive_loss(model, batch, device)
        loss = result["loss_total"]
    if backward:
        loss.backward()
    gradients = {}
    if backward:
        for name, module in (
            ("receptor_fusion", model.receptor_fusion),
            ("peptide_fusion", model.peptide_fusion),
            ("model_1d", model.model_1d),
            ("model_3d", model.model_3d),
        ):
            grads = [
                parameter.grad.detach().float().norm()
                for parameter in module.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            gradients[name] = float(torch.stack(grads).norm().cpu()) if grads else 0.0
    tensors = (
        result["receptor_embedding"],
        result["peptide_embedding"],
        result["similarity_matrix"],
        result["loss_total"].reshape(1),
    )
    return {
        "embedding_shape": [list(result["receptor_embedding"].shape), list(result["peptide_embedding"].shape)],
        "similarity_shape": list(result["similarity_matrix"].shape),
        "loss_total": float(result["loss_total"].detach().float().cpu()),
        "finite": all(bool(torch.isfinite(t.detach().float()).all()) for t in tensors),
        "mask_r2p_true": int(result["receptor_to_peptide_mask"].sum().detach().cpu()),
        "mask_p2r_true": int(result["peptide_to_receptor_mask"].sum().detach().cpu()),
        "mask_r2p_indices": result["receptor_to_peptide_mask"].nonzero().detach().cpu().tolist(),
        "mask_p2r_indices": result["peptide_to_receptor_mask"].nonzero().detach().cpu().tolist(),
        "gradient_norms": gradients,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "_raw": result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", default="phase3/runs/drugclip/random_conformer_v2")
    parser.add_argument("--v3-root", default="phase3/runs/drugclip/random_conformer_v3")
    parser.add_argument("--phase2-checkpoint", default=DEFAULT_PHASE2_FINAL_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    v2_root = resolve_path(args.v2_root, repo).resolve()
    v3_root = resolve_path(args.v3_root, repo).resolve()
    checkpoint = resolve_path(args.phase2_checkpoint, repo).resolve()
    biological_v2 = repo / "phase3/runs/receptor_identity_mapping_v1/biological_pairs.jsonl"
    v2 = Phase3RandomConformerDataset(
        v2_root / "04_training_input/random_conformer_pairs.jsonl",
        v2_root / "03_random_conformer_cache/random_conformer_cache.jsonl",
        biological_v2,
        v2_root / "02_leakage_safe_split/pair_splits.jsonl",
        split="train",
        mode="train_random",
        global_seed=args.seed,
        data_version="v2",
    )
    v3 = Phase3RandomConformerDataset(
        v3_root / "04_training_input/random_conformer_pairs.jsonl",
        v3_root / "03_random_conformer_cache/random_conformer_cache.jsonl",
        v3_root / "dependencies/biological_pairs.jsonl",
        v3_root / "02_leakage_safe_split/pair_splits.jsonl",
        split="train",
        mode="train_random",
        global_seed=args.seed,
        data_version="v3",
        dataset_root=v3_root,
    )
    if len(v2) != len(v3):
        raise RuntimeError("v2_v3_train_pair_count_mismatch")
    sampling = _audit_sampling(v3, args.batch_size, args.seed)

    unchanged, replaced = _select_examples(v3_root, "train", 10)
    source_configs = load_source_configs(checkpoint, None, repo)
    model_args = SimpleNamespace(
        hf_model_name_or_path_1d=None,
        fusion_hidden_dim=None,
        fusion_output_dim=None,
        dropout=None,
        temperature=None,
    )
    model = load_phase2_fusion_model(checkpoint, source_configs, torch.device("cpu"), model_args, repo)
    model.eval()

    def build_batch(rows: list[dict[str, Any]], dataset: Phase3RandomConformerDataset) -> dict[str, Any]:
        items = [
            _cache_conformer(dataset, str(row["interface_pair_id"]), int(row["conformer_index"]))
            for row in rows
        ]
        return collate_phase3(items)

    unchanged_v2_batch = build_batch(unchanged, v2)
    unchanged_v3_batch = build_batch(unchanged, v3)
    replaced_v2_batch = build_batch(replaced, v2)
    replaced_v3_batch = build_batch(replaced, v3)

    with torch.no_grad():
        unchanged_v2_forward = forward_and_known_positive_loss(model, unchanged_v2_batch, torch.device("cpu"))
        unchanged_v3_forward = forward_and_known_positive_loss(model, unchanged_v3_batch, torch.device("cpu"))
        replaced_v3_forward = forward_and_known_positive_loss(model, replaced_v3_batch, torch.device("cpu"))

    unchanged_coord_diffs: list[float] = []
    unchanged_rows = []
    for row in unchanged:
        v2_item = _cache_conformer(v2, row["interface_pair_id"], row["conformer_index"])
        v3_item = _cache_conformer(v3, row["interface_pair_id"], row["conformer_index"])
        h2 = coordinate_sha256(v2_item["peptide_atoms"])
        h3 = coordinate_sha256(v3_item["peptide_atoms"])
        unchanged_rows.append({**row, "v2_coordinate_sha256": h2, "v3_coordinate_sha256": h3, "same": h2 == h3})
        for a, b in zip(v2_item["peptide_atoms"], v3_item["peptide_atoms"]):
            unchanged_coord_diffs.append(max(abs(float(a[k]) - float(b[k])) for k in ("x", "y", "z")))

    replaced_coord_diffs: list[float] = []
    replaced_rows = []
    for row in replaced:
        v2_item = _cache_conformer(v2, row["interface_pair_id"], row["conformer_index"])
        v3_item = _cache_conformer(v3, row["interface_pair_id"], row["conformer_index"])
        h2 = coordinate_sha256(v2_item["peptide_atoms"])
        h3 = coordinate_sha256(v3_item["peptide_atoms"])
        qc = clash15_details(v3_item["peptide_atoms"])
        replaced_rows.append(
            {
                "interface_pair_id": row["interface_pair_id"],
                "cache_id": row["cache_id"],
                "peptide_sequence": row["peptide_sequence"],
                "conformer_index": row["conformer_index"],
                "v2_coordinate_sha256": h2,
                "v3_coordinate_sha256": h3,
                "v3_clash15_pass": not qc["has_clash"],
                "v3_minimum_nonlocal_backbone_distance_angstrom": qc["minimum_nonlocal_backbone_distance_angstrom"],
            }
        )
        for a, b in zip(v2_item["peptide_atoms"], v3_item["peptide_atoms"]):
            replaced_coord_diffs.append(max(abs(float(a[k]) - float(b[k])) for k in ("x", "y", "z")))

    smoke_batch = collate_phase3(
        [
            _cache_conformer(v3, replaced[0]["interface_pair_id"], replaced[0]["conformer_index"]),
            _cache_conformer(v3, unchanged[0]["interface_pair_id"], unchanged[0]["conformer_index"]),
        ]
    )
    known_positive_batch = None
    if all(pair_id in v3.interface_pair_rows for pair_id in FORMAL_KNOWN_POSITIVE_PAIR_IDS):
        known_positive_batch = collate_phase3(
            [_cache_conformer(v3, pair_id, 0) for pair_id in FORMAL_KNOWN_POSITIVE_PAIR_IDS]
        )
    cpu = _run_forward(model, smoke_batch, torch.device("cpu"), backward=False, amp=False)
    known_positive_cpu = (
        None
        if known_positive_batch is None
        else _run_forward(model, known_positive_batch, torch.device("cpu"), backward=False, amp=False)
    )
    cuda_eval = cuda_backward = cuda_bf16_backward = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        cuda_eval = _run_forward(model, smoke_batch, torch.device("cuda"), backward=False, amp=False)
        torch.cuda.reset_peak_memory_stats()
        cuda_bf16_backward = _run_forward(model, smoke_batch, torch.device("cuda"), backward=True, amp=True)
        torch.cuda.reset_peak_memory_stats()
        cuda_backward = _run_forward(model, smoke_batch, torch.device("cuda"), backward=True, amp=False)

    result = {
        "data_contract": {
            "v2": v2.mapping_summary(),
            "v3": v3.mapping_summary(),
        },
        "sampling": sampling,
        "targeted_v3_replaced_pair": replaced_rows[0],
        "targeted_v3_unchanged_pair": unchanged_rows[0],
        "unchanged_examples": unchanged_rows,
        "replaced_examples": replaced_rows,
        "diff_summary": {
            "unchanged_coordinate_max_abs": _stats(unchanged_coord_diffs),
            "replaced_coordinate_max_abs": _stats(replaced_coord_diffs),
            "unchanged_receptor_embedding": _tensor_diff(
                unchanged_v2_forward["receptor_embedding"], unchanged_v3_forward["receptor_embedding"]
            ),
            "unchanged_peptide_embedding": _tensor_diff(
                unchanged_v2_forward["peptide_embedding"], unchanged_v3_forward["peptide_embedding"]
            ),
            "unchanged_similarity": _tensor_diff(
                unchanged_v2_forward["similarity_matrix"], unchanged_v3_forward["similarity_matrix"]
            ),
            "replaced_v3_forward_finite": all(
                bool(torch.isfinite(t.detach().float()).all())
                for t in (
                    replaced_v3_forward["receptor_embedding"],
                    replaced_v3_forward["peptide_embedding"],
                    replaced_v3_forward["similarity_matrix"],
                    replaced_v3_forward["loss_total"].reshape(1),
                )
            ),
        },
        "model_smoke": {
            "cpu": {key: value for key, value in cpu.items() if key != "_raw"},
            "known_positive_cpu": (
                None if known_positive_cpu is None else {key: value for key, value in known_positive_cpu.items() if key != "_raw"}
            ),
            "cuda_eval": None if cuda_eval is None else {key: value for key, value in cuda_eval.items() if key != "_raw"},
            "cuda_bf16_backward": None if cuda_bf16_backward is None else {key: value for key, value in cuda_bf16_backward.items() if key != "_raw"},
            "cuda_backward": None if cuda_backward is None else {key: value for key, value in cuda_backward.items() if key != "_raw"},
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output_json:
        output = resolve_path(args.output_json, repo)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

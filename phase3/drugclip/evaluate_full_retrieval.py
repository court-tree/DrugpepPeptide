"""Fixed-plan full-candidate retrieval evaluation for the Phase-3 Pilot.

This evaluator deliberately does not use an in-batch loss or candidate queue.
It encodes the frozen 512-pair validation plan once per checkpoint, then ranks
each query against the same exact-identity-deduplicated candidate bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from phase3.drugclip.batching import UniquePeptideBatchSampler, collate_phase3
from phase3.drugclip.forward import forward_phase2_fusion_batch
from phase3.drugclip.io_utils import read_jsonl, write_json, write_jsonl
from phase3.drugclip.random_augmentation_dataset import (
    InterfacePairSubsetDataset,
    Phase3RandomConformerDataset,
)
from phase3.drugclip.train import (
    _interface_pair_subset,
    _plan_hash,
    load_phase2_fusion_model,
    load_source_configs,
    resolve_path,
)


DEFAULT_PILOT_OUTPUT = Path("phase3/runs/drugclip/pilot_interface_pair_4096_512_v1")


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _sequence_hash(values: list[str]) -> str:
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-candidate evaluation of the Phase-3 4096/512 Pilot.")
    parser.add_argument("--pilot_output", default=str(DEFAULT_PILOT_OUTPUT))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _validate_pilot_contract(pilot: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = _read_json(pilot / "config.json")
    subset = _read_json(pilot / "interface_pair_subsets.json")
    plan = list(read_jsonl(pilot / "validation_sampling_plan.jsonl"))
    valid_ids = [str(value) for value in subset["valid_interface_pair_ids"]]
    if len(valid_ids) != 512 or len(set(valid_ids)) != 512:
        raise ValueError("pilot validation subset must contain exactly 512 unique interface_pair_id values")
    if _sequence_hash(valid_ids) != config.get("valid_interface_pair_ids_sha256"):
        raise ValueError("validation interface-pair list SHA256 mismatch")
    if _plan_hash(plan) != config.get("fixed_validation_plan_sha256"):
        raise ValueError("fixed validation plan SHA256 mismatch")
    if len(plan) != 512 or len({str(row["interface_pair_id"]) for row in plan}) != 512:
        raise ValueError("fixed validation plan must contain exactly 512 unique interface pairs")
    if [str(row["interface_pair_id"]) for row in plan] != valid_ids:
        raise ValueError("fixed validation plan order differs from the saved validation subset")
    if any(int(row["conformer_index"]) != 0 for row in plan):
        raise ValueError("this Pilot contract requires fixed conformer index 0")
    return config, subset, plan


def _load_fixed_dataset(
    config: dict[str, Any], subset_manifest: dict[str, Any], expected_plan: list[dict[str, Any]]
) -> InterfacePairSubsetDataset:
    args = config["args"]
    base = Phase3RandomConformerDataset(
        args["valid_random_conformer_pairs"],
        args["random_conformer_cache"],
        args["biological_pairs_jsonl"],
        args["pair_splits_jsonl"],
        split="valid",
        mode="fixed",
        global_seed=int(config["global_seed"]) + 17,
        fixed_conformer_index=0,
        data_version=str(config.get("data_version") or "v2"),
        dataset_root=config.get("dataset_root"),
        expected_manifest_sha256=config.get("data_manifest_sha256"),
    )
    dataset = InterfacePairSubsetDataset(base, [str(value) for value in subset_manifest["valid_interface_pair_ids"]])
    dataset.set_epoch(0)
    actual_plan = dataset.epoch_plan()
    if actual_plan != expected_plan:
        raise ValueError("materialized fixed dataset plan differs from validation_sampling_plan.jsonl")
    return dataset


def _load_checkpoint_model(
    label: str,
    checkpoint_path: Path,
    phase2_checkpoint: Path,
    source_configs: dict[str, Any],
    train_args: argparse.Namespace,
    repo_root: Path,
    device: torch.device,
) -> torch.nn.Module:
    model = load_phase2_fusion_model(
        phase2_checkpoint, source_configs, device, train_args, repo_root
    )
    if label != "phase2_baseline":
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    return model


def _checkpoint_specs(
    checkpoint: str | None,
    model_label: str | None,
    pilot: Path,
    phase2_checkpoint: Path,
    repo_root: Path,
) -> list[tuple[str, Path]]:
    if bool(checkpoint) != bool(model_label):
        raise ValueError("--checkpoint and --model-label must be provided together")
    if checkpoint is None:
        return [
            ("phase2_baseline", phase2_checkpoint),
            ("epoch0_best", pilot / "checkpoint_best.pt"),
            ("epoch4_last", pilot / "checkpoint_last.pt"),
        ]
    assert model_label is not None
    if model_label == "phase2_baseline" or not re.fullmatch(r"[A-Za-z0-9_.-]+", model_label):
        raise ValueError("model-label must be a safe, non-reserved identifier")
    return [
        ("phase2_baseline", phase2_checkpoint),
        (model_label, resolve_path(checkpoint, repo_root).resolve()),
    ]


def _validate_checkpoint_data_contract(path: Path, pilot_config: dict[str, Any]) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    stored = state.get("run_config")
    if not isinstance(stored, dict):
        raise ValueError(f"training checkpoint lacks run_config:{path}")
    keys = (
        "data_version", "dataset_version", "dataset_root", "data_manifest_sha256",
        "database_contract", "cache_schema", "generator_id", "qc_id",
    )
    for key in keys:
        stored_value = stored.get(key)
        expected_value = pilot_config.get(key)
        if key == "data_version" and stored_value is None:
            stored_value = "v2"
        if key == "data_version" and expected_value is None:
            expected_value = "v2"
        if stored_value != expected_value:
            raise ValueError(f"checkpoint/pilot data contract mismatch for {key}")
    return state


def _merge_named_checkpoint_records(
    records_by_label: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    labels = list(records_by_label)
    indexed = {label: {row["interface_pair_id"]: row for row in rows} for label, rows in records_by_label.items()}
    query_sets = [set(rows) for rows in indexed.values()]
    if not query_sets or any(values != query_sets[0] for values in query_sets[1:]):
        raise RuntimeError("checkpoints did not evaluate exactly the same queries")
    prefix = {"phase2_baseline": "baseline", "epoch0_best": "epoch0", "epoch4_last": "epoch4"}
    merged: list[dict[str, Any]] = []
    for pair_id in sorted(query_sets[0]):
        rows = [indexed[label][pair_id] for label in labels]
        if len({row["target_id"] for row in rows}) != 1 or len({row["candidate_count"] for row in rows}) != 1:
            raise RuntimeError(f"candidate contract drift across checkpoints:{pair_id}")
        item = {
            "interface_pair_id": pair_id,
            "target_id": rows[0]["target_id"],
            "candidate_count": rows[0]["candidate_count"],
            "known_positive_candidates_excluded": rows[0]["known_positive_candidates_excluded"],
        }
        for label, row in zip(labels, rows):
            name = prefix.get(label, label)
            item[f"{name}_rank"] = row["rank"]
            item[f"{name}_score"] = row["target_score"]
        merged.append(item)
    return merged


def _encode_plan(
    model: torch.nn.Module,
    dataset: InterfacePairSubsetDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    sampler = UniquePeptideBatchSampler(dataset, batch_size=batch_size, seed=17, epoch=0)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=collate_phase3)
    by_pair: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for batch in loader:
            result = forward_phase2_fusion_batch(model, batch, device)
            receptors = result["receptor_embedding"].detach().float().cpu()
            peptides = result["peptide_embedding"].detach().float().cpu()
            for index, interface_pair_id in enumerate(result["batch_interface_pair_id"]):
                pair_id = str(interface_pair_id)
                if pair_id in by_pair:
                    raise RuntimeError(f"duplicate encoded interface_pair_id:{pair_id}")
                by_pair[pair_id] = {
                    "interface_pair_id": pair_id,
                    "receptor_interface_id": str(result["batch_receptor_interface_id"][index]),
                    "peptide_sequence": str(result["batch_peptide_sequence"][index]),
                    "known_positive_group": dict(result["known_positive_group"][index]),
                    "receptor_embedding": receptors[index],
                    "peptide_embedding": peptides[index],
                }
    plan_ids = [str(row["interface_pair_id"]) for row in dataset.epoch_plan()]
    if set(by_pair) != set(plan_ids):
        raise RuntimeError("encoded interface-pair IDs do not close over the fixed plan")
    return {"by_pair": by_pair, "sampler": sampler.summary()}


def _deduplicated_candidates(
    by_pair: dict[str, dict[str, Any]],
    direction: Literal["r2p", "p2r"],
) -> tuple[list[str], torch.Tensor, dict[str, int]]:
    field = "peptide_sequence" if direction == "r2p" else "receptor_interface_id"
    embedding_field = "peptide_embedding" if direction == "r2p" else "receptor_embedding"
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in by_pair.values():
        groups[str(row[field])].append(row)
    candidate_ids = sorted(groups)
    representative = [groups[candidate_id][0] for candidate_id in candidate_ids]
    embeddings = torch.stack([row[embedding_field] for row in representative])
    max_duplicate_error = 0.0
    for rows in groups.values():
        anchor = rows[0][embedding_field]
        for row in rows[1:]:
            max_duplicate_error = max(max_duplicate_error, float((anchor - row[embedding_field]).abs().max().item()))
    return candidate_ids, embeddings, {
        "source_interface_pairs": len(by_pair),
        "unique_candidates": len(candidate_ids),
        "duplicate_source_rows_removed": len(by_pair) - len(candidate_ids),
        "max_duplicate_embedding_abs_error": max_duplicate_error,
    }


def _rank_direction(
    by_pair: dict[str, dict[str, Any]],
    plan: list[dict[str, Any]],
    candidate_ids: list[str],
    candidate_embeddings: torch.Tensor,
    direction: Literal["r2p", "p2r"],
    temperature: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if direction == "r2p":
        query_embedding_field, target_field, known_field = (
            "receptor_embedding", "peptide_sequence", "receptor_peptides"
        )
    else:
        query_embedding_field, target_field, known_field = (
            "peptide_embedding", "receptor_interface_id", "peptide_receptors"
        )
    candidate_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    candidate_set = set(candidate_ids)
    query_rows = [by_pair[str(plan_row["interface_pair_id"])] for plan_row in plan]
    query_embeddings = torch.stack([row[query_embedding_field] for row in query_rows])
    scores = (query_embeddings @ candidate_embeddings.t()) / float(temperature)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("full-candidate similarity matrix is non-finite")
    records: list[dict[str, Any]] = []
    total_excluded = 0
    queries_with_exclusions = 0
    target_not_declared = 0
    for row_index, row in enumerate(query_rows):
        target_id = str(row[target_field])
        if target_id not in candidate_index:
            raise ValueError(f"target missing from {direction} candidate bank:{target_id}")
        declared_positives = {str(value) for value in row["known_positive_group"].get(known_field, [])}
        if target_id not in declared_positives:
            target_not_declared += 1
        excluded = (declared_positives - {target_id}) & candidate_set
        if excluded:
            queries_with_exclusions += 1
            total_excluded += len(excluded)
        allowed_ids = [candidate_id for candidate_id in candidate_ids if candidate_id not in excluded]
        ranked_ids = sorted(
            allowed_ids,
            key=lambda candidate_id: (-float(scores[row_index, candidate_index[candidate_id]]), candidate_id),
        )
        rank = ranked_ids.index(target_id) + 1
        records.append({
            "interface_pair_id": str(row["interface_pair_id"]),
            "target_id": target_id,
            "rank": rank,
            "target_score": float(scores[row_index, candidate_index[target_id]]),
            "candidate_count": len(allowed_ids),
            "known_positive_candidates_excluded": len(excluded),
        })
    if target_not_declared:
        raise ValueError(f"{direction} target absent from formal known-positive group for {target_not_declared} queries")
    ranks = [int(row["rank"]) for row in records]
    metrics = {
        "queries": len(records),
        "recall_at_1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "recall_at_5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "recall_at_10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "median_rank": float(torch.tensor(ranks, dtype=torch.float32).median().item()),
        "mean_rank": sum(ranks) / len(ranks),
        "hit_at_1": sum(rank <= 1 for rank in ranks),
        "hit_at_5": sum(rank <= 5 for rank in ranks),
        "hit_at_10": sum(rank <= 10 for rank in ranks),
        "candidate_count_min": min(int(row["candidate_count"]) for row in records),
        "candidate_count_max": max(int(row["candidate_count"]) for row in records),
        "known_positive_exclusion_total": total_excluded,
        "queries_with_known_positive_exclusions": queries_with_exclusions,
        "known_positive_target_missing": target_not_declared,
        "tie_policy": "descending_score_then_lexicographic_candidate_id",
    }
    return records, metrics


def _merge_checkpoint_records(
    baseline: list[dict[str, Any]], epoch0: list[dict[str, Any]], epoch4: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    indexed = [{row["interface_pair_id"]: row for row in rows} for rows in (baseline, epoch0, epoch4)]
    if not (set(indexed[0]) == set(indexed[1]) == set(indexed[2])):
        raise RuntimeError("checkpoints did not evaluate exactly the same queries")
    merged: list[dict[str, Any]] = []
    for pair_id in sorted(indexed[0]):
        rows = [index[pair_id] for index in indexed]
        if len({row["target_id"] for row in rows}) != 1 or len({row["candidate_count"] for row in rows}) != 1:
            raise RuntimeError(f"candidate contract drift across checkpoints:{pair_id}")
        merged.append({
            "interface_pair_id": pair_id,
            "target_id": rows[0]["target_id"],
            "candidate_count": rows[0]["candidate_count"],
            "known_positive_candidates_excluded": rows[0]["known_positive_candidates_excluded"],
            "baseline_rank": rows[0]["rank"], "epoch0_rank": rows[1]["rank"], "epoch4_rank": rows[2]["rank"],
            "baseline_score": rows[0]["target_score"], "epoch0_score": rows[1]["target_score"], "epoch4_score": rows[2]["target_score"],
        })
    return merged


def _comparison(rows: list[dict[str, Any]], earlier: str, later: str) -> tuple[dict[str, int], list[dict[str, Any]]]:
    earlier_rank, later_rank = f"{earlier}_rank", f"{later}_rank"
    improved = [row for row in rows if int(row[later_rank]) < int(row[earlier_rank])]
    worsened = [row for row in rows if int(row[later_rank]) > int(row[earlier_rank])]
    unchanged = [row for row in rows if int(row[later_rank]) == int(row[earlier_rank])]
    gained = [row for row in rows if int(row[earlier_rank]) > 10 and int(row[later_rank]) <= 10]
    lost = [row for row in rows if int(row[earlier_rank]) <= 10 and int(row[later_rank]) > 10]
    details = [
        {**row, "comparison": f"{later}_vs_{earlier}", "hit10_change": "gained" if row in gained else "lost"}
        for row in gained + lost
    ]
    return {
        "improved_rank": len(improved), "worsened_rank": len(worsened), "unchanged_rank": len(unchanged),
        "gained_hit_at_10": len(gained), "lost_hit_at_10": len(lost),
    }, details


def main() -> None:
    cli = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    pilot = resolve_path(cli.pilot_output, repo_root).resolve()
    output = resolve_path(cli.output_dir, repo_root).resolve() if cli.output_dir else pilot / "full_retrieval_fixed_valid_512"
    output.mkdir(parents=True, exist_ok=True)
    config, subset_manifest, plan = _validate_pilot_contract(pilot)
    dataset = _load_fixed_dataset(config, subset_manifest, plan)
    device = torch.device(cli.device)
    train_args = argparse.Namespace(**config["args"])
    train_args.device = str(device)
    phase2_checkpoint = resolve_path(config["phase2_checkpoint"], repo_root).resolve()
    source_configs = load_source_configs(phase2_checkpoint, train_args.source_model_configs, repo_root)
    checkpoint_specs = _checkpoint_specs(cli.checkpoint, cli.model_label, pilot, phase2_checkpoint, repo_root)
    checkpoint_states: dict[str, dict[str, Any]] = {}
    for label, path in checkpoint_specs:
        if not path.exists():
            raise FileNotFoundError(f"evaluation checkpoint not found: {path}")
        if label != "phase2_baseline":
            checkpoint_states[label] = _validate_checkpoint_data_contract(path, config)
    all_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    all_metrics: dict[str, dict[str, Any]] = {}
    candidate_banks: dict[str, Any] | None = None
    sampler_summary: dict[str, Any] | None = None
    for label, checkpoint_path in checkpoint_specs:
        model = _load_checkpoint_model(label, checkpoint_path, phase2_checkpoint, source_configs, train_args, repo_root, device)
        encoded = _encode_plan(model, dataset, device, cli.batch_size)
        by_pair = encoded["by_pair"]
        if sampler_summary is None:
            sampler_summary = encoded["sampler"]
        r2p_ids, r2p_embeddings, r2p_bank = _deduplicated_candidates(by_pair, "r2p")
        p2r_ids, p2r_embeddings, p2r_bank = _deduplicated_candidates(by_pair, "p2r")
        r2p_records, r2p_metrics = _rank_direction(by_pair, plan, r2p_ids, r2p_embeddings, "r2p", model.temperature)
        p2r_records, p2r_metrics = _rank_direction(by_pair, plan, p2r_ids, p2r_embeddings, "p2r", model.temperature)
        all_records[label] = {"r2p": r2p_records, "p2r": p2r_records}
        all_metrics[label] = {"receptor_to_peptide": r2p_metrics, "peptide_to_receptor": p2r_metrics}
        bank_payload = {
            "receptor_to_peptide": {**r2p_bank, "candidate_ids": r2p_ids, "candidate_ids_sha256": _sequence_hash(r2p_ids)},
            "peptide_to_receptor": {**p2r_bank, "candidate_ids": p2r_ids, "candidate_ids_sha256": _sequence_hash(p2r_ids)},
        }
        if candidate_banks is None:
            candidate_banks = bank_payload
        elif candidate_banks["receptor_to_peptide"]["candidate_ids"] != r2p_ids or candidate_banks["peptide_to_receptor"]["candidate_ids"] != p2r_ids:
            raise RuntimeError("candidate bank drift across checkpoints")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    merged_r2p = _merge_named_checkpoint_records({label: all_records[label]["r2p"] for label, _ in checkpoint_specs})
    merged_p2r = _merge_named_checkpoint_records({label: all_records[label]["p2r"] for label, _ in checkpoint_specs})
    comparisons: dict[str, Any] = {}
    comparison_pairs = [("baseline", "epoch0"), ("epoch0", "epoch4")] if cli.checkpoint is None else [("baseline", str(cli.model_label))]
    for direction, rows in (("receptor_to_peptide", merged_r2p), ("peptide_to_receptor", merged_p2r)):
        comparisons[direction] = {}
        for earlier, later in comparison_pairs:
            stats, details = _comparison(rows, earlier, later)
            name = f"{later}_vs_{earlier}"
            comparisons[direction][name] = stats
            _write_rows(output / f"hit10_changes_{direction}_{name}.jsonl", details)
    _write_rows(output / "per_query_r2p_ranks.jsonl", merged_r2p)
    _write_rows(output / "per_query_p2r_ranks.jsonl", merged_p2r)
    write_json(output / "candidate_bank.json", candidate_banks or {})
    fixed_losses: dict[str, Any] = {
        "phase2_baseline": _read_json(pilot / "initial_validation_metrics.json")["metrics"],
    }
    if cli.checkpoint is None:
        training_log = list(read_jsonl(pilot / "train_log.jsonl"))
        fixed_losses.update({"epoch0_best": training_log[0]["valid"], "epoch4_last": training_log[-1]["valid"]})
    else:
        state = checkpoint_states[str(cli.model_label)]
        fixed_losses[str(cli.model_label)] = {
            "best_validation_loss": state.get("best_validation_loss"),
            "global_step": state.get("global_step"),
        }
    report = {
        "schema_version": "phase3-drugclip-full-candidate-retrieval-v1",
        "command": shlex.join([sys.executable, "-m", "phase3.drugclip.evaluate_full_retrieval", *sys.argv[1:]]),
        "pilot_output": str(pilot),
        "requested_checkpoint": str(resolve_path(cli.checkpoint, repo_root).resolve()) if cli.checkpoint else None,
        "requested_model_label": cli.model_label,
        "fixed_validation_plan": {
            "path": str((pilot / "validation_sampling_plan.jsonl").resolve()),
            "file_sha256": _file_hash(pilot / "validation_sampling_plan.jsonl"),
            "canonical_sha256": _plan_hash(plan), "queries": len(plan), "fixed_conformer_indices": [0],
        },
        "validation_interface_pair_ids_sha256": _sequence_hash([str(row["interface_pair_id"]) for row in plan]),
        "checkpoints": {label: {"path": str(path.resolve()), "sha256": _file_hash(path)} for label, path in checkpoint_specs},
        "candidate_banks": candidate_banks,
        "known_positive_policy": "exact formal receptor_peptides / peptide_receptors; target retained; only other in-bank positives excluded",
        "embedding_batching": sampler_summary,
        "metrics": all_metrics,
        "fixed_validation_losses": fixed_losses,
        "per_query_files": {"receptor_to_peptide": "per_query_r2p_ranks.jsonl", "peptide_to_receptor": "per_query_p2r_ranks.jsonl"},
        "comparisons": comparisons,
    }
    write_json(output / "retrieval_report.json", report)
    print(json.dumps({"output_dir": str(output), "metrics": all_metrics, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()

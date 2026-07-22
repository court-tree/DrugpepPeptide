"""Ten-conformer, full-candidate robustness evaluation for the Phase-3 Pilot.

The primary result ranks every candidate using the same conformer index k
(k=0..9).  The optional mean-score result averages all ten score matrices; it
is reported separately and is never substituted for the robustness result.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shlex
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from phase3.drugclip.batching import UniquePeptideBatchSampler, collate_phase3
from phase3.drugclip.evaluate_full_retrieval import (
    _deduplicated_candidates,
    _checkpoint_specs,
    _file_hash,
    _load_checkpoint_model,
    _load_fixed_dataset,
    _read_json,
    _sequence_hash,
    _validate_pilot_contract,
    _validate_checkpoint_data_contract,
)
from phase3.drugclip.forward import forward_phase2_fusion_batch
from phase3.drugclip.io_utils import read_jsonl, write_json, write_jsonl
from phase3.drugclip.random_augmentation_dataset import InterfacePairSubsetDataset
from phase3.drugclip.train import load_source_configs, resolve_path


CONFORMER_INDICES = tuple(range(10))
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260721


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ten-conformer full-candidate retrieval robustness evaluation.")
    parser.add_argument("--pilot_output", default="phase3/runs/drugclip/pilot_interface_pair_4096_512_v1")
    parser.add_argument("--single_conformer_output", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--bootstrap_replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap_seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _configure_logging(output: Path) -> logging.Logger:
    logger = logging.getLogger("phase3.multi_conformer_retrieval")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(output / "run.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _set_fixed_conformer(
    dataset: InterfacePairSubsetDataset,
    conformer_index: int,
    reference_plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use one fixed conformer index for every interface pair, never target only."""
    if conformer_index not in CONFORMER_INDICES:
        raise ValueError("conformer index must be in [0, 9]")
    dataset.base.fixed_conformer_index = int(conformer_index)
    dataset.set_epoch(0)
    plan = dataset.epoch_plan()
    if len(plan) != len(reference_plan):
        raise RuntimeError("conformer plan changed query count")
    for actual, reference in zip(plan, reference_plan):
        for key in ("interface_pair_id", "receptor_interface_id", "peptide_sequence", "biological_pair_id"):
            if actual.get(key) != reference.get(key):
                raise RuntimeError(f"conformer plan changed fixed identity field:{key}")
        if int(actual["conformer_index"]) != conformer_index:
            raise RuntimeError("not every candidate/query received the requested conformer index")
    return plan


def _encode_current_plan(
    model: torch.nn.Module,
    dataset: InterfacePairSubsetDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    sampler = UniquePeptideBatchSampler(dataset, batch_size=batch_size, seed=17, epoch=0)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=collate_phase3)
    rows: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for batch in loader:
            result = forward_phase2_fusion_batch(model, batch, device)
            receptor = result["receptor_embedding"].detach().float().cpu()
            peptide = result["peptide_embedding"].detach().float().cpu()
            if not torch.isfinite(receptor).all() or not torch.isfinite(peptide).all():
                raise FloatingPointError("non-finite embedding in multi-conformer evaluation")
            for index, pair_id_value in enumerate(result["batch_interface_pair_id"]):
                pair_id = str(pair_id_value)
                if pair_id in rows:
                    raise RuntimeError(f"duplicate encoded interface pair:{pair_id}")
                rows[pair_id] = {
                    "interface_pair_id": pair_id,
                    "receptor_interface_id": str(result["batch_receptor_interface_id"][index]),
                    "peptide_sequence": str(result["batch_peptide_sequence"][index]),
                    "known_positive_group": dict(result["known_positive_group"][index]),
                    "receptor_embedding": receptor[index],
                    "peptide_embedding": peptide[index],
                }
    expected = {str(row["interface_pair_id"]) for row in dataset.epoch_plan()}
    if set(rows) != expected:
        raise RuntimeError("encoded rows do not equal the fixed conformer plan")
    return rows, sampler.summary()


def _rank_from_scores(
    rows_by_pair: dict[str, dict[str, Any]],
    plan: list[dict[str, Any]],
    candidate_ids: list[str],
    score_matrix: torch.Tensor,
    direction: Literal["r2p", "p2r"],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not torch.isfinite(score_matrix).all():
        raise FloatingPointError("non-finite full-candidate score")
    if direction == "r2p":
        target_field, known_field = "peptide_sequence", "receptor_peptides"
    else:
        target_field, known_field = "receptor_interface_id", "peptide_receptors"
    if score_matrix.shape != (len(plan), len(candidate_ids)):
        raise ValueError("score matrix does not match query/candidate dimensions")
    candidate_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    candidate_set = set(candidate_ids)
    records: list[dict[str, Any]] = []
    excluded_total = 0
    exclusion_queries = 0
    target_missing = 0
    for query_index, plan_row in enumerate(plan):
        row = rows_by_pair[str(plan_row["interface_pair_id"])]
        target = str(row[target_field])
        if target not in candidate_index:
            target_missing += 1
            continue
        declared = {str(value) for value in row["known_positive_group"].get(known_field, [])}
        if target not in declared:
            raise ValueError(f"target absent from exact known-positive group:{direction}:{target}")
        excluded = (declared - {target}) & candidate_set
        if excluded:
            exclusion_queries += 1
            excluded_total += len(excluded)
        allowed = [candidate for candidate in candidate_ids if candidate not in excluded]
        ranked = sorted(
            allowed,
            key=lambda candidate: (-float(score_matrix[query_index, candidate_index[candidate]]), candidate),
        )
        records.append({
            "interface_pair_id": str(row["interface_pair_id"]),
            "target_id": target,
            "rank": ranked.index(target) + 1,
            "target_score": float(score_matrix[query_index, candidate_index[target]]),
            "candidate_count": len(allowed),
            "known_positive_candidates_excluded": len(excluded),
        })
    if target_missing:
        raise ValueError(f"missing target count is nonzero:{direction}:{target_missing}")
    ranks = [int(row["rank"]) for row in records]
    if not ranks:
        raise ValueError("no valid retrieval queries")
    return records, {
        "query_count": len(records),
        "recall_at_1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "recall_at_5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "recall_at_10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "median_rank": float(torch.tensor(ranks, dtype=torch.float64).median().item()),
        "mean_rank": sum(ranks) / len(ranks),
        "hit_at_1": sum(rank <= 1 for rank in ranks),
        "hit_at_5": sum(rank <= 5 for rank in ranks),
        "hit_at_10": sum(rank <= 10 for rank in ranks),
        "candidate_count_min": min(int(row["candidate_count"]) for row in records),
        "candidate_count_max": max(int(row["candidate_count"]) for row in records),
        "known_positive_exclusion_total": excluded_total,
        "queries_with_known_positive_exclusions": exclusion_queries,
        "target_missing": target_missing,
        "tie_policy": "descending_score_then_lexicographic_candidate_id",
    }


def _mean_score(score_matrices: list[torch.Tensor]) -> torch.Tensor:
    """Secondary result: all candidates, all ten conformers, arithmetic mean only."""
    if len(score_matrices) != len(CONFORMER_INDICES):
        raise ValueError("mean-score evaluation requires exactly ten conformer score matrices")
    shape = score_matrices[0].shape
    if not all(matrix.shape == shape and torch.isfinite(matrix).all() for matrix in score_matrices):
        raise ValueError("mean-score matrices have inconsistent shapes or non-finite values")
    return torch.stack(score_matrices, dim=0).mean(dim=0)


def _aggregate_queries(
    per_conformer: dict[int, list[dict[str, Any]]], model_name: str
) -> list[dict[str, Any]]:
    indices = sorted(per_conformer)
    expected = {row["interface_pair_id"] for row in per_conformer[indices[0]]}
    records_by_index = [{row["interface_pair_id"]: row for row in per_conformer[index]} for index in indices]
    if any(set(mapping) != expected for mapping in records_by_index):
        raise RuntimeError("query IDs differ across conformer indices")
    aggregated: list[dict[str, Any]] = []
    for pair_id in sorted(expected):
        rows = [mapping[pair_id] for mapping in records_by_index]
        if len({row["target_id"] for row in rows}) != 1 or len({row["candidate_count"] for row in rows}) != 1:
            raise RuntimeError("target or candidate contract changed across conformers")
        ranks = [int(row["rank"]) for row in rows]
        scores = [float(row["target_score"]) for row in rows]
        aggregated.append({
            "model_name": model_name,
            "interface_pair_id": pair_id,
            "target_id": rows[0]["target_id"],
            "candidate_count": rows[0]["candidate_count"],
            "ranks_by_conformer": ranks,
            "scores_by_conformer": scores,
            "mean_rank": statistics.mean(ranks),
            "median_rank": statistics.median(ranks),
            "best_rank": min(ranks),
            "worst_rank": max(ranks),
            "rank_std": statistics.pstdev(ranks),
            "mean_target_score": statistics.mean(scores),
            "target_score_std": statistics.pstdev(scores),
            "hit_at_1_count": sum(rank <= 1 for rank in ranks),
            "hit_at_10_count": sum(rank <= 10 for rank in ranks),
        })
    return aggregated


def _metric_summary(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "mean_rank")
    summary = {key: {"mean": statistics.mean(row[key] for row in metric_rows), "std": statistics.pstdev(row[key] for row in metric_rows)} for key in numeric}
    medians = [row["median_rank"] for row in metric_rows]
    summary["median_rank"] = {"mean": statistics.mean(medians), "min": min(medians), "max": max(medians)}
    return summary


def _paired_comparison(earlier: list[dict[str, Any]], later: list[dict[str, Any]]) -> dict[str, int]:
    first = {row["interface_pair_id"]: row for row in earlier}
    second = {row["interface_pair_id"]: row for row in later}
    if set(first) != set(second):
        raise RuntimeError("paired comparison query mismatch")
    rows = [(first[key], second[key]) for key in sorted(first)]
    def counts(field: str) -> tuple[int, int, int]:
        improved = sum(second_row[field] < first_row[field] for first_row, second_row in rows)
        worsened = sum(second_row[field] > first_row[field] for first_row, second_row in rows)
        return improved, worsened, len(rows) - improved - worsened
    mean_better, mean_worse, mean_same = counts("mean_rank")
    worst_better, worst_worse, _ = counts("worst_rank")
    std_better, std_worse, _ = counts("rank_std")
    hit_increase = sum(second_row["hit_at_10_count"] > first_row["hit_at_10_count"] for first_row, second_row in rows)
    hit_decrease = sum(second_row["hit_at_10_count"] < first_row["hit_at_10_count"] for first_row, second_row in rows)
    never_to_any = sum(first_row["hit_at_10_count"] == 0 and second_row["hit_at_10_count"] > 0 for first_row, second_row in rows)
    any_to_never = sum(first_row["hit_at_10_count"] > 0 and second_row["hit_at_10_count"] == 0 for first_row, second_row in rows)
    return {
        "queries": len(rows),
        "mean_rank_improved": mean_better, "mean_rank_worsened": mean_worse, "mean_rank_unchanged": mean_same,
        "worst_rank_improved": worst_better, "worst_rank_worsened": worst_worse,
        "rank_std_decreased": std_better, "rank_std_increased": std_worse,
        "hit_at_10_count_increased": hit_increase, "hit_at_10_count_decreased": hit_decrease,
        "never_hit10_to_any_hit10": never_to_any, "any_hit10_to_never_hit10": any_to_never,
    }


def _bootstrap_difference(
    earlier: list[dict[str, Any]], later: list[dict[str, Any]], replicates: int, seed: int
) -> dict[str, Any]:
    if replicates < 2000:
        raise ValueError("paired bootstrap requires at least 2000 replicates")
    first = {row["interface_pair_id"]: row for row in earlier}
    second = {row["interface_pair_id"]: row for row in later}
    if set(first) != set(second):
        raise RuntimeError("bootstrap query mismatch")
    ordered = sorted(first)
    values: dict[str, torch.Tensor] = {
        "mean_recall_at_1": torch.tensor([(sum(rank <= 1 for rank in first[key]["ranks_by_conformer"]) / 10.0) for key in ordered], dtype=torch.float64),
        "mean_recall_at_10": torch.tensor([(sum(rank <= 10 for rank in first[key]["ranks_by_conformer"]) / 10.0) for key in ordered], dtype=torch.float64),
        "mean_mrr": torch.tensor([statistics.mean(1.0 / rank for rank in first[key]["ranks_by_conformer"]) for key in ordered], dtype=torch.float64),
        "mean_rank": torch.tensor([first[key]["mean_rank"] for key in ordered], dtype=torch.float64),
        "worst_rank": torch.tensor([first[key]["worst_rank"] for key in ordered], dtype=torch.float64),
        "rank_std": torch.tensor([first[key]["rank_std"] for key in ordered], dtype=torch.float64),
    }
    later_values: dict[str, torch.Tensor] = {
        "mean_recall_at_1": torch.tensor([(sum(rank <= 1 for rank in second[key]["ranks_by_conformer"]) / 10.0) for key in ordered], dtype=torch.float64),
        "mean_recall_at_10": torch.tensor([(sum(rank <= 10 for rank in second[key]["ranks_by_conformer"]) / 10.0) for key in ordered], dtype=torch.float64),
        "mean_mrr": torch.tensor([statistics.mean(1.0 / rank for rank in second[key]["ranks_by_conformer"]) for key in ordered], dtype=torch.float64),
        "mean_rank": torch.tensor([second[key]["mean_rank"] for key in ordered], dtype=torch.float64),
        "worst_rank": torch.tensor([second[key]["worst_rank"] for key in ordered], dtype=torch.float64),
        "rank_std": torch.tensor([second[key]["rank_std"] for key in ordered], dtype=torch.float64),
    }
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, len(ordered), (replicates, len(ordered)), generator=generator)
    output: dict[str, Any] = {"queries": len(ordered), "replicates": replicates, "seed": seed, "later_minus_earlier": {}}
    for name, earlier_values in values.items():
        delta = later_values[name] - earlier_values
        samples = delta[indices].mean(dim=1)
        ci_low, ci_high = torch.quantile(samples, torch.tensor([0.025, 0.975], dtype=samples.dtype)).tolist()
        point = float(delta.mean().item())
        output["later_minus_earlier"][name] = {
            "point_estimate": point, "ci_95": [float(ci_low), float(ci_high)],
            "crosses_zero": bool(ci_low <= 0.0 <= ci_high),
        }
    return output


def _check_conformer_zero_regression(
    output_records: dict[str, dict[str, list[dict[str, Any]]]],
    single_output: Path,
    labels: tuple[str, ...],
) -> None:
    source_files = {"r2p": "per_query_r2p_ranks.jsonl", "p2r": "per_query_p2r_ranks.jsonl"}
    for direction, filename in source_files.items():
        source = {str(row["interface_pair_id"]): row for row in read_jsonl(single_output / filename)}
        current = {str(row["interface_pair_id"]): row for row in output_records[direction]["conformer_0"]}
        if set(source) != set(current):
            raise RuntimeError(f"conformer 0 regression query mismatch:{direction}")
        for pair_id in source:
            old, new = source[pair_id], current[pair_id]
            label_to_prefix = {"phase2_baseline": "baseline", "epoch0_best": "epoch0", "epoch4_last": "epoch4"}
            for label in labels:
                prefix = label_to_prefix.get(label, label)
                model_record = new[label]
                if int(old[f"{prefix}_rank"]) != int(model_record["rank"]):
                    raise RuntimeError(f"conformer 0 rank regression:{direction}:{label}:{pair_id}")
                if abs(float(old[f"{prefix}_score"]) - float(model_record["target_score"])) > 2e-5:
                    raise RuntimeError(f"conformer 0 score regression:{direction}:{label}:{pair_id}")


def main() -> None:
    cli = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    pilot = resolve_path(cli.pilot_output, repo_root).resolve()
    single_output = resolve_path(cli.single_conformer_output, repo_root).resolve() if cli.single_conformer_output else pilot / "full_retrieval_fixed_valid_512"
    output = resolve_path(cli.output_dir, repo_root).resolve() if cli.output_dir else pilot / "multi_conformer_retrieval_valid_512"
    output.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(output)
    config, subset_manifest, reference_plan = _validate_pilot_contract(pilot)
    dataset = _load_fixed_dataset(config, subset_manifest, reference_plan)
    device = torch.device(cli.device)
    train_args = argparse.Namespace(**config["args"])
    train_args.device = str(device)
    phase2_checkpoint = resolve_path(config["phase2_checkpoint"], repo_root).resolve()
    source_configs = load_source_configs(phase2_checkpoint, train_args.source_model_configs, repo_root)
    checkpoint_specs = _checkpoint_specs(cli.checkpoint, cli.model_label, pilot, phase2_checkpoint, repo_root)
    checkpoint_paths = dict(checkpoint_specs)
    for model_name, checkpoint_path in checkpoint_specs:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"evaluation checkpoint not found: {checkpoint_path}")
        if model_name != "phase2_baseline":
            _validate_checkpoint_data_contract(checkpoint_path, config)
    metrics_rows: list[dict[str, Any]] = []
    mean_score_rows: list[dict[str, Any]] = []
    aggregated_by_model_direction: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    candidate_contract: dict[str, Any] | None = None
    expected_candidate_bank = _read_json(single_output / "candidate_bank.json")
    expected_candidate_ids = {
        "r2p": [str(value) for value in expected_candidate_bank["receptor_to_peptide"]["candidate_ids"]],
        "p2r": [str(value) for value in expected_candidate_bank["peptide_to_receptor"]["candidate_ids"]],
    }
    global_candidate_ids: dict[str, list[str]] | None = None
    embedding_batching: dict[str, Any] | None = None
    all_per_conformer: dict[str, dict[str, dict[int, list[dict[str, Any]]]]] = {}

    for model_name, _ in checkpoint_specs:
        logger.info("loading model=%s", model_name)
        model = _load_checkpoint_model(model_name, checkpoint_paths[model_name], phase2_checkpoint, source_configs, train_args, repo_root, device)
        per_direction: dict[str, dict[int, list[dict[str, Any]]]] = {"r2p": {}, "p2r": {}}
        score_matrices: dict[str, list[torch.Tensor]] = {"r2p": [], "p2r": []}
        receptor_embeddings_fixed: dict[str, torch.Tensor] | None = None
        receptor_candidate_ids: list[str] | None = None
        receptor_candidate_embeddings: torch.Tensor | None = None
        candidate_ids_reference: dict[str, list[str]] | None = None
        rows_by_k: dict[int, dict[str, dict[str, Any]]] = {}
        for conformer_index in CONFORMER_INDICES:
            plan = _set_fixed_conformer(dataset, conformer_index, reference_plan)
            rows, sampler_summary = _encode_current_plan(model, dataset, device, cli.batch_size)
            rows_by_k[conformer_index] = rows
            if embedding_batching is None:
                embedding_batching = sampler_summary
            peptide_ids, peptide_embeddings, peptide_bank = _deduplicated_candidates(rows, "r2p")
            receptor_ids, receptor_embeddings, receptor_bank = _deduplicated_candidates(rows, "p2r")
            ids = {"r2p": peptide_ids, "p2r": receptor_ids}
            if ids != expected_candidate_ids:
                raise RuntimeError("candidate IDs/order differ from the completed single-conformer evaluation")
            if global_candidate_ids is None:
                global_candidate_ids = ids
            elif ids != global_candidate_ids:
                raise RuntimeError("candidate IDs/order changed across checkpoints")
            if candidate_ids_reference is None:
                candidate_ids_reference = ids
                candidate_contract = {"receptor_to_peptide": peptide_bank, "peptide_to_receptor": receptor_bank,
                                      "receptor_to_peptide_candidate_ids_sha256": _sequence_hash(peptide_ids),
                                      "peptide_to_receptor_candidate_ids_sha256": _sequence_hash(receptor_ids),
                                      "matches_single_conformer_candidate_bank": True}
            elif ids != candidate_ids_reference:
                raise RuntimeError("candidate IDs/order changed across conformer indices")
            if conformer_index == 0:
                receptor_embeddings_fixed = {pair_id: row["receptor_embedding"] for pair_id, row in rows.items()}
                receptor_candidate_ids = receptor_ids
                receptor_candidate_embeddings = receptor_embeddings
            else:
                assert receptor_embeddings_fixed is not None
                max_error = max(float((rows[pair_id]["receptor_embedding"] - receptor_embeddings_fixed[pair_id]).abs().max().item()) for pair_id in rows)
                if max_error > 2e-5:
                    raise RuntimeError(f"receptor changed when only peptide conformer changed:{max_error}")
            assert receptor_embeddings_fixed is not None and receptor_candidate_ids is not None and receptor_candidate_embeddings is not None
            r2p_queries = torch.stack([receptor_embeddings_fixed[str(row["interface_pair_id"])] for row in plan])
            p2r_queries = torch.stack([rows[str(row["interface_pair_id"])]["peptide_embedding"] for row in plan])
            r2p_scores = r2p_queries @ peptide_embeddings.t() / float(model.temperature)
            p2r_scores = p2r_queries @ receptor_candidate_embeddings.t() / float(model.temperature)
            r2p_records, r2p_metrics = _rank_from_scores(rows, plan, peptide_ids, r2p_scores, "r2p")
            p2r_records, p2r_metrics = _rank_from_scores(rows, plan, receptor_candidate_ids, p2r_scores, "p2r")
            for direction, records, metrics in (("r2p", r2p_records, r2p_metrics), ("p2r", p2r_records, p2r_metrics)):
                per_direction[direction][conformer_index] = records
                metrics_rows.append({"model_name": model_name, "direction": direction, "conformer_index": conformer_index, **metrics})
            score_matrices["r2p"].append(r2p_scores)
            score_matrices["p2r"].append(p2r_scores)
            logger.info("model=%s conformer=%d r2p_r10=%.4f p2r_r10=%.4f", model_name, conformer_index, r2p_metrics["recall_at_10"], p2r_metrics["recall_at_10"])
        for direction, candidate_ids in (("r2p", candidate_ids_reference["r2p"]), ("p2r", candidate_ids_reference["p2r"])):
            mean_records, mean_metrics = _rank_from_scores(rows_by_k[0], reference_plan, candidate_ids, _mean_score(score_matrices[direction]), direction)  # metadata is identity-invariant
            mean_score_rows.append({"model_name": model_name, "direction": direction, "aggregation": "multi_conformer_mean_score", **mean_metrics})
            aggregated_by_model_direction[model_name][direction] = _aggregate_queries(per_direction[direction], model_name)
        all_per_conformer[model_name] = per_direction
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Keep an explicit conformer-0, per-model view for strict regression against the preceding evaluation.
    regression_view: dict[str, dict[str, list[dict[str, Any]]]] = {"r2p": {"conformer_0": {}}, "p2r": {"conformer_0": {}}}  # type: ignore[assignment]
    for direction in ("r2p", "p2r"):
        merged: dict[str, dict[str, Any]] = {}
        for model_name in checkpoint_paths:
            for row in all_per_conformer[model_name][direction][0]:
                merged.setdefault(row["interface_pair_id"], {"interface_pair_id": row["interface_pair_id"]})[model_name] = row
        regression_view[direction]["conformer_0"] = list(merged.values())
    _check_conformer_zero_regression(regression_view, single_output, tuple(checkpoint_paths))

    per_r2p = [row for model in checkpoint_paths for row in aggregated_by_model_direction[model]["r2p"]]
    per_p2r = [row for model in checkpoint_paths for row in aggregated_by_model_direction[model]["p2r"]]
    write_jsonl(output / "multi_conformer_metrics.jsonl", metrics_rows)
    write_jsonl(output / "per_query_multi_conformer_r2p.jsonl", per_r2p)
    write_jsonl(output / "per_query_multi_conformer_p2r.jsonl", per_p2r)
    write_json(output / "mean_score_retrieval_metrics.json", {"aggregation": "arithmetic_mean_across_all_10_conformer_scores", "metrics": mean_score_rows})
    summaries = {model: {direction: _metric_summary([row for row in metrics_rows if row["model_name"] == model and row["direction"] == direction]) for direction in ("r2p", "p2r")} for model in checkpoint_paths}
    write_json(output / "multi_conformer_summary.json", {"primary_per_conformer_summary": summaries, "mean_score_secondary": mean_score_rows})

    bootstrap: dict[str, Any] = {}
    paired_outputs: dict[str, dict[str, Any]] = {}
    later_labels = [name for name in checkpoint_paths if name != "phase2_baseline"]
    for direction in ("r2p", "p2r"):
        phase2 = aggregated_by_model_direction["phase2_baseline"][direction]
        earlier_label = "phase2_baseline"
        earlier = phase2
        for comparison_index, later_label in enumerate(later_labels):
            later = aggregated_by_model_direction[later_label][direction]
            name = f"{later_label}_vs_{earlier_label}"
            direction_name = "receptor_to_peptide" if direction == "r2p" else "peptide_to_receptor"
            paired_outputs.setdefault(name, {})[direction_name] = _paired_comparison(earlier, later)
            bootstrap[f"{name}_{direction}"] = _bootstrap_difference(
                earlier, later, cli.bootstrap_replicates,
                cli.bootstrap_seed + comparison_index * 10 + (0 if direction == "r2p" else 1),
            )
            earlier_label, earlier = later_label, later
    for name, payload in paired_outputs.items():
        legacy_name = {
            "epoch0_best_vs_phase2_baseline": "epoch0_vs_phase2",
            "epoch4_last_vs_epoch0_best": "epoch4_vs_epoch0",
        }.get(name, name)
        write_json(output / f"paired_comparison_{legacy_name}.json", payload)
    write_json(output / "bootstrap_confidence_intervals.json", bootstrap)

    evaluation_config = {
        "schema_version": "phase3-drugclip-multi-conformer-full-retrieval-v1",
        "command": shlex.join([sys.executable, "-m", "phase3.drugclip.evaluate_multi_conformer_retrieval", *sys.argv[1:]]),
        "pilot_output": str(pilot), "single_conformer_output": str(single_output),
        "requested_checkpoint": str(resolve_path(cli.checkpoint, repo_root).resolve()) if cli.checkpoint else None,
        "requested_model_label": cli.model_label,
        "fixed_validation_plan_sha256": config["fixed_validation_plan_sha256"],
        "validation_interface_pair_ids_sha256": config["valid_interface_pair_ids_sha256"],
        "conformer_indices": list(CONFORMER_INDICES),
        "primary_result": "one full candidate bank retrieval per conformer index; all peptide candidates/query peptides use index k",
        "secondary_result": "arithmetic mean of all ten full score matrices; max score is not used",
        "known_positive_policy": "same exact formal group as single-conformer evaluation; target retained; other in-bank positives excluded",
        "candidate_contract": candidate_contract, "embedding_batching": embedding_batching,
        "checkpoints": {name: {"path": str(path.resolve()), "sha256": _file_hash(path)} for name, path in checkpoint_paths.items()},
        "bootstrap": {"replicates": cli.bootstrap_replicates, "seed": cli.bootstrap_seed, "paired_queries": 512},
        "conformer_zero_regression": "passed",
    }
    write_json(output / "evaluation_config.json", evaluation_config)
    logger.info("complete output=%s", output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

# Step-5 probabilistic sampling: absolute dedup + rule-based eligible pool + coverage-weighted sampling.

import argparse
import json
import os
import random
import shutil
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def mean_or_zero(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def normalize_step4_row(row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)

    if "rBSA_raw" not in row:
        if "rBSA_proxy" in row:
            row["rBSA_raw"] = row["rBSA_proxy"]
        elif "rBSA" in row:
            row["rBSA_raw"] = row["rBSA"]
        else:
            row["rBSA_raw"] = 0.0

    if "rBSA_proxy" not in row:
        row["rBSA_proxy"] = row["rBSA_raw"]

    if "pocket_num_residues" not in row:
        if "pocket_size_6A" in row:
            row["pocket_num_residues"] = row["pocket_size_6A"]
        elif "n_contact_residues_step4" in row:
            row["pocket_num_residues"] = row["n_contact_residues_step4"]
        elif "n_contact_residues_6A" in row:
            row["pocket_num_residues"] = row["n_contact_residues_6A"]
        else:
            row["pocket_num_residues"] = 0

    if "n_contact_residues_step4" not in row and "n_contact_residues_6A" in row:
        row["n_contact_residues_step4"] = row["n_contact_residues_6A"]

    if "contact_coverage_6A" not in row:
        if "contact_coverage_6a" in row:
            row["contact_coverage_6A"] = row["contact_coverage_6a"]
        else:
            row["contact_coverage_6A"] = 0.0

    if "covalent_bias_risk" not in row:
        row["covalent_bias_risk"] = 0.0

    if "peptide_length" not in row:
        if "selected_window_len" in row:
            row["peptide_length"] = row["selected_window_len"]
        else:
            row["peptide_length"] = 0

    return row


def compute_score_final(row: Dict[str, Any]) -> float:
    rbsa = safe_float(row.get("rBSA_raw"), 0.0)
    cov6 = safe_float(row.get("contact_coverage_6A"), 0.0)
    bias = safe_float(row.get("covalent_bias_risk"), 0.0)
    return 0.60 * rbsa + 0.30 * cov6 - 0.15 * bias


def selection_priority_key(row: Dict[str, Any]) -> Tuple[float, int, float, float, str]:
    return (
        safe_float(row.get("contact_coverage_6A"), 0.0),
        safe_int(row.get("n_contact_residues_6A"), 0),
        safe_float(row.get("rBSA_raw"), 0.0),
        -safe_float(row.get("covalent_bias_risk"), 0.0),
        str(row.get("candidate_id", "")),
    )


def join_step3_step4(
    step3_rows: List[Dict[str, Any]],
    step4_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    step3_by_id: Dict[str, Dict[str, Any]] = {}
    for row in step3_rows:
        cid = row.get("candidate_id")
        if cid:
            step3_by_id[cid] = row

    step4_by_id: Dict[str, Dict[str, Any]] = {}
    for row in step4_rows:
        norm = normalize_step4_row(row)
        cid = norm.get("candidate_id")
        if cid:
            step4_by_id[cid] = norm

    missing_in_step4 = sum(1 for cid in step3_by_id if cid not in step4_by_id)
    missing_in_step3 = 0
    joined: List[Dict[str, Any]] = []

    for cid, s4 in step4_by_id.items():
        s3 = step3_by_id.get(cid)
        if s3 is None:
            missing_in_step3 += 1
            continue

        merged = dict(s3)
        merged.update(s4)
        if "parent_task_id" not in merged or merged["parent_task_id"] in (None, ""):
            merged["parent_task_id"] = s3.get("parent_task_id", s4.get("parent_task_id"))
        merged["score_final_before"] = compute_score_final(merged)
        merged["score_final"] = merged["score_final_before"]
        joined.append(merged)

    return joined, missing_in_step4, missing_in_step3


def join_task_step3_step4(
    step3_rows: List[Dict[str, Any]],
    step4_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    return join_step3_step4(step3_rows, step4_rows)


def get_window_bounds(row: Dict[str, Any]) -> Tuple[int, int]:
    left = row.get("final_left_index", row.get("grown_left_index", row.get("left_index", -1)))
    right = row.get("final_right_index", row.get("grown_right_index", row.get("right_index", -1)))
    return safe_int(left, -1), safe_int(right, -1)


def interval_iou(a_l: int, a_r: int, b_l: int, b_r: int) -> float:
    inter = max(0, min(a_r, b_r) - max(a_l, b_l) + 1)
    if inter <= 0:
        return 0.0
    union = (a_r - a_l + 1) + (b_r - b_l + 1) - inter
    if union <= 0:
        return 0.0
    return inter / union


def interval_contains(a_l: int, a_r: int, b_l: int, b_r: int) -> bool:
    return a_l <= b_l and a_r >= b_r


def is_absolute_clone(
    a: Dict[str, Any],
    b: Dict[str, Any],
    iou_threshold: float = 0.80,
    peptide_anchor_tol: int = 2,
    receptor_anchor_tol: int = 2,
    length_diff_tol: int = 2,
) -> bool:
    a_l, a_r = get_window_bounds(a)
    b_l, b_r = get_window_bounds(b)
    if min(a_l, a_r, b_l, b_r) < 0:
        return False

    overlap_like_clone = (
        interval_contains(a_l, a_r, b_l, b_r)
        or interval_contains(b_l, b_r, a_l, a_r)
        or interval_iou(a_l, a_r, b_l, b_r) >= iou_threshold
    )
    if not overlap_like_clone:
        return False

    a_anchor_pep = safe_int(a.get("anchor_peptide_res_index"), -999999)
    b_anchor_pep = safe_int(b.get("anchor_peptide_res_index"), -999999)
    if abs(a_anchor_pep - b_anchor_pep) > peptide_anchor_tol:
        return False

    a_anchor_rec = safe_int(a.get("anchor_receptor_res_index"), -999999)
    b_anchor_rec = safe_int(b.get("anchor_receptor_res_index"), -999999)
    if abs(a_anchor_rec - b_anchor_rec) > receptor_anchor_tol:
        return False

    a_len = safe_int(a.get("peptide_length"), 0)
    b_len = safe_int(b.get("peptide_length"), 0)
    if abs(a_len - b_len) > length_diff_tol:
        return False

    return True


def redundancy_priority_key(row: Dict[str, Any]) -> Tuple[float, float, float, float, str]:
    return selection_priority_key(row)


def dedup_within_task(
    rows: List[Dict[str, Any]],
    iou_threshold: float,
    peptide_anchor_tol: int,
    receptor_anchor_tol: int,
    length_diff_tol: int,
) -> Tuple[List[Dict[str, Any]], int]:
    ordered = sorted(rows, key=redundancy_priority_key, reverse=True)
    kept: List[Dict[str, Any]] = []
    dropped = 0

    for row in ordered:
        redundant = False
        for prev in kept:
            if is_absolute_clone(
                row,
                prev,
                iou_threshold=iou_threshold,
                peptide_anchor_tol=peptide_anchor_tol,
                receptor_anchor_tol=receptor_anchor_tol,
                length_diff_tol=length_diff_tol,
            ):
                redundant = True
                break
        if redundant:
            dropped += 1
        else:
            kept.append(row)

    return kept, dropped


def stable_task_seed(parent_task_id: str, base_seed: int) -> int:
    return int(base_seed + sum(ord(c) for c in str(parent_task_id)))


def attach_sampling_weight(
    rows: List[Dict[str, Any]],
    density_threshold: float,
    min_weight: float = 1e-3,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        cov = safe_float(row.get("contact_coverage_6A"), 0.0)

        density_margin = max(0.0, cov - density_threshold)
        sampling_weight = max(min_weight, density_margin if density_margin > 0 else cov)

        row["_density_margin"] = density_margin
        row["_sampling_basis"] = "contact_coverage_6A"
        row["_sampling_weight"] = sampling_weight
        out.append(row)
    return out


def weighted_sample_without_replacement(
    rows: List[Dict[str, Any]],
    k: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    pool = list(rows)
    chosen: List[Dict[str, Any]] = []

    while pool and len(chosen) < k:
        weights = [max(0.0, safe_float(x.get("_sampling_weight"), 0.0)) for x in pool]
        total = sum(weights)
        if total <= 0:
            idx = rng.randrange(len(pool))
            chosen.append(pool.pop(idx))
            continue

        r = rng.random() * total
        acc = 0.0
        idx = len(pool) - 1
        for i, w in enumerate(weights):
            acc += w
            if acc >= r:
                idx = i
                break
        chosen.append(pool.pop(idx))

    return chosen


def fallback_rank_key(row: Dict[str, Any]) -> Tuple[float, float, float, float, str]:
    return selection_priority_key(row)


def bucket_length(length: int) -> str:
    if 8 <= length <= 10:
        return "8-10"
    if 11 <= length <= 15:
        return "11-15"
    if 16 <= length <= 20:
        return "16-20"
    return "other"


def process_task_rows(
    parent_task_id: str,
    rows: List[Dict[str, Any]],
    top_k: int,
    density_threshold: float,
    min_len: int,
    max_len: int,
    min_contact_residues_6a: int,
    min_rbsa_raw: float,
    random_seed: int,
    abs_iou_threshold: float,
    abs_peptide_anchor_tol: int,
    abs_receptor_anchor_tol: int,
    abs_length_diff_tol: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    before_scores: List[float] = [safe_float(x.get("score_final_before"), 0.0) for x in rows]

    deduped, n_redundant = dedup_within_task(
        rows=rows,
        iou_threshold=abs_iou_threshold,
        peptide_anchor_tol=abs_peptide_anchor_tol,
        receptor_anchor_tol=abs_receptor_anchor_tol,
        length_diff_tol=abs_length_diff_tol,
    )

    eligible_pool: List[Dict[str, Any]] = []
    fallback_pool: List[Dict[str, Any]] = []

    for row in deduped:
        peptide_len = safe_int(row.get("peptide_length"), 0)
        cov = safe_float(row.get("contact_coverage_6A"), 0.0)
        n_contact_res_6a = safe_int(row.get("n_contact_residues_6A"), 0)
        rbsa_raw = safe_float(row.get("rBSA_raw"), 0.0)
        if (
            min_len <= peptide_len <= max_len
            and cov >= density_threshold
            and n_contact_res_6a >= min_contact_residues_6a
            and rbsa_raw >= min_rbsa_raw
        ):
            eligible_pool.append(row)
        else:
            fallback_pool.append(row)

    chosen: List[Dict[str, Any]] = []
    chosen_ids = set()
    rng = random.Random(stable_task_seed(parent_task_id, random_seed))
    sampled_count = 0
    filled_count = 0
    sampling_active = False

    if eligible_pool:
        sampling_active = True
        eligible_scored = attach_sampling_weight(
            rows=eligible_pool,
            density_threshold=density_threshold,
            min_weight=1e-3,
        )
        sampled = weighted_sample_without_replacement(
            rows=eligible_scored,
            k=min(top_k, len(eligible_scored)),
            rng=rng,
        )
        sampled_count = len(sampled)
        for row in sampled:
            row = dict(row)
            row["step5_selected_by"] = "eligible_sampling"
            row["step5_selected_by_sampling"] = True
            row["step5_selection_mode"] = "sampling"
            row["step5_sampling_weight"] = safe_float(row.get("_sampling_weight"), 0.0)
            chosen.append(row)
            chosen_ids.add(row.get("candidate_id"))

    if len(chosen) < top_k:
        need = top_k - len(chosen)
        fallback_candidates = [x for x in (fallback_pool + eligible_pool) if x.get("candidate_id") not in chosen_ids]
        fallback_candidates = sorted(fallback_candidates, key=fallback_rank_key, reverse=True)
        filled_rows = fallback_candidates[:need]
        filled_count = len(filled_rows)

        for row in filled_rows:
            row = dict(row)
            row["_density_margin"] = safe_float(row.get("_density_margin"), 0.0)
            row["_sampling_basis"] = str(row.get("_sampling_basis", "contact_coverage_6A"))
            row["_sampling_weight"] = safe_float(row.get("_sampling_weight"), 0.0)
            row["step5_selected_by"] = "fallback_score"
            row["step5_selected_by_sampling"] = False
            row["step5_selection_mode"] = "fallback_score"
            row["step5_sampling_weight"] = None
            chosen.append(row)
            chosen_ids.add(row.get("candidate_id"))

    chosen = sorted(
        chosen,
        key=lambda x: (
            0 if bool(x.get("step5_selected_by_sampling", False)) else 1,
            -safe_float(x.get("contact_coverage_6A"), 0.0),
            -safe_int(x.get("n_contact_residues_6A"), 0),
            -safe_float(x.get("rBSA_raw"), 0.0),
            safe_float(x.get("covalent_bias_risk"), 0.0),
            str(x.get("candidate_id", "")),
        ),
    )

    after_scores: List[float] = []
    for rank, row in enumerate(chosen, start=1):
        row["score_final_after"] = safe_float(row.get("score_final_before"), 0.0)
        row["score_final"] = row["score_final_after"]
        row["step5_parent_task_rank"] = rank
        row["step5_rank_within_task"] = rank
        row["step5_top_k"] = top_k
        row["step5_task_candidate_count_before"] = len(rows)
        row["step5_task_candidate_count_after_dedup"] = len(deduped)
        row["step5_task_candidate_count_eligible"] = len(eligible_pool)
        row["step5_task_candidate_count_final"] = len(chosen)
        row["step5_density_threshold"] = density_threshold
        row["step5_sampling_min_len"] = min_len
        row["step5_sampling_max_len"] = max_len
        row["step5_min_contact_residues_6A"] = min_contact_residues_6a
        row["step5_min_rbsa_raw"] = min_rbsa_raw
        row["step5_sampling_power"] = None
        row["step5_length_bonus_strength"] = None
        row["step5_abs_iou_threshold"] = abs_iou_threshold
        row["step5_abs_peptide_anchor_tol"] = abs_peptide_anchor_tol
        row["step5_abs_receptor_anchor_tol"] = abs_receptor_anchor_tol
        row["step5_abs_length_diff_tol"] = abs_length_diff_tol
        after_scores.append(row["score_final_after"])

    stats = {
        "before_count": len(rows),
        "after_dedup_count": len(deduped),
        "final_count": len(chosen),
        "eligible_count": len(eligible_pool),
        "abs_redundant_dropped": n_redundant,
        "topk_dropped": max(0, len(deduped) - len(chosen)),
        "truncated_by_topk": 1 if len(deduped) > top_k else 0,
        "sampled_count": sampled_count,
        "filled_count": filled_count,
        "sampling_active": 1 if sampling_active else 0,
        "no_eligible_pool": 0 if eligible_pool else 1,
        "before_scores": before_scores,
        "after_scores": after_scores,
    }
    return chosen, stats


def shard_index(task_id: str, num_shards: int) -> int:
    return zlib.crc32(task_id.encode("utf-8")) % num_shards


def shard_jsonl_by_task(
    input_path: Path,
    shard_dir: Path,
    prefix: str,
    num_shards: int,
) -> int:
    shard_dir.mkdir(parents=True, exist_ok=True)
    handles = {}
    count = 0
    try:
        for row in iter_jsonl(input_path):
            task_id = str(row.get("parent_task_id", ""))
            idx = shard_index(task_id, num_shards)
            shard_path = shard_dir / f"{prefix}_{idx:03d}.jsonl"
            if idx not in handles:
                handles[idx] = shard_path.open("w", encoding="utf-8")
            handles[idx].write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    finally:
        for fh in handles.values():
            fh.close()
    return count


def load_shard_grouped_by_task(path: Path, normalize_step4: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return grouped
    for row in iter_jsonl(path):
        if normalize_step4:
            row = normalize_step4_row(row)
        task_id = str(row.get("parent_task_id", ""))
        grouped[task_id].append(row)
    return grouped


def run_step5_sharded(
    step3_path: Path,
    step4_path: Path,
    output_path: Path,
    top_k: int,
    density_threshold: float,
    min_len: int,
    max_len: int,
    min_contact_residues_6a: int,
    min_rbsa_raw: float,
    random_seed: int,
    abs_iou_threshold: float,
    abs_peptide_anchor_tol: int,
    abs_receptor_anchor_tol: int,
    abs_length_diff_tol: int,
) -> Dict[str, Any]:
    t0 = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path.parent / "_step5_shards"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    num_shards = 256

    step3_input_count = shard_jsonl_by_task(step3_path, temp_dir, "step3", num_shards)
    step4_input_count = shard_jsonl_by_task(step4_path, temp_dir, "step4", num_shards)
    joined_candidate_count = 0
    missing_in_step4 = 0
    missing_in_step3 = 0
    num_parent_tasks = 0
    total_candidates_before_step5 = 0
    total_abs_redundant_dropped = 0
    total_topk_dropped = 0
    tasks_truncated_by_topk = 0
    total_eligible_after_dedup = 0
    total_sampled_from_eligible = 0
    total_filled_by_fallback = 0
    tasks_with_no_eligible_pool = 0
    tasks_with_sampling_active = 0
    total_candidates_after_step5 = 0
    rbsa_present_count = 0
    pocket_num_residues_present_count = 0
    final_both_cap_count = 0
    final_length_bucket = {"8-10": 0, "11-15": 0, "16-20": 0, "other": 0}
    before_scores: List[float] = []
    after_scores: List[float] = []

    with output_path.open("w", encoding="utf-8") as fout:
        for shard_idx in range(num_shards):
            step3_shard = temp_dir / f"step3_{shard_idx:03d}.jsonl"
            step4_shard = temp_dir / f"step4_{shard_idx:03d}.jsonl"
            step3_groups = load_shard_grouped_by_task(step3_shard, normalize_step4=False)
            step4_groups = load_shard_grouped_by_task(step4_shard, normalize_step4=True)

            all_task_ids = sorted(set(step3_groups.keys()) | set(step4_groups.keys()))
            for task_id in all_task_ids:
                step3_rows = step3_groups.get(task_id, [])
                step4_task_rows = step4_groups.get(task_id, [])
                if not step3_rows:
                    missing_in_step3 += len(step4_task_rows)
                    continue

                num_parent_tasks += 1
                joined_rows, task_missing_in_step4, task_missing_in_step3 = join_task_step3_step4(step3_rows, step4_task_rows)
                joined_candidate_count += len(joined_rows)
                missing_in_step4 += task_missing_in_step4
                missing_in_step3 += task_missing_in_step3
                rbsa_present_count += sum(1 for x in joined_rows if x.get("rBSA_proxy") is not None)
                pocket_num_residues_present_count += sum(1 for x in joined_rows if x.get("pocket_num_residues") is not None)

                chosen_rows, task_stats = process_task_rows(
                    parent_task_id=task_id,
                    rows=joined_rows,
                    top_k=top_k,
                    density_threshold=density_threshold,
                    min_len=min_len,
                    max_len=max_len,
                    min_contact_residues_6a=min_contact_residues_6a,
                    min_rbsa_raw=min_rbsa_raw,
                    random_seed=random_seed,
                    abs_iou_threshold=abs_iou_threshold,
                    abs_peptide_anchor_tol=abs_peptide_anchor_tol,
                    abs_receptor_anchor_tol=abs_receptor_anchor_tol,
                    abs_length_diff_tol=abs_length_diff_tol,
                )

                total_candidates_before_step5 += task_stats["before_count"]
                total_abs_redundant_dropped += task_stats["abs_redundant_dropped"]
                total_topk_dropped += task_stats["topk_dropped"]
                tasks_truncated_by_topk += task_stats["truncated_by_topk"]
                total_eligible_after_dedup += task_stats["eligible_count"]
                total_sampled_from_eligible += task_stats["sampled_count"]
                total_filled_by_fallback += task_stats["filled_count"]
                tasks_with_no_eligible_pool += task_stats["no_eligible_pool"]
                tasks_with_sampling_active += task_stats["sampling_active"]
                total_candidates_after_step5 += task_stats["final_count"]
                before_scores.extend(task_stats["before_scores"])
                after_scores.extend(task_stats["after_scores"])

                for row in chosen_rows:
                    final_length_bucket[bucket_length(safe_int(row.get("peptide_length"), 0))] += 1
                    hit_left = bool(row.get("hit_left_growth_cap", False))
                    hit_right = bool(row.get("hit_right_growth_cap", False))
                    if hit_left and hit_right:
                        final_both_cap_count += 1
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            for p in (step3_shard, step4_shard):
                if p.exists():
                    os.remove(p)

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    summary = {
        "step5_version": "method_a_probabilistic_sampling_v11_sharded_task_merge",
        "top_k": top_k,
        "elapsed_seconds": time.time() - t0,
        "step3_input_count": step3_input_count,
        "step4_input_count": step4_input_count,
        "joined_candidate_count": joined_candidate_count,
        "missing_in_step4": missing_in_step4,
        "missing_in_step3": missing_in_step3,
        "num_parent_tasks": num_parent_tasks,
        "total_candidates_before_step5": total_candidates_before_step5,
        "total_abs_redundant_dropped": total_abs_redundant_dropped,
        "total_topk_dropped": total_topk_dropped,
        "tasks_truncated_by_topk": tasks_truncated_by_topk,
        "total_candidates_after_step5": total_candidates_after_step5,
        "avg_candidates_per_task_before": (
            total_candidates_before_step5 / num_parent_tasks if num_parent_tasks else 0.0
        ),
        "avg_candidates_per_task_after": (
            total_candidates_after_step5 / num_parent_tasks if num_parent_tasks else 0.0
        ),
        "mean_score_final_before": mean_or_zero(before_scores),
        "mean_score_final_after": mean_or_zero(after_scores),
        "rbsa_present_count": rbsa_present_count,
        "rbsa_present_ratio": rbsa_present_count / joined_candidate_count if joined_candidate_count else 0.0,
        "pocket_num_residues_present_count": pocket_num_residues_present_count,
        "pocket_num_residues_present_ratio": (
            pocket_num_residues_present_count / joined_candidate_count if joined_candidate_count else 0.0
        ),
        "final_length_bucket": final_length_bucket,
        "final_both_cap_count": final_both_cap_count,
        "final_both_cap_ratio": final_both_cap_count / total_candidates_after_step5 if total_candidates_after_step5 else 0.0,
        "density_threshold": density_threshold,
        "sampling_min_len": min_len,
        "sampling_max_len": max_len,
        "min_contact_residues_6A": min_contact_residues_6a,
        "min_rbsa_raw": min_rbsa_raw,
        "sampling_power": None,
        "length_bonus_strength": None,
        "random_seed": random_seed,
        "abs_iou_threshold": abs_iou_threshold,
        "abs_peptide_anchor_tol": abs_peptide_anchor_tol,
        "abs_receptor_anchor_tol": abs_receptor_anchor_tol,
        "abs_length_diff_tol": abs_length_diff_tol,
        "total_eligible_after_dedup": total_eligible_after_dedup,
        "total_sampled_from_eligible": total_sampled_from_eligible,
        "total_filled_by_fallback": total_filled_by_fallback,
        "tasks_with_no_eligible_pool": tasks_with_no_eligible_pool,
        "tasks_with_sampling_active": tasks_with_sampling_active,
        "score_formula": "0.60*rBSA_raw + 0.30*contact_coverage_6A - 0.15*covalent_bias_risk (kept only for downstream compatibility)",
        "note": (
            "Hash-sharded task-wise Step3/Step4 merge to avoid full-file memory blowup and avoid relying on input order; absolute dedup uses physical-priority ordering; "
            "eligible pool requires peptide length, contact_coverage_6A, n_contact_residues_6A, and rBSA_raw thresholds; "
            "probabilistic sampling uses only contact_coverage_6A; fallback uses lexicographic physical-priority ranking."
        ),
    }
    return summary


def run_step5(
    step3_rows: List[Dict[str, Any]],
    step4_rows: List[Dict[str, Any]],
    top_k: int,
    density_threshold: float,
    min_len: int,
    max_len: int,
    min_contact_residues_6a: int,
    min_rbsa_raw: float,
    random_seed: int,
    abs_iou_threshold: float,
    abs_peptide_anchor_tol: int,
    abs_receptor_anchor_tol: int,
    abs_length_diff_tol: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    t0 = time.time()

    joined_rows, missing_in_step4, missing_in_step3 = join_step3_step4(step3_rows, step4_rows)

    task_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        parent_task_id = row.get("parent_task_id")
        if parent_task_id is not None:
            task_groups[str(parent_task_id)].append(row)

    final_rows: List[Dict[str, Any]] = []

    total_candidates_before_step5 = 0
    total_abs_redundant_dropped = 0
    total_topk_dropped = 0
    tasks_truncated_by_topk = 0
    total_eligible_after_dedup = 0
    total_sampled_from_eligible = 0
    total_filled_by_fallback = 0
    tasks_with_no_eligible_pool = 0
    tasks_with_sampling_active = 0

    before_scores: List[float] = []
    after_scores: List[float] = []

    for parent_task_id, rows in task_groups.items():
        total_candidates_before_step5 += len(rows)
        before_scores.extend(safe_float(x.get("score_final_before"), 0.0) for x in rows)

        deduped, n_redundant = dedup_within_task(
            rows=rows,
            iou_threshold=abs_iou_threshold,
            peptide_anchor_tol=abs_peptide_anchor_tol,
            receptor_anchor_tol=abs_receptor_anchor_tol,
            length_diff_tol=abs_length_diff_tol,
        )
        total_abs_redundant_dropped += n_redundant

        eligible_pool: List[Dict[str, Any]] = []
        fallback_pool: List[Dict[str, Any]] = []

        for row in deduped:
            peptide_len = safe_int(row.get("peptide_length"), 0)
            cov = safe_float(row.get("contact_coverage_6A"), 0.0)
            n_contact_res_6a = safe_int(row.get("n_contact_residues_6A"), 0)
            rbsa_raw = safe_float(row.get("rBSA_raw"), 0.0)
            if (
                min_len <= peptide_len <= max_len
                and cov >= density_threshold
                and n_contact_res_6a >= min_contact_residues_6a
                and rbsa_raw >= min_rbsa_raw
            ):
                eligible_pool.append(row)
            else:
                fallback_pool.append(row)

        total_eligible_after_dedup += len(eligible_pool)

        chosen: List[Dict[str, Any]] = []
        chosen_ids = set()
        rng = random.Random(stable_task_seed(parent_task_id, random_seed))

        if eligible_pool:
            tasks_with_sampling_active += 1
            eligible_scored = attach_sampling_weight(
                rows=eligible_pool,
                density_threshold=density_threshold,
                min_weight=1e-3,
            )
            sampled = weighted_sample_without_replacement(
                rows=eligible_scored,
                k=min(top_k, len(eligible_scored)),
                rng=rng,
            )
            total_sampled_from_eligible += len(sampled)
            for row in sampled:
                row = dict(row)
                row["step5_selected_by"] = "eligible_sampling"
                row["step5_selected_by_sampling"] = True
                row["step5_selection_mode"] = "sampling"
                row["step5_sampling_weight"] = safe_float(row.get("_sampling_weight"), 0.0)
                chosen.append(row)
                chosen_ids.add(row.get("candidate_id"))
        else:
            tasks_with_no_eligible_pool += 1

        if len(chosen) < top_k:
            need = top_k - len(chosen)
            fallback_candidates = [x for x in (fallback_pool + eligible_pool) if x.get("candidate_id") not in chosen_ids]
            fallback_candidates = sorted(fallback_candidates, key=fallback_rank_key, reverse=True)
            filled_rows = fallback_candidates[:need]

            total_filled_by_fallback += len(filled_rows)
            for row in filled_rows:
                row = dict(row)
                row["_density_margin"] = safe_float(row.get("_density_margin"), 0.0)
                row["_sampling_basis"] = str(row.get("_sampling_basis", "contact_coverage_6A"))
                row["_sampling_weight"] = safe_float(row.get("_sampling_weight"), 0.0)
                row["step5_selected_by"] = "fallback_score"
                row["step5_selected_by_sampling"] = False
                row["step5_selection_mode"] = "fallback_score"
                row["step5_sampling_weight"] = None
                chosen.append(row)
                chosen_ids.add(row.get("candidate_id"))

        chosen = sorted(
            chosen,
            key=lambda x: (
                0 if bool(x.get("step5_selected_by_sampling", False)) else 1,
                -safe_float(x.get("contact_coverage_6A"), 0.0),
                -safe_int(x.get("n_contact_residues_6A"), 0),
                -safe_float(x.get("rBSA_raw"), 0.0),
                safe_float(x.get("covalent_bias_risk"), 0.0),
                str(x.get("candidate_id", "")),
            ),
        )

        if len(deduped) > top_k:
            tasks_truncated_by_topk += 1
        total_topk_dropped += max(0, len(deduped) - len(chosen))

        for rank, row in enumerate(chosen, start=1):
            row = dict(row)
            row["score_final_after"] = safe_float(row.get("score_final_before"), 0.0)
            row["score_final"] = row["score_final_after"]
            row["step5_parent_task_rank"] = rank
            row["step5_rank_within_task"] = rank
            row["step5_top_k"] = top_k
            row["step5_task_candidate_count_before"] = len(rows)
            row["step5_task_candidate_count_after_dedup"] = len(deduped)
            row["step5_task_candidate_count_eligible"] = len(eligible_pool)
            row["step5_task_candidate_count_final"] = len(chosen)
            row["step5_density_threshold"] = density_threshold
            row["step5_sampling_min_len"] = min_len
            row["step5_sampling_max_len"] = max_len
            row["step5_min_contact_residues_6A"] = min_contact_residues_6a
            row["step5_min_rbsa_raw"] = min_rbsa_raw
            row["step5_sampling_power"] = None
            row["step5_length_bonus_strength"] = None
            row["step5_abs_iou_threshold"] = abs_iou_threshold
            row["step5_abs_peptide_anchor_tol"] = abs_peptide_anchor_tol
            row["step5_abs_receptor_anchor_tol"] = abs_receptor_anchor_tol
            row["step5_abs_length_diff_tol"] = abs_length_diff_tol
            final_rows.append(row)
            after_scores.append(row["score_final_after"])

    final_rows = sorted(
        final_rows,
        key=lambda x: (
            str(x.get("parent_task_id", "")),
            safe_int(x.get("step5_parent_task_rank"), 0),
            str(x.get("candidate_id", "")),
        ),
    )

    rbsa_present_count = sum(1 for x in joined_rows if x.get("rBSA_proxy") is not None)
    pocket_num_residues_present_count = sum(1 for x in joined_rows if x.get("pocket_num_residues") is not None)

    final_length_bucket = {"8-10": 0, "11-15": 0, "16-20": 0, "other": 0}
    final_both_cap_count = 0

    for row in final_rows:
        final_length_bucket[bucket_length(safe_int(row.get("peptide_length"), 0))] += 1
        hit_left = bool(row.get("hit_left_growth_cap", False))
        hit_right = bool(row.get("hit_right_growth_cap", False))
        if hit_left and hit_right:
            final_both_cap_count += 1

    summary = {
        "step5_version": "method_a_probabilistic_sampling_v9_rule_based_eligible_pool",
        "top_k": top_k,
        "elapsed_seconds": time.time() - t0,
        "step3_input_count": len(step3_rows),
        "step4_input_count": len(step4_rows),
        "joined_candidate_count": len(joined_rows),
        "missing_in_step4": missing_in_step4,
        "missing_in_step3": missing_in_step3,
        "num_parent_tasks": len(task_groups),
        "total_candidates_before_step5": total_candidates_before_step5,
        "total_abs_redundant_dropped": total_abs_redundant_dropped,
        "total_topk_dropped": total_topk_dropped,
        "tasks_truncated_by_topk": tasks_truncated_by_topk,
        "total_candidates_after_step5": len(final_rows),
        "avg_candidates_per_task_before": (
            total_candidates_before_step5 / len(task_groups) if task_groups else 0.0
        ),
        "avg_candidates_per_task_after": (
            len(final_rows) / len(task_groups) if task_groups else 0.0
        ),
        "mean_score_final_before": mean_or_zero(before_scores),
        "mean_score_final_after": mean_or_zero(after_scores),
        "rbsa_present_count": rbsa_present_count,
        "rbsa_present_ratio": rbsa_present_count / len(joined_rows) if joined_rows else 0.0,
        "pocket_num_residues_present_count": pocket_num_residues_present_count,
        "pocket_num_residues_present_ratio": (
            pocket_num_residues_present_count / len(joined_rows) if joined_rows else 0.0
        ),
        "final_length_bucket": final_length_bucket,
        "final_both_cap_count": final_both_cap_count,
        "final_both_cap_ratio": final_both_cap_count / len(final_rows) if final_rows else 0.0,
        "density_threshold": density_threshold,
        "sampling_min_len": min_len,
        "sampling_max_len": max_len,
        "min_contact_residues_6A": min_contact_residues_6a,
        "min_rbsa_raw": min_rbsa_raw,
        "sampling_power": None,
        "length_bonus_strength": None,
        "random_seed": random_seed,
        "abs_iou_threshold": abs_iou_threshold,
        "abs_peptide_anchor_tol": abs_peptide_anchor_tol,
        "abs_receptor_anchor_tol": abs_receptor_anchor_tol,
        "abs_length_diff_tol": abs_length_diff_tol,
        "total_eligible_after_dedup": total_eligible_after_dedup,
        "total_sampled_from_eligible": total_sampled_from_eligible,
        "total_filled_by_fallback": total_filled_by_fallback,
        "tasks_with_no_eligible_pool": tasks_with_no_eligible_pool,
        "tasks_with_sampling_active": tasks_with_sampling_active,
        "score_formula": "0.60*rBSA_raw + 0.30*contact_coverage_6A - 0.15*covalent_bias_risk (kept only for downstream compatibility)",
        "note": (
            "Absolute dedup uses physical-priority ordering; eligible pool requires peptide length, contact_coverage_6A, "
            "n_contact_residues_6A, and rBSA_raw thresholds; probabilistic sampling uses only contact_coverage_6A; "
            "fallback uses lexicographic physical-priority ranking instead of composite weighted scoring."
        ),
    }

    return final_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step-5: task-internal absolute dedup + probabilistic dice sampling"
    )
    parser.add_argument("--step3_jsonl", type=str, required=True)
    parser.add_argument("--step4_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--density_threshold", type=float, default=0.50)
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--min_contact_residues_6a", type=int, default=4)
    parser.add_argument("--min_rbsa_raw", type=float, default=0.05)
    parser.add_argument("--sampling_power", type=float, default=1.0)
    parser.add_argument("--length_bonus_strength", type=float, default=0.25)
    parser.add_argument("--random_seed", type=int, default=20260406)
    parser.add_argument("--abs_iou_threshold", type=float, default=0.80)
    parser.add_argument("--abs_peptide_anchor_tol", type=int, default=2)
    parser.add_argument("--abs_receptor_anchor_tol", type=int, default=2)
    parser.add_argument("--abs_length_diff_tol", type=int, default=2)
    args = parser.parse_args()

    step3_path = Path(args.step3_jsonl)
    step4_path = Path(args.step4_jsonl)
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)

    print("=" * 80, flush=True)
    print("[START] Step-5 Method A probabilistic custom", flush=True)
    print(f"[START] step3_jsonl = {step3_path}", flush=True)
    print(f"[START] step4_jsonl = {step4_path}", flush=True)
    print(f"[START] output_jsonl = {output_path}", flush=True)
    print(f"[START] summary_json = {summary_path}", flush=True)
    print("=" * 80, flush=True)

    print("[INFO] Running streaming task-wise Step-5 merge to reduce memory usage", flush=True)
    print("=" * 80, flush=True)

    summary = run_step5_sharded(
        step3_path=step3_path,
        step4_path=step4_path,
        output_path=output_path,
        top_k=args.top_k,
        density_threshold=args.density_threshold,
        min_len=args.min_len,
        max_len=args.max_len,
        min_contact_residues_6a=args.min_contact_residues_6a,
        min_rbsa_raw=args.min_rbsa_raw,
        random_seed=args.random_seed,
        abs_iou_threshold=args.abs_iou_threshold,
        abs_peptide_anchor_tol=args.abs_peptide_anchor_tol,
        abs_receptor_anchor_tol=args.abs_receptor_anchor_tol,
        abs_length_diff_tol=args.abs_length_diff_tol,
    )

    write_json(summary_path, summary)

    print("[DONE] Step-5 finished.", flush=True)
    print(f"[DONE] Before Step-5 : {summary['total_candidates_before_step5']}", flush=True)
    print(f"[DONE] After Step-5  : {summary['total_candidates_after_step5']}", flush=True)
    print(f"[DONE] Abs dropped   : {summary['total_abs_redundant_dropped']}", flush=True)
    print(f"[DONE] Output JSONL  : {output_path}", flush=True)
    print(f"[DONE] Summary JSON  : {summary_path}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()


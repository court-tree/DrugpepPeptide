from __future__ import annotations

# Step-5 top-k sampling: absolute dedup + deterministic score-ranked top-k selection.

import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Canonical fields required after Step-3 + Step-4 merge.
REQUIRED_FIELDS = [
    "candidate_id",
    "parent_task_id",
    "final_left_index",
    "final_right_index",
    "anchor_peptide_res_index",
    "anchor_receptor_res_index",
    "peptide_length",
    "rBSA_raw",
    "contact_coverage_6A",
    "n_contact_atoms_6A",
    "covalent_bias_risk",
]


def validate_required_fields(x: Dict[str, Any]) -> None:
    for k in REQUIRED_FIELDS:
        if k not in x:
            raise KeyError(f"Missing required field after canonicalization: '{k}'")


# ---------------------------------------------------------------------------
# Validation / compatibility
# ---------------------------------------------------------------------------
def canonicalize_step4_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept both old/new Step-4 schemas and normalize to the SOP-oriented names.

    Important for downstream Step-6:
      - always emit rBSA_proxy
      - always emit pocket_num_residues when recoverable
      - keep original step4 fields when present
    """
    x = dict(row)

    # rBSA
    if "rBSA_raw" not in x:
        if "rBSA_proxy" in x:
            x["rBSA_raw"] = x["rBSA_proxy"]
        elif "rBSA" in x:
            x["rBSA_raw"] = x["rBSA"]
        else:
            x["rBSA_raw"] = None

    # Step-6 compatibility alias
    if "rBSA_proxy" not in x:
        x["rBSA_proxy"] = x.get("rBSA_raw")

    # coverage
    if "contact_coverage_6A" not in x:
        if "contact_coverage_5A" in x:
            x["contact_coverage_6A"] = x["contact_coverage_5A"]
        elif "contact_coverage" in x:
            x["contact_coverage_6A"] = x["contact_coverage"]

    # contact atoms
    if "n_contact_atoms_6A" not in x:
        if "n_contact_atoms_5A" in x:
            x["n_contact_atoms_6A"] = x["n_contact_atoms_5A"]
        elif "n_heavy_contacts" in x:
            x["n_contact_atoms_6A"] = x["n_heavy_contacts"]

    # risk
    if "covalent_bias_risk" not in x:
        x["covalent_bias_risk"] = 0.0

    # pocket size aliases for downstream Step-6/7
    if "pocket_num_residues" not in x:
        if "pocket_size_6A" in x:
            x["pocket_num_residues"] = x["pocket_size_6A"]
        elif "n_contact_residues_6A" in x:
            x["pocket_num_residues"] = x["n_contact_residues_6A"]

    # legacy alias used by current Step-6 code path
    if "n_contact_residues_step4" not in x and "n_contact_residues_6A" in x:
        x["n_contact_residues_step4"] = x["n_contact_residues_6A"]

    validate_required_fields(x)
    return x


# ---------------------------------------------------------------------------
# Interval utilities
# ---------------------------------------------------------------------------
def interval_iou(a_left: int, a_right: int, b_left: int, b_right: int) -> float:
    inter_left = max(a_left, b_left)
    inter_right = min(a_right, b_right)
    if inter_right < inter_left:
        return 0.0

    inter = inter_right - inter_left + 1
    union = (a_right - a_left + 1) + (b_right - b_left + 1) - inter
    return inter / union if union > 0 else 0.0



def interval_contains(a_left: int, a_right: int, b_left: int, b_right: int) -> bool:
    return a_left <= b_left and a_right >= b_right


# ---------------------------------------------------------------------------
# Scoring / ranking / sampling
# ---------------------------------------------------------------------------
def compute_score_final(x: Dict[str, Any]) -> Tuple[float, bool]:
    rbsa_raw = x.get("rBSA_raw", None)
    has_rbsa = rbsa_raw is not None
    rbsa = float(rbsa_raw) if has_rbsa else 0.0
    coverage = float(x.get("contact_coverage_6A", 0.0) or 0.0)
    risk = float(x.get("covalent_bias_risk", 0.0) or 0.0)

    score = 0.60 * rbsa + 0.30 * coverage - 0.15 * risk
    return float(score), has_rbsa



def redundancy_priority_key(x: Dict[str, Any]) -> Tuple[float, float, float, float, int]:
    coverage = float(x.get("contact_coverage_6A", 0.0) or 0.0)
    risk = float(x.get("covalent_bias_risk", 0.0) or 0.0)
    rbsa = float(x.get("rBSA_raw", 0.0) or 0.0)
    score = float(x.get("score_final", 0.0) or 0.0)
    length = int(x.get("peptide_length", 0) or 0)
    return (coverage, -risk, rbsa, score, length)



def is_absolute_redundant(
    keep: Dict[str, Any],
    cand: Dict[str, Any],
    iou_threshold: float = 0.80,
    peptide_anchor_tol: int = 2,
    receptor_anchor_tol: int = 2,
    length_diff_tol: int = 2,
) -> bool:
    a_left, a_right = keep["final_left_index"], keep["final_right_index"]
    b_left, b_right = cand["final_left_index"], cand["final_right_index"]

    contains = interval_contains(a_left, a_right, b_left, b_right) or interval_contains(b_left, b_right, a_left, a_right)
    iou = interval_iou(a_left, a_right, b_left, b_right)

    peptide_anchor_close = abs(int(keep["anchor_peptide_res_index"]) - int(cand["anchor_peptide_res_index"])) <= peptide_anchor_tol
    receptor_anchor_close = abs(int(keep["anchor_receptor_res_index"]) - int(cand["anchor_receptor_res_index"])) <= receptor_anchor_tol
    length_close = abs(int(keep["peptide_length"]) - int(cand["peptide_length"])) <= length_diff_tol

    return (contains or iou >= iou_threshold) and peptide_anchor_close and receptor_anchor_close and length_close



def is_multi_mode_distinct(
    keep: Dict[str, Any],
    cand: Dict[str, Any],
    length_diff_threshold: int = 5,
    peptide_anchor_diff_threshold: int = 4,
    receptor_anchor_diff_threshold: int = 4,
    iou_upper_for_distinct: float = 0.60,
) -> bool:
    a_left, a_right = keep["final_left_index"], keep["final_right_index"]
    b_left, b_right = cand["final_left_index"], cand["final_right_index"]

    iou = interval_iou(a_left, a_right, b_left, b_right)
    length_diff = abs(int(keep["peptide_length"]) - int(cand["peptide_length"]))
    peptide_anchor_diff = abs(int(keep["anchor_peptide_res_index"]) - int(cand["anchor_peptide_res_index"]))
    receptor_anchor_diff = abs(int(keep["anchor_receptor_res_index"]) - int(cand["anchor_receptor_res_index"]))

    if length_diff >= length_diff_threshold:
        return True
    if peptide_anchor_diff >= peptide_anchor_diff_threshold:
        return True
    if receptor_anchor_diff >= receptor_anchor_diff_threshold:
        return True
    if iou <= iou_upper_for_distinct:
        return True
    return False



def compute_sampling_weight(
    row: Dict[str, Any],
    density_threshold: float,
    sampling_power: float,
    length_bonus_strength: float,
    min_len: int,
    max_len: int,
) -> float:
    coverage = float(row.get("contact_coverage_6A", 0.0) or 0.0)
    peptide_len = int(row["peptide_length"])

    base = max(coverage - density_threshold, 1e-6)
    density_term = base ** sampling_power

    if max_len > min_len:
        length_norm = (peptide_len - min_len) / float(max_len - min_len)
        length_norm = max(0.0, min(1.0, length_norm))
    else:
        length_norm = 0.0

    length_bonus = 1.0 + length_bonus_strength * length_norm
    return float(density_term * length_bonus)



def weighted_sample_without_replacement(rows: List[Dict[str, Any]], k: int, rng: random.Random) -> List[Dict[str, Any]]:
    pool = list(rows)
    selected: List[Dict[str, Any]] = []

    while pool and len(selected) < k:
        weights = [max(float(x.get("_sampling_weight", 0.0)), 0.0) for x in pool]
        total = sum(weights)

        if total <= 0:
            idx = rng.randrange(len(pool))
        else:
            r = rng.random() * total
            cumsum = 0.0
            idx = len(pool) - 1
            for i, w in enumerate(weights):
                cumsum += w
                if r <= cumsum:
                    idx = i
                    break

        selected.append(pool.pop(idx))

    return selected



def stable_task_seed(parent_task_id: str, base_seed: int) -> int:
    return int(base_seed + sum(ord(c) for c in str(parent_task_id)))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows



def bucket_length(L: int) -> str:
    if L <= 5:
        return "<=5"
    if L <= 10:
        return "6-10"
    if L <= 15:
        return "11-15"
    if L <= 20:
        return "16-20"
    if L <= 25:
        return "21-25"
    return ">25"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="PeptideCLIP Step-5 Method A v3 fixed: preserve Step-4 features for downstream Step-6"
    )
    parser.add_argument("--step3_jsonl", type=str, required=True)
    parser.add_argument("--step4_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)

    parser.add_argument("--density_threshold", type=float, default=0.50)
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--sampling_power", type=float, default=1.0)
    parser.add_argument("--length_bonus_strength", type=float, default=0.25)
    parser.add_argument("--random_seed", type=int, default=20260406)

    parser.add_argument("--abs_iou_threshold", type=float, default=0.80)
    parser.add_argument("--abs_peptide_anchor_tol", type=int, default=2)
    parser.add_argument("--abs_receptor_anchor_tol", type=int, default=2)
    parser.add_argument("--abs_length_diff_tol", type=int, default=2)

    parser.add_argument("--distinct_length_diff_threshold", type=int, default=5)
    parser.add_argument("--distinct_peptide_anchor_diff_threshold", type=int, default=4)
    parser.add_argument("--distinct_receptor_anchor_diff_threshold", type=int, default=4)
    parser.add_argument("--distinct_iou_upper_for_distinct", type=float, default=0.60)

    args = parser.parse_args()

    start_time = time.time()

    step3_jsonl = Path(args.step3_jsonl)
    step4_jsonl = Path(args.step4_jsonl)
    output_jsonl = Path(args.output_jsonl)
    summary_json = Path(args.summary_json)

    print("=" * 80, flush=True)
    print("[START] Step-5 Method A v3 fixed", flush=True)
    print(f"[START] step3_jsonl = {step3_jsonl}", flush=True)
    print(f"[START] step4_jsonl = {step4_jsonl}", flush=True)
    print(f"[START] output_jsonl = {output_jsonl}", flush=True)
    print(f"[START] summary_json = {summary_json}", flush=True)
    print("=" * 80, flush=True)

    step3_rows = load_jsonl(step3_jsonl)
    step4_rows = load_jsonl(step4_jsonl)

    print(f"[INFO] Loaded Step-3 candidates: {len(step3_rows)}", flush=True)
    print(f"[INFO] Loaded Step-4 features  : {len(step4_rows)}", flush=True)

    step3_by_id = {x["candidate_id"]: x for x in step3_rows}
    step4_by_id = {x["candidate_id"]: x for x in step4_rows}

    common_ids = sorted(set(step3_by_id.keys()) & set(step4_by_id.keys()))
    missing_in_step4 = len(step3_by_id) - len(common_ids)
    missing_in_step3 = len(step4_by_id) - len(common_ids)

    merged_rows: List[Dict[str, Any]] = []
    rbsa_present_count = 0
    pocket_num_residues_present_count = 0

    for cid in common_ids:
        s3 = step3_by_id[cid]
        s4 = canonicalize_step4_fields(step4_by_id[cid])

        merged = dict(s3)
        merged.update(s4)

        # Critical downstream aliases for Step-6 / Step-7
        merged["rBSA_proxy"] = merged.get("rBSA_proxy", merged.get("rBSA_raw"))
        if "pocket_num_residues" not in merged:
            if "pocket_size_6A" in merged:
                merged["pocket_num_residues"] = merged["pocket_size_6A"]
            elif "n_contact_residues_6A" in merged:
                merged["pocket_num_residues"] = merged["n_contact_residues_6A"]
        if "n_contact_residues_step4" not in merged and "n_contact_residues_6A" in merged:
            merged["n_contact_residues_step4"] = merged["n_contact_residues_6A"]

        score_final, has_rbsa = compute_score_final(merged)
        merged["score_final"] = score_final
        merged["step5_has_rbsa"] = has_rbsa
        if has_rbsa:
            rbsa_present_count += 1
        if merged.get("pocket_num_residues") is not None:
            pocket_num_residues_present_count += 1

        merged_rows.append(merged)

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in merged_rows:
        groups[row["parent_task_id"]].append(row)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    total_before = 0
    total_after = 0
    total_abs_redundant_dropped = 0
    total_topk_dropped = 0
    tasks_truncated_by_topk = 0

    per_task_before: List[int] = []
    per_task_after: List[int] = []

    final_length_bucket = Counter()
    final_both_cap_count = 0

    total_eligible_after_dedup = 0
    total_sampled_from_eligible = 0
    total_filled_by_fallback = 0
    tasks_with_no_eligible_pool = 0
    tasks_with_sampling_active = 0

    score_before_vals: List[float] = []
    score_after_vals: List[float] = []

    with output_jsonl.open("w", encoding="utf-8") as f_out:
        for parent_task_id, rows in groups.items():
            total_before += len(rows)
            per_task_before.append(len(rows))
            score_before_vals.extend(float(x["score_final"]) for x in rows)

            rows_sorted_for_dedup = sorted(rows, key=redundancy_priority_key, reverse=True)

            kept: List[Dict[str, Any]] = []
            for cand in rows_sorted_for_dedup:
                redundant = False
                for existing in kept:
                    if is_absolute_redundant(
                        existing,
                        cand,
                        iou_threshold=args.abs_iou_threshold,
                        peptide_anchor_tol=args.abs_peptide_anchor_tol,
                        receptor_anchor_tol=args.abs_receptor_anchor_tol,
                        length_diff_tol=args.abs_length_diff_tol,
                    ):
                        if not is_multi_mode_distinct(
                            existing,
                            cand,
                            length_diff_threshold=args.distinct_length_diff_threshold,
                            peptide_anchor_diff_threshold=args.distinct_peptide_anchor_diff_threshold,
                            receptor_anchor_diff_threshold=args.distinct_receptor_anchor_diff_threshold,
                            iou_upper_for_distinct=args.distinct_iou_upper_for_distinct,
                        ):
                            redundant = True
                            break

                if redundant:
                    total_abs_redundant_dropped += 1
                else:
                    kept.append(cand)

            eligible_pool: List[Dict[str, Any]] = []
            for cand in kept:
                peptide_len = int(cand["peptide_length"])
                coverage = float(cand.get("contact_coverage_6A", 0.0) or 0.0)
                if args.min_len <= peptide_len <= args.max_len and coverage >= args.density_threshold:
                    x = dict(cand)
                    x["_sampling_weight"] = compute_sampling_weight(
                        row=x,
                        density_threshold=args.density_threshold,
                        sampling_power=args.sampling_power,
                        length_bonus_strength=args.length_bonus_strength,
                        min_len=args.min_len,
                        max_len=args.max_len,
                    )
                    eligible_pool.append(x)

            total_eligible_after_dedup += len(eligible_pool)
            if len(eligible_pool) == 0:
                tasks_with_no_eligible_pool += 1

            topk_selected = sorted(
                eligible_pool,
                key=lambda x: (float(x["score_final"]), float(x.get("contact_coverage_6A", 0.0))),
                reverse=True,
            )[: min(args.top_k, len(eligible_pool))]
            if topk_selected:
                tasks_with_sampling_active += 1

            total_sampled_from_eligible += len(topk_selected)
            selected_ids = {x["candidate_id"] for x in topk_selected}

            remaining_for_fallback = [x for x in kept if x["candidate_id"] not in selected_ids]
            remaining_for_fallback = sorted(remaining_for_fallback, key=lambda x: float(x["score_final"]), reverse=True)

            need = max(0, args.top_k - len(topk_selected))
            filled = remaining_for_fallback[:need]
            total_filled_by_fallback += len(filled)

            final_kept = topk_selected + filled
            if len(kept) > args.top_k:
                tasks_truncated_by_topk += 1
            total_topk_dropped += max(0, len(kept) - len(final_kept))

            final_kept = sorted(
                final_kept,
                key=lambda x: (
                    0 if x["candidate_id"] in selected_ids else 1,
                    -float(x["score_final"]),
                ),
            )

            for rank, row in enumerate(final_kept, start=1):
                out_row = dict(row)
                out_row["step5_rank_within_task"] = rank
                out_row["step5_task_candidate_count_before"] = len(rows)
                out_row["step5_task_candidate_count_after_dedup"] = len(kept)
                out_row["step5_task_candidate_count_eligible"] = len(eligible_pool)
                out_row["step5_task_candidate_count_final"] = len(final_kept)

                out_row["step5_density_threshold"] = args.density_threshold
                out_row["step5_sampling_min_len"] = args.min_len
                out_row["step5_sampling_max_len"] = args.max_len
                out_row["step5_selected_by_sampling"] = False
                out_row["step5_selection_mode"] = "topk" if out_row["candidate_id"] in selected_ids else "fallback"
                out_row["step5_sampling_weight"] = None
                out_row.pop("_sampling_weight", None)

                # Explicit downstream-compatible aliases
                out_row["rBSA_proxy"] = out_row.get("rBSA_proxy", out_row.get("rBSA_raw"))
                if "pocket_num_residues" not in out_row or out_row.get("pocket_num_residues") is None:
                    if "pocket_size_6A" in out_row:
                        out_row["pocket_num_residues"] = out_row["pocket_size_6A"]
                    elif "n_contact_residues_6A" in out_row:
                        out_row["pocket_num_residues"] = out_row["n_contact_residues_6A"]
                if "n_contact_residues_step4" not in out_row and "n_contact_residues_6A" in out_row:
                    out_row["n_contact_residues_step4"] = out_row["n_contact_residues_6A"]

                f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                score_after_vals.append(float(out_row["score_final"]))

                L = int(out_row["peptide_length"])
                final_length_bucket[bucket_length(L)] += 1
                if out_row.get("both_cap"):
                    final_both_cap_count += 1

            total_after += len(final_kept)
            per_task_after.append(len(final_kept))

    elapsed = time.time() - start_time
    mean_score_before = sum(score_before_vals) / len(score_before_vals) if score_before_vals else 0.0
    mean_score_after = sum(score_after_vals) / len(score_after_vals) if score_after_vals else 0.0

    summary = {
        "step5_version": "method_a_topk_sampling_v1",
        "top_k": args.top_k,
        "elapsed_seconds": elapsed,
        "step3_input_count": len(step3_rows),
        "step4_input_count": len(step4_rows),
        "joined_candidate_count": len(merged_rows),
        "missing_in_step4": missing_in_step4,
        "missing_in_step3": missing_in_step3,
        "num_parent_tasks": len(groups),
        "total_candidates_before_step5": total_before,
        "total_abs_redundant_dropped": total_abs_redundant_dropped,
        "total_topk_dropped": total_topk_dropped,
        "tasks_truncated_by_topk": tasks_truncated_by_topk,
        "total_candidates_after_step5": total_after,
        "avg_candidates_per_task_before": (sum(per_task_before) / len(per_task_before)) if per_task_before else 0.0,
        "avg_candidates_per_task_after": (sum(per_task_after) / len(per_task_after)) if per_task_after else 0.0,
        "mean_score_final_before": mean_score_before,
        "mean_score_final_after": mean_score_after,
        "rbsa_present_count": rbsa_present_count,
        "rbsa_present_ratio": (rbsa_present_count / len(merged_rows)) if merged_rows else 0.0,
        "pocket_num_residues_present_count": pocket_num_residues_present_count,
        "pocket_num_residues_present_ratio": (pocket_num_residues_present_count / len(merged_rows)) if merged_rows else 0.0,
        "final_length_bucket": dict(final_length_bucket),
        "final_both_cap_count": final_both_cap_count,
        "final_both_cap_ratio": (final_both_cap_count / total_after) if total_after > 0 else 0.0,
        "density_threshold": args.density_threshold,
        "sampling_min_len": args.min_len,
        "sampling_max_len": args.max_len,
        "sampling_power": args.sampling_power,
        "length_bonus_strength": args.length_bonus_strength,
        "random_seed": args.random_seed,
        "total_eligible_after_dedup": total_eligible_after_dedup,
        "total_sampled_from_eligible": total_sampled_from_eligible,
        "total_filled_by_fallback": total_filled_by_fallback,
        "tasks_with_no_eligible_pool": tasks_with_no_eligible_pool,
        "tasks_with_sampling_active": tasks_with_sampling_active,
        "score_formula": "0.60*rBSA_raw + 0.30*contact_coverage_6A - 0.15*covalent_bias_risk",
        "note": "Deterministic top-k baseline after absolute dedup; preserves Step-6-compatible aliases.",
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80, flush=True)
    print("[DONE] Step-5 Method A v3 fixed finished.", flush=True)
    print(f"[DONE] Before Step-5 : {total_before}", flush=True)
    print(f"[DONE] After Step-5  : {total_after}", flush=True)
    print(f"[DONE] Abs dropped   : {total_abs_redundant_dropped}", flush=True)
    print(f"[DONE] Output JSONL  : {output_jsonl}", flush=True)
    print(f"[DONE] Summary JSON  : {summary_json}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()


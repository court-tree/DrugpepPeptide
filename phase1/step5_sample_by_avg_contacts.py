from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common import write_json


def compute_sampling_weight(row: Dict[str, Any]) -> Dict[str, float]:
    avg_contact_count = max(1e-6, float(row.get("avg_contact_count", 0.0)))
    return {
        "raw_avg_contact_count": avg_contact_count,
        "sampling_weight": avg_contact_count,
    }


def select_weighted_candidates(
    pool: List[Dict[str, Any]],
    max_keep_per_task: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    chosen: List[Dict[str, Any]] = []
    working_pool = list(pool)

    while working_pool and len(chosen) < max_keep_per_task:
        working_weights = [max(1e-6, float(x["sampling_weight"])) for x in working_pool]
        picked = rng.choices(working_pool, weights=working_weights, k=1)[0]
        chosen.append(picked)
        working_pool = [x for x in working_pool if x["candidate_id"] != picked["candidate_id"]]

    return chosen


LengthBucket = Tuple[str, int, int]


DEFAULT_LENGTH_BUCKETS: List[LengthBucket] = [
    ("short_8_10", 8, 10),
    ("mid_11_14", 11, 14),
    ("long_15_20", 15, 20),
]


def length_bucket_name(peptide_length: int, buckets: Sequence[LengthBucket]) -> Optional[str]:
    for name, min_len, max_len in buckets:
        if min_len <= peptide_length <= max_len:
            return name
    return None


def weighted_pick(pool: List[Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    weights = [max(1e-6, float(x["sampling_weight"])) for x in pool]
    return rng.choices(pool, weights=weights, k=1)[0]


def candidate_interval(row: Dict[str, Any]) -> Tuple[int, int]:
    return int(row["final_left_index"]), int(row["final_right_index"])


def interval_overlap_stats(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[float, float]:
    a_l, a_r = candidate_interval(a)
    b_l, b_r = candidate_interval(b)
    inter = max(0, min(a_r, b_r) - max(a_l, b_l) + 1)
    if inter <= 0:
        return 0.0, 0.0
    a_len = a_r - a_l + 1
    b_len = b_r - b_l + 1
    union = a_len + b_len - inter
    jaccard = inter / union if union > 0 else 0.0
    short_containment = inter / min(a_len, b_len) if min(a_len, b_len) > 0 else 0.0
    return float(jaccard), float(short_containment)


def max_overlap_to_previous(row: Dict[str, Any], chosen: Sequence[Dict[str, Any]]) -> Tuple[float, float]:
    max_jaccard = 0.0
    max_short_containment = 0.0
    for prev in chosen:
        jaccard, short_containment = interval_overlap_stats(row, prev)
        max_jaccard = max(max_jaccard, jaccard)
        max_short_containment = max(max_short_containment, short_containment)
    return max_jaccard, max_short_containment


def annotate_overlap(
    row: Dict[str, Any],
    chosen: Sequence[Dict[str, Any]],
    decision: str,
) -> None:
    max_jaccard, max_short_containment = max_overlap_to_previous(row, chosen)
    row["max_jaccard_to_previous"] = max_jaccard
    row["max_short_containment_to_previous"] = max_short_containment
    row["step5_overlap_decision"] = decision


def overlap_reject_reason(
    row: Dict[str, Any],
    chosen: Sequence[Dict[str, Any]],
    jaccard_threshold: float,
    short_containment_threshold: float,
) -> Optional[str]:
    max_jaccard, max_short_containment = max_overlap_to_previous(row, chosen)
    row["max_jaccard_to_previous"] = max_jaccard
    row["max_short_containment_to_previous"] = max_short_containment
    if max_jaccard > jaccard_threshold:
        return "rejected_jaccard"
    if max_short_containment > short_containment_threshold:
        return "rejected_short_containment"
    return None


def weighted_pick_with_overlap(
    pool: List[Dict[str, Any]],
    chosen: Sequence[Dict[str, Any]],
    rng: random.Random,
    jaccard_threshold: float,
    short_containment_threshold: float,
) -> Optional[Dict[str, Any]]:
    working_pool = list(pool)
    while working_pool:
        picked = weighted_pick(working_pool, rng)
        reject_reason = overlap_reject_reason(
            picked,
            chosen,
            jaccard_threshold=jaccard_threshold,
            short_containment_threshold=short_containment_threshold,
        )
        if reject_reason is None:
            picked["step5_overlap_decision"] = "accepted"
            return picked
        working_pool = [x for x in working_pool if x["candidate_id"] != picked["candidate_id"]]
    return None


def force_best_candidate(pool: List[Dict[str, Any]], chosen: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not pool:
        return None
    picked = max(pool, key=lambda x: float(x["sampling_weight"]))
    annotate_overlap(picked, chosen, "forced_bucket_keep")
    return picked


def select_with_length_bucket_retention(
    pool: List[Dict[str, Any]],
    max_keep_per_task: int,
    rng: random.Random,
    buckets: Sequence[LengthBucket] = DEFAULT_LENGTH_BUCKETS,
    relaxed_bucket_jaccard: float = 0.85,
    strict_jaccard: float = 0.70,
    short_containment_threshold: float = 0.80,
) -> List[Dict[str, Any]]:
    chosen: List[Dict[str, Any]] = []
    chosen_ids = set()

    # First pass: retain one representative from every available length band.
    # This prevents high-contact short windows from consuming all per-task slots.
    for bucket_name, _min_len, _max_len in buckets:
        if len(chosen) >= max_keep_per_task:
            break
        bucket_pool = [
            x for x in pool
            if x["candidate_id"] not in chosen_ids
            and x.get("length_bucket") == bucket_name
        ]
        if not bucket_pool:
            continue
        picked = weighted_pick_with_overlap(
            bucket_pool,
            chosen,
            rng,
            jaccard_threshold=relaxed_bucket_jaccard,
            short_containment_threshold=short_containment_threshold,
        )
        if picked is None:
            picked = force_best_candidate(bucket_pool, chosen)
            if picked is None:
                continue
            picked["step5_selection_stage"] = "bucket_forced_keep"
        else:
            picked["step5_selection_stage"] = "length_bucket_retention"
        chosen.append(picked)
        chosen_ids.add(picked["candidate_id"])

    # Second pass: fill remaining slots by the original avg-contact weighting.
    while len(chosen) < max_keep_per_task:
        remaining = [x for x in pool if x["candidate_id"] not in chosen_ids]
        if not remaining:
            break
        picked = weighted_pick_with_overlap(
            remaining,
            chosen,
            rng,
            jaccard_threshold=strict_jaccard,
            short_containment_threshold=short_containment_threshold,
        )
        if picked is None:
            break
        picked["step5_selection_stage"] = "avg_contact_backfill"
        chosen.append(picked)
        chosen_ids.add(picked["candidate_id"])

    return chosen


def select_with_length_bucket_cap(
    pool: List[Dict[str, Any]],
    max_keep_per_task: int,
    max_per_bucket: int,
    rng: random.Random,
    buckets: Sequence[LengthBucket] = DEFAULT_LENGTH_BUCKETS,
    relaxed_bucket_jaccard: float = 0.85,
    strict_jaccard: float = 0.70,
    short_containment_threshold: float = 0.80,
) -> List[Dict[str, Any]]:
    chosen: List[Dict[str, Any]] = []
    chosen_ids = set()

    # Diagnostic mode: keep up to N representatives per length bucket first,
    # then let contact quality fill any remaining task slots.
    for bucket_name, _min_len, _max_len in buckets:
        if len(chosen) >= max_keep_per_task:
            break
        bucket_pool = [
            x for x in pool
            if x["candidate_id"] not in chosen_ids
            and x.get("length_bucket") == bucket_name
        ]
        n_from_bucket = 0
        while bucket_pool and n_from_bucket < max_per_bucket and len(chosen) < max_keep_per_task:
            jaccard_threshold = relaxed_bucket_jaccard if n_from_bucket == 0 else strict_jaccard
            picked = weighted_pick_with_overlap(
                bucket_pool,
                chosen,
                rng,
                jaccard_threshold=jaccard_threshold,
                short_containment_threshold=short_containment_threshold,
            )
            if picked is None:
                if n_from_bucket == 0:
                    picked = force_best_candidate(bucket_pool, chosen)
                    if picked is None:
                        break
                    picked["step5_selection_stage"] = "bucket_forced_keep"
                else:
                    break
            else:
                picked["step5_selection_stage"] = "length_bucket_cap"
            chosen.append(picked)
            chosen_ids.add(picked["candidate_id"])
            n_from_bucket += 1
            bucket_pool = [x for x in bucket_pool if x["candidate_id"] != picked["candidate_id"]]

    while len(chosen) < max_keep_per_task:
        remaining = [x for x in pool if x["candidate_id"] not in chosen_ids]
        if not remaining:
            break
        picked = weighted_pick_with_overlap(
            remaining,
            chosen,
            rng,
            jaccard_threshold=strict_jaccard,
            short_containment_threshold=short_containment_threshold,
        )
        if picked is None:
            break
        picked["step5_selection_stage"] = "avg_contact_backfill"
        chosen.append(picked)
        chosen_ids.add(picked["candidate_id"])

    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase1 Step5: sample by average contact count")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--max_keep_per_task", type=int, default=4)
    parser.add_argument(
        "--selection_mode",
        choices=["avg_contact_only", "length_bucket_retention", "length_bucket_cap"],
        default="length_bucket_retention",
    )
    parser.add_argument("--length_bucket_max_per_bucket", type=int, default=2)
    parser.add_argument("--overlap_jaccard_threshold", type=float, default=0.70)
    parser.add_argument("--bucket_first_jaccard_threshold", type=float, default=0.85)
    parser.add_argument("--short_containment_threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260416)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    with Path(args.input_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))

    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["parent_task_id"])].append(row)

    selected: List[Dict[str, Any]] = []
    for task_id, task_rows in by_task.items():
        rng = random.Random(args.seed + sum(ord(c) for c in task_id))
        pool = [dict(row) for row in task_rows]
        weights = []
        for row in pool:
            weight_parts = compute_sampling_weight(row)
            row.update(weight_parts)
            row["length_bucket"] = length_bucket_name(
                int(row.get("peptide_length", 0)),
                DEFAULT_LENGTH_BUCKETS,
            )
            weights.append(max(1e-6, weight_parts["sampling_weight"]))
        weight_sum = sum(weights)
        for row, weight in zip(pool, weights):
            row["keep_prob_proxy"] = (weight / weight_sum) if weight_sum > 0 else 0.0

        if len(pool) <= args.max_keep_per_task and args.selection_mode == "avg_contact_only":
            for row in pool:
                annotate_overlap(row, [], "accepted")
                row["step5_selection_stage"] = "avg_contact_only"
            selected.extend(pool)
            continue

        if args.selection_mode == "length_bucket_retention":
            chosen = select_with_length_bucket_retention(
                pool=pool,
                max_keep_per_task=args.max_keep_per_task,
                rng=rng,
                relaxed_bucket_jaccard=args.bucket_first_jaccard_threshold,
                strict_jaccard=args.overlap_jaccard_threshold,
                short_containment_threshold=args.short_containment_threshold,
            )
        elif args.selection_mode == "length_bucket_cap":
            chosen = select_with_length_bucket_cap(
                pool=pool,
                max_keep_per_task=args.max_keep_per_task,
                max_per_bucket=args.length_bucket_max_per_bucket,
                rng=rng,
                relaxed_bucket_jaccard=args.bucket_first_jaccard_threshold,
                strict_jaccard=args.overlap_jaccard_threshold,
                short_containment_threshold=args.short_containment_threshold,
            )
        else:
            chosen = select_weighted_candidates(
                pool=pool,
                max_keep_per_task=args.max_keep_per_task,
                rng=rng,
            )
            for row in chosen:
                row["step5_selection_stage"] = "avg_contact_only"
        selected.extend(chosen)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_json(
        output_path.with_name("step5_summary.json"),
        {
            "input_candidates": len(rows),
            "selected_candidates": len(selected),
            "tasks": len(by_task),
            "max_keep_per_task": args.max_keep_per_task,
            "sampling_basis": args.selection_mode,
            "overlap_filter": {
                "strict_jaccard_threshold": args.overlap_jaccard_threshold,
                "bucket_first_jaccard_threshold": args.bucket_first_jaccard_threshold,
                "short_containment_threshold": args.short_containment_threshold,
            },
            "length_buckets": [
                {
                    "name": name,
                    "min_len": min_len,
                    "max_len": max_len,
                    "target_per_task": 1,
                    "max_per_bucket": args.length_bucket_max_per_bucket,
                }
                for name, min_len, max_len in DEFAULT_LENGTH_BUCKETS
            ],
            "length_bucket_note": "length_bucket_retention samples one candidate from each available length bucket, then backfills by avg_contact_count. length_bucket_cap samples up to length_bucket_max_per_bucket candidates from each bucket, then backfills by avg_contact_count.",
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()

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


def select_with_length_bucket_retention(
    pool: List[Dict[str, Any]],
    max_keep_per_task: int,
    rng: random.Random,
    buckets: Sequence[LengthBucket] = DEFAULT_LENGTH_BUCKETS,
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
        picked = weighted_pick(bucket_pool, rng)
        picked["step5_selection_stage"] = "length_bucket_retention"
        chosen.append(picked)
        chosen_ids.add(picked["candidate_id"])

    # Second pass: fill remaining slots by the original avg-contact weighting.
    while len(chosen) < max_keep_per_task:
        remaining = [x for x in pool if x["candidate_id"] not in chosen_ids]
        if not remaining:
            break
        picked = weighted_pick(remaining, rng)
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
        choices=["avg_contact_only", "length_bucket_retention"],
        default="length_bucket_retention",
    )
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

        if len(pool) <= args.max_keep_per_task:
            selected.extend(pool)
            continue

        if args.selection_mode == "length_bucket_retention":
            chosen = select_with_length_bucket_retention(
                pool=pool,
                max_keep_per_task=args.max_keep_per_task,
                rng=rng,
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
            "length_buckets": [
                {"name": name, "min_len": min_len, "max_len": max_len, "target_per_task": 1}
                for name, min_len, max_len in DEFAULT_LENGTH_BUCKETS
            ],
            "length_bucket_note": "When selection_mode=length_bucket_retention, Step5 first samples one candidate from each available length bucket, then backfills remaining slots by avg_contact_count.",
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()

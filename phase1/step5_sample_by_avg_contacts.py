from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase1 Step5: sample by average contact count")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--max_keep_per_task", type=int, default=4)
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
            weights.append(max(1e-6, weight_parts["sampling_weight"]))
        weight_sum = sum(weights)
        for row, weight in zip(pool, weights):
            row["keep_prob_proxy"] = (weight / weight_sum) if weight_sum > 0 else 0.0

        if len(pool) <= args.max_keep_per_task:
            selected.extend(pool)
            continue

        chosen = select_weighted_candidates(
            pool=pool,
            max_keep_per_task=args.max_keep_per_task,
            rng=rng,
        )
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
            "sampling_basis": "avg_contact_count_only",
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()

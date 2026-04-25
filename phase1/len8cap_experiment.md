# Step 5 Per-Task 8-mer Cap Experiment

This experiment replaces the previous alpha length compensation with a simpler
within-task diversity constraint:

```text
max_keep_per_task = 4
max_len8_per_task = 2
```

The rule is intentionally lightweight: 8-mers are capped only when longer
candidates are available in the same task. If a task only has valid 8-mer
candidates, 8-mers are still allowed to backfill so the task is not emptied.

## Experimental Setup

- Fixed candidate pool: `runs/pilot_1000/step4/step4_features.jsonl`
- Re-ran only Step 5 -> Step 6 -> Step 7.
- Compared:
  - baseline alpha=0
  - alpha=1
  - alpha=2
  - per-task 8-mer cap

## Result

| run | Final n | 8-mer % | 9-12 aa % | 13-20 aa % | avg contact mean | avg contact p50 | coverage mean | longest run mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline alpha=0 | 3274 | 77.55 | 20.19 | 2.26 | 5.1121 | 4.75 | 0.9296 | 7.5165 |
| alpha=1 | 3285 | 76.50 | 21.00 | 2.50 | 5.1068 | 4.75 | 0.9297 | 7.5470 |
| alpha=2 | 3274 | 75.81 | 21.59 | 2.60 | 5.1024 | 4.75 | 0.9299 | 7.5764 |
| per-task 8-mer cap | 3246 | 73.32 | 24.25 | 2.43 | 5.0948 | 4.75 | 0.9288 | 7.5967 |

## Interpretation

- The alpha-based length factor is safe but only weakly effective.
- The per-task 8-mer cap has a clearer effect:
  - 8-mer fraction drops from 77.55% to 73.32%.
  - 9-12 aa fraction increases from 20.19% to 24.25%.
  - `avg_contact_count` and `contact_coverage` remain essentially stable.
- The final sample count changes only slightly, from 3274 to 3246 in this pilot.

## Current Recommendation

Use the per-task 8-mer cap as the default Step 5 length-diversity control:

```text
max_keep_per_task = 4
max_len8_per_task = 2
sampling_weight = avg_contact_count
```

This is easier to explain than a large alpha value and better matches the
intended behavior: keep average contact count as the main signal, while
preventing one task from being filled entirely by 8-mer windows when reasonable
longer alternatives exist.

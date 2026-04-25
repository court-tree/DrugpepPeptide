# Step 5 Length-Balance Sensitivity Experiment

This experiment tests the optional Step 5 length compensation:

```text
sampling_weight = avg_contact_count * (peptide_length / 8)^alpha
```

`alpha = 0` exactly reproduces the original Step 5 behavior.

## Experimental Setup

- Fixed candidate pool: `runs/pilot_1000/step4/step4_features.jsonl`
- Re-ran only Step 5 -> Step 6 -> Step 7.
- Tested `alpha = 0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0`.
- Quality guardrails:
  - `avg_contact_count` should not decrease meaningfully.
  - `contact_coverage` should remain stable.
  - Final sample count should remain stable.

## Step 5 Output

| alpha | Step5 n | 8-mer % | 9-12 aa % | 13-20 aa % | avg contact mean | coverage mean |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6802 | 78.21 | 19.30 | 2.48 | 5.1141 | 0.9293 |
| 0.25 | 6802 | 77.70 | 19.67 | 2.63 | 5.1120 | 0.9293 |
| 0.5 | 6802 | 77.07 | 20.16 | 2.78 | 5.1097 | 0.9293 |
| 0.75 | 6802 | 76.51 | 20.61 | 2.88 | 5.1078 | 0.9290 |
| 1.0 | 6802 | 76.10 | 20.85 | 3.06 | 5.1060 | 0.9289 |
| 1.5 | 6802 | 75.02 | 21.71 | 3.26 | 5.1004 | 0.9284 |
| 2.0 | 6802 | 73.95 | 22.67 | 3.38 | 5.0953 | 0.9285 |

## Final Metadata After Step 6/7

| alpha | Final n | 8-mer % | 9-12 aa % | 13-20 aa % | avg contact mean | avg contact p50 | coverage mean | longest run mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3274 | 77.55 | 20.19 | 2.26 | 5.1121 | 4.75 | 0.9296 | 7.5165 |
| 0.25 | 3282 | 77.33 | 20.35 | 2.32 | 5.1102 | 4.75 | 0.9296 | 7.5250 |
| 0.5 | 3290 | 76.93 | 20.70 | 2.37 | 5.1090 | 4.75 | 0.9296 | 7.5340 |
| 0.75 | 3285 | 76.65 | 20.91 | 2.44 | 5.1075 | 4.75 | 0.9296 | 7.5419 |
| 1.0 | 3285 | 76.50 | 21.00 | 2.50 | 5.1068 | 4.75 | 0.9297 | 7.5470 |
| 1.5 | 3285 | 76.23 | 21.22 | 2.56 | 5.1062 | 4.75 | 0.9296 | 7.5583 |
| 2.0 | 3274 | 75.81 | 21.59 | 2.60 | 5.1024 | 4.75 | 0.9299 | 7.5764 |

## Interpretation

- Length compensation is safe in this pilot: quality metrics remain essentially unchanged.
- The effect is real but modest. Even `alpha = 2.0` only reduces the final 8-mer fraction by about 1.7 percentage points in this pilot.
- This means Step 5 length compensation alone cannot fully solve 8-mer dominance; most of the short-window bias is already present in the Step 3/4 candidate pool.

## Current Recommendation

Use `alpha = 1.0` as a conservative optional setting when length balancing is desired.

Rationale:

- It gives a measurable but still mild shift away from 8-mers.
- It does not meaningfully reduce `avg_contact_count` or `contact_coverage`.
- It stays easier to explain than stronger compensation values.

If stronger length balancing is required, `alpha = 2.0` is still safe in this pilot, but the improvement remains limited; a stronger solution would need to adjust candidate generation or add a length-aware cap/sampling policy.

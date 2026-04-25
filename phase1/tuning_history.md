# Phase1 Tuning History

This file records the main tuning rounds that led to the current working defaults.

| Test Round | Scale | Main Finding | What Changed | Result |
|---|---:|---|---|---|
| Round 1 | `smoke_5` | Step 3 saturated candidate cap; Step 4 passed everything; Step 6 removed nothing | No code change yet, used as baseline inspection | Confirmed Step 3 was too loose and Step 4 thresholding was ineffective |
| Round 2 | `smoke_5_v2` | Reducing Step 3 cap helped, but Step 4 still passed all candidates | `Step3 max_candidates_per_task: 32 -> 16`; `Step4 min_avg_contact_count: 1.0 -> 1.5` | Step 3 shrank, Step 4 still ineffective |
| Round 3 | `smoke_5_v3` | Raising Step 4 from `1.5` to `2.0` still had no effect | `Step4 min_avg_contact_count: 1.5 -> 2.0` | Revealed the metric definition itself was wrong |
| Round 4 | distribution check after `smoke_5_v3` | `avg_contact_count` was in the `44-67` range, obviously too large for a residue-level average contact metric | Re-defined `avg_contact_count` to mean average number of unique receptor residues contacted per peptide residue | Metric scale returned to a realistic range |
| Round 5 | `smoke_5_v4` | Metric scale looked correct, but Step 4 still passed everything | No threshold change yet; validated the new metric definition | Confirmed thresholds, not metric definition, were now the issue |
| Round 6 | `smoke_5_v5` | Step 4 finally started filtering and deduplicating meaningfully | `Step4 min_avg_contact_count -> 4.0`; added `min_contact_coverage = 0.5` | Step 4 began acting as real physical filtering plus dedup |
| Round 7 | `pilot_100` | Step 4 looked too strong; Step 6 also removed over half the samples | Expanded to larger scale without immediate parameter changes | Showed that Step 4 and Step 6 were both aggressive |
| Round 8 | `pilot_100_v2` | Step 4 became more balanced and preserved more tasks | `Step4 min_avg_contact_count: 4.0 -> 3.5`; kept `min_contact_coverage = 0.5` | Better balance; final sample count increased from `296` to `363` |
| Round 9 | `pilot_100_v3` | Relaxing Step 6 helped only slightly, but the behavior was more conservative and easier to justify | `Step6 receptor_identity_threshold: 0.8 -> 0.85`; `peptide_identity_threshold: 0.8 -> 0.85` | Final sample count rose from `363` to `375`, a modest but acceptable improvement |
| Round 10 | `pilot_1000` | The full chain behaved stably at larger pilot scale; Step 4 and Step 6 both had real effects | No further parameter changes; used current defaults | Promoted current settings to working default baseline |

## Current Working Default

### Step 3

- `max_candidates_per_task = 16`

### Step 4

- `min_avg_contact_count = 3.5`
- `min_contact_coverage = 0.5`

### Step 5

- `max_keep_per_task = 4`

### Step 6

- `receptor_identity_threshold = 0.85`
- `peptide_identity_threshold = 0.85`

### Step 7

- `patch_cutoff = 6.0`

## Notes

- Step 3 still tends to saturate the task-level candidate cap.
- Step 4 is now in a useful filtering range rather than passing everything.
- Step 6 remains a strong dedup stage even at `0.85 / 0.85`, which likely reflects real redundancy in the data rather than only an overly strict threshold.

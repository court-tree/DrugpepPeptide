# Next Experiments After Stable Full Run

## Goal

Record the next round of method experiments only after the current end-to-end full run is stable and reproducible.

## 1. Step 4 rBSA Threshold Scan

Purpose:
- Measure how `min_rbsa_raw` affects Step 4 retention, downstream candidate diversity, and final dataset quality.

Suggested scan:
- `0.03`
- `0.05`
- `0.08`
- `0.10`

Track at minimum:
- Step 4: `processed`, `kept`, `dropped_sanity`, `errors`
- Step 5: average candidates per task after filtering
- Step 6: survived count and survived ratio
- Final dataset: peptide length / pocket size / rBSA distribution shift toward reference

## 2. Step 5 Eligible Pool Simplification

Current note:
- Step 3 already enumerates peptide windows in the `8-20` range.
- Therefore Step 5 eligible-pool gating does not need to focus on peptide length again.

Recommended eligible-pool emphasis:
- high enough `contact_coverage_6A`
- sufficient absolute `n_contact_residues_6A`

Question to revisit after Step 4 scan:
- whether `rBSA_raw` should remain a hard gate in Step 5, or be used later / more softly

## 3. Step 5 Length-Bucket Sampling

Replace global in-task weighted sampling with bucketed sampling by peptide length.

Suggested buckets:
- `8-10`
- `11-15`
- `16-20`

Suggested strategy:
- first group candidates by length bucket
- then perform weighted sampling inside each bucket
- use contact-based weights inside bucket
- finally merge selected representatives across buckets

Expected benefit:
- preserve task-level length diversity better
- reduce collapse toward only one local optimum length regime

## 4. Step 6X Upgrade Path

Current stable implementation:
- MMseqs-based receptor sequence clustering

Safe fallback:
- exact receptor-sequence clustering

Longer-term ideal target:
- real family/domain annotation
- e.g. PFAM / UniProt-linked family proxy

Recommended order:
1. MMseqs similarity clustering for stable production
2. exact clustering only as fallback when MMseqs runtime is unavailable
3. true family annotation when engineering bandwidth allows

## Recommended Execution Order

1. First stabilize the current full pipeline end to end
2. Run Step 4 `rBSA` threshold scan
3. Simplify Step 5 eligible-pool conditions toward contact-quality gating
4. Test Step 5 length-bucket sampling
5. Stabilize Step 6X MMseqs clustering defaults and monitor cluster-size behavior
6. Consider true family annotation later

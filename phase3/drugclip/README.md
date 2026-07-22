# Phase-3 Random-Conformer Algorithm

This package implements the current Phase-3 algorithm independently of the
historical PDB-pool implementation in `phase3/active_algorithm`.

The data contract is:

```text
experimental complex -> receptor interface + interface-peptide positive
split biological relations -> sequence-only random conformer cache
one interface-peptide pair -> one uniform conformer
```

Training is interface-pair equal-weighted: every retained `interface_pair_id`
is visited once per epoch. Its receptor interface and peptide sequence are
fixed; only one of the peptide's ten cached conformers is selected uniformly.
An interface pair denotes `receptor_interface_id + peptide_sequence`.
`biological_pair_id` is retained for audit, statistics, logging, and later
ablations, not for epoch length or relation-level interface sampling. Multiple
interfaces under one biological relation remain separate positives because the
same biological relation can contain different receptor binding sites.

Fixed validation loss uses fixed `interface_pair_id` inputs and fixed
conformers for engineering monitoring only. It is not the final all-candidate
retrieval metric.

## Formal release contract

`random_conformer_v3` is the formal dataset and training contract for every
new Phase-3 run. It is selected explicitly with `--data-version v3`, its
release root, and the frozen manifest SHA256. `random_conformer_v2` is only the
historical parent release and a diagnostic compatibility contract; it must not
be used for a new Pilot.

A parent-release reconstruction, when independently authorized, uses
repository-relative inputs and a caller-supplied external MMseqs executable:

```powershell
python -m phase3.drugclip.finalize `
  --biological_pairs_jsonl phase3\runs\receptor_identity_mapping_v1\biological_pairs.jsonl `
  --candidate_evidence_jsonl phase3\runs\raw_rebuild_q_biolip_biolip_v1\candidate_evidence.jsonl `
  --expanded_evidence_jsonl phase3\runs\three_source_dedup_v1\evidence_records.jsonl `
  --mmcif_root phase3\data_sources_raw\RCSB_targeted_phase3_v1\mmcif `
  --qbiolip_root phase3\data_sources_raw\Q-BioLiP `
  --biolip_root dow\BioLiP_download `
  --mmseqs <path-to-mmseqs> `
  --output_dir phase3\runs\drugclip\random_conformer_v2_parent_rebuild `
  --workers 12 `
  --resume
```

The resulting v2-compatible directory is a historical parent build, not a
formal training release. The geometry-hardened v3 release is built separately:

```powershell
python -m phase3.drugclip.build_random_conformer_v3 `
  --parent_dir phase3\runs\drugclip\random_conformer_v2 `
  --output_dir phase3\runs\drugclip\random_conformer_v3
```

`--resume` reuses completed interface and leakage-safe split layers in the
parent reconstruction. Random conformer generation checkpoints after every
peptide and resumes only missing peptides after interruption.

The split graph uses receptor homology, exact peptide identity, and shared PDB
evidence only. It does not use an 80% peptide-similarity threshold. Similar but
non-identical peptides remain independent examples.

The random 3D view generator is `internal-coordinate-rama-v1`: it constructs
the N/CA/C backbone from standard peptide geometry and seeded Ramachandran
basin samples. It has no access to PDB coordinates or receptor context.

The formal data layer joins each interface row to `biological_pairs.jsonl` by
the exact key `(biological_receptor_id, peptide_sequence)`. Dataset length is
the number of unique interface pairs, not the number of biological relations.
The main process builds a deterministic epoch plan containing every interface
pair once; the batch sampler delays interface pairs when needed so a batch
never repeats `peptide_sequence`.

Run the formal v3 read-only data-layer validation with:

```powershell
python -m phase3.drugclip.validate_data_layer `
  --data-version v3 `
  --dataset-root phase3\runs\drugclip\random_conformer_v3 `
  --expected_manifest_sha256 043278F18EFC9B9C3238788D4C6B34C35641C9C26895E5045D8598FA99D5C309 `
  --random_pairs_jsonl phase3\runs\drugclip\random_conformer_v3\04_training_input\random_conformer_pairs.jsonl `
  --random_conformer_cache_jsonl phase3\runs\drugclip\random_conformer_v3\03_random_conformer_cache\random_conformer_cache.jsonl `
  --biological_pairs_jsonl phase3\runs\drugclip\random_conformer_v3\dependencies\biological_pairs.jsonl `
  --pair_splits_jsonl phase3\runs\drugclip\random_conformer_v3\02_leakage_safe_split\pair_splits.jsonl `
  --receptor_interfaces_jsonl phase3\runs\drugclip\random_conformer_v3\01_interface_pairs\receptor_interfaces.jsonl `
  --output_dir phase3\runs\drugclip\data_layer_validation_v3
```

The Dataset reads `DATA_MANIFEST.json`, verifies the formal file hashes, and
rejects v2/v3 contract mixing.

## Formal bounded step-32 Pilot

The only approved new-training shape before result review is a fresh bounded
v3 Pilot. The output directory below must not already exist when the real run
is authorized:

```powershell
python -m phase3.drugclip.train `
  --data-version v3 `
  --dataset-root phase3\runs\drugclip\random_conformer_v3 `
  --expected_manifest_sha256 043278F18EFC9B9C3238788D4C6B34C35641C9C26895E5045D8598FA99D5C309 `
  --train_random_conformer_pairs phase3\runs\drugclip\random_conformer_v3\04_training_input\random_conformer_pairs.jsonl `
  --valid_random_conformer_pairs phase3\runs\drugclip\random_conformer_v3\04_training_input\random_conformer_pairs.jsonl `
  --random_conformer_cache phase3\runs\drugclip\random_conformer_v3\03_random_conformer_cache\random_conformer_cache.jsonl `
  --biological_pairs_jsonl phase3\runs\drugclip\random_conformer_v3\dependencies\biological_pairs.jsonl `
  --pair_splits_jsonl phase3\runs\drugclip\random_conformer_v3\02_leakage_safe_split\pair_splits.jsonl `
  --train_interface_pair_limit 4096 `
  --valid_interface_pair_limit 512 `
  --max_steps 32 `
  --save_steps 32 `
  --output_dir phase3\runs\drugclip\v3_bounded_step032_pilot_review_pending_v1
```

The strict successful-optimizer-step cap must stop after exactly 32 calls to
`optimizer.step()`. The run must produce
`phase3\runs\drugclip\v3_bounded_step032_pilot_review_pending_v1\step_032.pt`.
That checkpoint records `global_step=32`, the v3 manifest/hash contract, model,
optimizer, scheduler, scaler, RNG, and sampling state required for recovery.

Evaluate that exact checkpoint with an explicit safe label:

```powershell
python -m phase3.drugclip.evaluate_full_retrieval `
  --pilot_output phase3\runs\drugclip\v3_bounded_step032_pilot_review_pending_v1 `
  --checkpoint phase3\runs\drugclip\v3_bounded_step032_pilot_review_pending_v1\step_032.pt `
  --model-label step_032 `
  --output_dir phase3\runs\drugclip\v3_bounded_step032_full_retrieval_v1
```

Then run multi-conformer retrieval against the completed single-conformer
candidate contract:

```powershell
python -m phase3.drugclip.evaluate_multi_conformer_retrieval `
  --pilot_output phase3\runs\drugclip\v3_bounded_step032_pilot_review_pending_v1 `
  --checkpoint phase3\runs\drugclip\v3_bounded_step032_pilot_review_pending_v1\step_032.pt `
  --model-label step_032 `
  --single_conformer_output phase3\runs\drugclip\v3_bounded_step032_full_retrieval_v1 `
  --output_dir phase3\runs\drugclip\v3_bounded_step032_multi_conformer_v1
```

The evaluators retain `epoch0_best` and `epoch4_last` only in the explicit
legacy compatibility branch used when no checkpoint/label pair is supplied.
They are not defaults for explicit checkpoint mode. Before the bounded result
is reviewed, do not start full-data, multi-epoch, lower-learning-rate, or any
other expanded training.

The v3 read-only model smoke may compare the formal release with its historical
v2 diagnostic parent:

```powershell
python -m phase3.drugclip.smoke_random_conformer_v3 `
  --v2-root phase3\runs\drugclip\random_conformer_v2 `
  --v3-root phase3\runs\drugclip\random_conformer_v3 `
  --output-json phase3\runs\drugclip\random_conformer_v3_smoke.json
```

The Phase-3 adapter preserves the formal Phase-2 `atoms[:256]` receptor prefix
slice. It can split a residue, but the formal EGNN operates on independent atom
nodes and has no complete-residue or N/CA/C-triad requirement; real CPU/CUDA
forward and backward verification passed on such inputs.

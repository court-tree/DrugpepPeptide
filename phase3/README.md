# PepCLIP Phase-3 Workspace

The active Phase-3 implementation is the DrugCLIP-style random-conformer
fine-tuning algorithm described in `drugclip/ALGORITHM.md`.

## Active namespace

New development must live under:

```text
phase3.drugclip
```

Run outputs must live under:

```text
phase3/runs/drugclip/<run_name>/
```

The package intentionally does not import the previous Phase-3 implementation.
Code under `active_algorithm/` is frozen legacy code despite its historical
directory name. It is preserved for audit and comparison only and must not be
imported by `phase3.drugclip`.

## Preserved data

`data/`, `data_sources_raw/`, and existing `runs/` are preserved. They are not
automatically accepted as inputs by the new algorithm. In particular, the new
algorithm must not consume legacy PDB conformer pools, RMSD clusters, source
priorities, or true-bound conformers as training augmentation.

## Formal DrugCLIP database

Use only:

```text
phase3/runs/drugclip/random_conformer_v3/
```

`random_conformer_v2` is retained only as the historical diagnostic and parent
release for v3. It must not be selected for new training or evaluation runs.

`random_conformer_v1` is superseded. Its split used an unwanted peptide
similarity rule and it must not be used for new development or training.

## Environment check

From the repository root:

```powershell
python -m phase3.drugclip doctor
```

The command checks the active namespace, runtime dependencies, the Phase-2
initialization checkpoint, and the isolated output root without starting a
training run.

## Development rules

- Split real positive pairs before generating conformers.
- Generate each peptide conformer from that peptide sequence, common generator
  settings, and a seed only.
- Treat one interface-peptide relation as one training sample regardless of
  how many conformers were generated.
- Sample one retained conformer uniformly at each training access.
- Mask all other known positives in both contrastive-loss directions.
- Use fixed seeds and fixed conformers for validation and test.
- Start with smoke or pilot runs; do not run full data by default.

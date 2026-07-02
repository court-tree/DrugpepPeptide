# PepCLIP Phase-3 Workspace

This directory is being reset for the next Phase-3 algorithm design pass.

The previous implementation has been archived so new dataset logic can be
written without mixing old code paths, old terminology, and generated
presentation artifacts.

## Directory Layout

```text
phase3/
  README.md
  docs/
    algorithm/        # Current and draft algorithm documents.
    presentations/    # PPT, slide source, and rendered presentation artifacts.
    sources/          # Data-source notes and source-selection documentation.
  archive/
    20260630_current_implementation/
                       # Archived Phase-3 code, tests, scripts, and conformer_v1.
  data/               # Prepared source manifests and intermediate data.
  data_sources_raw/   # Raw downloaded source databases and PDB snapshot files.
  runs/               # Generated dataset runs and audit outputs.
```

## Current Working Rule

New Phase-3 algorithm documents should go under:

```text
E:\pep\phase3\docs\algorithm
```

New code should not be added back to the root directory until the simplified
algorithm is finalized. When implementation resumes, create a fresh package or
pipeline directory with a clear version name instead of modifying archived
files in place.

## Active V1 Builder

The first clean implementation of the simplified V1 algorithm lives in:

```text
E:\pep\phase3\v1
```

Run a small build with:

```powershell
python -m phase3.v1 `
  --input_jsonl E:\pep\phase3\data\phase3_v1_trusted_sources.jsonl `
  --structure_root E:\pep `
  --output_dir E:\pep\phase3\runs\<run_name> `
  --split_mode peptide_exact_sequence
```

V1 outputs only `positive_strong_bound` supervision. It does not generate
conformer augmentation edges, hard conformer negatives, or similar/motif
priors.

The split files are final training JSONL files, not only audit manifests.

Track A is directly readable by `phase2.pepclip.data.PepCLIPDataset` with its
default fields:

```text
receptor_patch_sequence
peptide_sequence
```

Track B is directly readable by `phase2.pepclip.data.PepCLIP3DDataset`.
Each `track_b_<split>.jsonl` row contains inline training atoms:

```text
patch_atoms / receptor_atoms
peptide_atoms
patch_residue_ids
peptide_residue_ids
receptor_key
peptide_key
```

For audit and inspection, each Track B row also points
`receptor_patch_coords_path` / `receptor_coords_path` to:

```text
coords/<interface_id>.receptor_patch.json
```

That sidecar file contains the 10A receptor context heavy atoms, receptor
backbone atoms, receptor residue metadata, 5A interface atom subset, and 5A
receptor-peptide atom contact pairs. Records with incomplete receptor context
backbone atoms are rejected with `receptor_patch_quality`, so Track B does not
silently export residue-only receptor inputs.

V1 also does not silently use asymmetric units for generic structure files. It
accepts source-provided curated complexes, source-provided separate
receptor/peptide structures, or explicit biological assembly reconstruction
from structure metadata. Asymmetric-unit fallback is a non-default debugging
option only.

Each V1 build also writes a deterministic human structure-review package:

```text
<run_dir>\manual_structure_audit\
  README.md
  manual_structure_audit_samples.jsonl
  manual_structure_audit_samples.csv
  case_*.pml
```

The default sample count is 24 and can be changed with
`--manual_audit_sample_count`. Samples are balanced across source, length group,
and split. Each PyMOL script colors receptor chain gray, peptide magenta, 5A
interface residues cyan, and 10A receptor context surface transparent. Close
core contacts <= 3.8A are shown by default as orange distance dashes. The full
<= 5.0A contact object is generated but disabled as `contacts5A_all_hidden`, so
the first view stays readable. The sample table includes blank `human_decision`
and `human_notes` columns for manual pass/fail review.

Receptor-family split is no longer exact receptor-sequence hashing. By default,
V1 clusters receptor sequences with a deterministic sequence-similarity
greedy method and records the thresholds in `dataset_audit.json`. For external
MMseqs/UniProt family assignments, pass `--receptor_family_map`. Use
`--split_mode strict` when peptide sequence, receptor family, and PDB leakage
must all be controlled simultaneously.

## MMseqs2 Environment

MMseqs2 is installed locally for WSL without requiring root:

```text
E:\pep\phase3\tools\mmseqs2_local
```

Check it with:

```powershell
wsl.exe bash -lc "/mnt/e/pep/phase3/tools/mmseqs2_local/mmseqs.sh version"
```

Build a receptor family map from a V1 anchor table:

```powershell
python -m phase3.sources.build_receptor_family_map_mmseqs `
  --anchor_jsonl E:\pep\phase3\runs\<first_pass_run>\receptor_peptide_anchor.jsonl `
  --output_jsonl E:\pep\phase3\runs\<family_map_run>\receptor_family_map.jsonl `
  --work_dir E:\pep\phase3\runs\<family_map_run> `
  --min_seq_id 0.4 `
  --coverage 0.6 `
  --cov_mode 0
```

Then rerun V1 with the MMseqs family map:

```powershell
python -m phase3.v1 `
  --input_jsonl E:\pep\phase3\data\phase3_v1_trusted_sources.jsonl `
  --structure_root E:\pep `
  --output_dir E:\pep\phase3\runs\<strict_mmseqs_run> `
  --split_mode strict `
  --receptor_family_map E:\pep\phase3\runs\<family_map_run>\receptor_family_map.jsonl
```

Current trusted source table:

```text
E:\pep\phase3\data\phase3_v1_trusted_sources.jsonl
```

Current trusted source coverage:

```text
BioLiP_peptide
Q-BioLiP_PIII
PepBDB
```

`raw PDB` is not allowed as a V1 label source. PDB/mmCIF files are only a
coordinate reservoir referenced by curated sources.

Propedia is not part of the default trusted V1 source table. A recoverable
Propedia subset exists under `E:\pep\phase3\data\propedia_full`, but the local
`complex.zip` lacks a central directory and is truncated, so Propedia remains a
partial/experimental source until a complete archive is available.

## Archived Implementation

The archived code is here:

```text
E:\pep\phase3\archive\20260630_current_implementation
```

It includes the previous pair-dataset builder, conformer evidence pipeline,
run scripts, and tests. Treat it as reference material, not the active
Phase-3 implementation.

## Data And Runs

The existing generated data and run outputs were not moved:

```text
E:\pep\phase3\data
E:\pep\phase3\data_sources_raw
E:\pep\phase3\runs
```

These remain available for inspection and comparison while the new algorithm
logic is being written.

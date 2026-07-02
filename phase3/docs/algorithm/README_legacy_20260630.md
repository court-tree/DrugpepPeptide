# Phase 3 Fine-Tuning Dataset

This directory builds the Phase-3 PepCLIP fine-tuning dataset from curated
protein-peptide complex records.

The dataset target is:

```text
receptor interface <-> peptide bound conformer
```

V1 uses real paired structure records as the peptide and receptor-pair source.
The default local candidate source is BioLiP peptide records, followed by the
same local contact and peptide-continuity validation used throughout this
directory. It does not use unpaired peptide sequence databases as training
supervision.

See `data_sources.md` for the paired-source selection rules.

## Version Roadmap

The current builder targets the V1/V1-beta real bound-pair dataset. The
full conformer-augmented design is split into later versions so baseline
fine-tuning, exact-sequence conformer evidence, hard conformer negatives, and
similar/motif priors can be evaluated separately.

See `phase3_version_plan.md` for the V1, V1.5, V2, and V3 boundaries.

## Inputs

The builder consumes a JSONL file of source records. Each row should include:

```json
{
  "source_database": "example",
  "source_entry_id": "entry-1",
  "pdb_id": "1abc",
  "complex_structure_file": "1abc-assembly1.cif",
  "biological_assembly_id": "1",
  "receptor_chain_id": "A",
  "peptide_chain_id": "B",
  "peptide_residue_start": 1,
  "peptide_residue_end": 8
}
```

`complex_structure_file` may be absolute or relative to `--structure_root`.
When `peptide_residue_start` and `peptide_residue_end` are omitted, the full
peptide chain is used.

## Build

BioLiP peptide-record preparation, using the local downloaded copy:

```powershell
python -m phase3.prepare_biolip_peptide_records `
  --biolip_table E:\pep\dow\BioLiP_nr.txt `
  --structure_root E:\pep\dow\BioLiP_download `
  --output_dir E:\pep\phase3\data\biolip_peptide_v1
```

This writes:

- `biolip_peptide_phase3_records.jsonl`
- `biolip_peptide_prepare_summary.json`

Build Phase-3 fine-tuning samples from the prepared BioLiP peptide records:

```powershell
python -m phase3.build_finetune_dataset `
  --input_jsonl E:\pep\phase3\data\biolip_peptide_v1\biolip_peptide_phase3_records.jsonl `
  --structure_root E:\pep\dow\BioLiP_download `
  --output_dir E:\pep\phase3\runs\biolip_peptide_v1
```

Generic source-record build:

```powershell
python -m phase3.build_finetune_dataset `
  --input_jsonl E:\pep\phase3\records.jsonl `
  --structure_root E:\pep\pdb `
  --output_dir E:\pep\phase3\runs\pilot_v1
```

Outputs include:

- `peptide_sequence_pool.jsonl`
- `peptide_source_record.jsonl`
- `receptor_peptide_pair.jsonl`
- `peptide_conformer_evidence.jsonl`
- `conformer_cluster.jsonl`
- `track_a_<split>.jsonl`
- `track_b_<split>.jsonl`
- `phase3_summary.json`

## V1 Rules

- Use biological assembly files when possible.
- Recompute heavy-atom receptor-peptide contacts.
- Extract receptor interface residues at 6.0 A around the bound peptide.
- Use the bound conformer from the current complex as the training conformer.
- Cluster exact-sequence bound conformers by aligned peptide C-alpha RMSD
  (default 1.5 A). Similarity is defined within a conformer cluster, not across
  every occurrence of the sequence.
- Use connected-component split grouping over peptide, conformer, receptor,
  interface, and PDB keys to reduce validation/test leakage.

## Full V1 Run

BioLiP peptide records and Q-BioLiP PIII are the primary V1 paired sources.
Q-BioLiP receptor assemblies and peptide structures are materialized once
under `data_sources_raw/Q-BioLiP/extracted`.

Prepare sources and run preflight without starting the expensive build:

```powershell
powershell -ExecutionPolicy Bypass -File E:\pep\phase3\run_full_v1.ps1
```

Start the full coordinate/contact build after preflight:

```powershell
powershell -ExecutionPolicy Bypass -File E:\pep\phase3\run_full_v1.ps1 `
  -SkipPrepare -RunBuild
```

Use `-ParseAllPreflight` for a slower parse of every referenced structure.

## V1-Alpha Full-Run Result

The June 24, 2026 full run is retained as an engineering alpha result:

```text
E:\pep\phase3\runs\phase3_v1_full_final
```

Counts:

- 7,945 receptor-interface/peptide positive pairs
- 3,099 unique peptide sequences
- 6,234 bound peptide conformer instances
- 3,240 sequence-specific conformer clusters
- train/validation/test: 6,356 / 794 / 795

`phase3_v1_full` is the preserved pre-finalization output and must not be used.
`phase3_v1_full_final` fixed its conformer IDs, but it predates strict
nonstandard-residue and peptide-continuity filtering. Treat both directories as
audit artifacts, not the final paper-grade training dataset. The next accepted
dataset must be produced by the V1-beta builder and pass the expanded quality
and leakage audit.

V1-beta adds whole-segment nonstandard-residue rejection, strict peptide
continuity checks, explicit source confidence tiers (`high_prior` /
`medium_prior`), Q-BioLiP single-chain validation, split-component statistics,
input hashes, parameter snapshots, and a final `dataset_audit.json`.

The prepared V1-beta source manifest is:

```text
E:\pep\phase3\data\phase3_v1_beta_full_sources.jsonl
```

## Full-PDB Conformer V1

`phase3/conformer_v1` adds the missing full-PDB exact-match conformer evidence
layer. It does not turn an external fragment into a receptor-positive pair.

Pipeline:

```text
positive peptide pool
  -> full experimental PDB SEQRES exact-match search
  -> entry mmCIF download and biological-assembly membership validation
  -> strict coordinate/backbone/altloc/occupancy quality control
  -> 30% source-family limits
  -> complete-linkage backbone RMSD clustering
  -> bound conformer mapping and auxiliary same-cluster views
  -> family-aware final split and audit
```

Prepare a 20-peptide smoke run:

```powershell
powershell -ExecutionPolicy Bypass -File E:\pep\phase3\run_conformer_v1.ps1 `
  -BaseDatasetDir E:\pep\phase3\runs\<phase3_pair_dataset> `
  -Mode smoke `
  -OutputDir E:\pep\phase3\runs\phase3_conformer_v1_smoke
```

Modes:

- `smoke`: 20 peptide queries, at most 10 coordinate PDBs per peptide.
- `pilot`: 500 peptide queries, at most 50 coordinate PDBs per peptide.
- `full`: all peptide queries, at most 250 coordinate PDBs per peptide.

All modes search the complete frozen `pdb_seqres` snapshot. The coordinate cap
only limits which exact-match hits proceed to expensive mmCIF extraction.
After coordinate QC, at most two instances per 30% source family and at most
50 external conformers per peptide are retained.

Key outputs:

- `pdb_snapshot_manifest.json`
- `pdb_sequence_hits.jsonl`
- `external_exact_match_conformers.jsonl`
- `peptide_conformer_evidence.jsonl`
- `conformer_cluster.jsonl`
- `bound_conformer_cluster_mapping.jsonl`
- `source_family_cluster.jsonl`
- `peptide_conformer_views_<split>.jsonl`
- `receptor_peptide_pair.jsonl` with the final family-aware split
- `conformer_mining_summary.json`
- `dataset_audit.json`

3D training accepts:

```powershell
python -m phase2.pepclip.train_3d `
  --train_track_b <conformer_run>\track_b_train.jsonl `
  --valid_track_b <conformer_run>\track_b_validation.jsonl `
  --data_format jsonl `
  --train_conformer_views <conformer_run>\peptide_conformer_views_train.jsonl `
  --conformer_evidence <conformer_run>\peptide_conformer_evidence.jsonl `
  --conformer_consistency_weight 0.1 `
  --output_dir <training_output>
```

The concat-fusion trainer supports the same three conformer arguments.
Consistency is only enabled for bound conformers whose cluster is
`repeated_independent`. If the coverage target is not reached,
`conformer_mining_summary.json` sets
`auxiliary_consistency_default_enabled=false`; callers should omit the
conformer training arguments for the baseline run.

Long full runs are resumable because existing mmCIF and API-cache files are
reused. After a completed sequence search, resume without rescanning SEQRES:

```powershell
powershell -ExecutionPolicy Bypass -File E:\pep\phase3\run_conformer_v1.ps1 `
  -BaseDatasetDir E:\pep\phase3\runs\phase3_v1_beta_full `
  -Mode full `
  -SkipSnapshot `
  -SkipSearch
```

`-MaxCandidatePdbPerPeptide N` may be used for staged coordinate runs while
leaving the complete exact-match hit table unchanged.

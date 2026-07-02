# Phase-3 V1 Source Reliability Audit

Date: 2026-06-30

## Conclusion

Do not treat all downloaded databases as equally reliable V1 label sources.

Default V1 should use the trusted source table:

```text
E:\pep\phase3\data\phase3_v1_trusted_sources.jsonl
```

Trusted source coverage:

```text
BioLiP_peptide: 3808
Q-BioLiP_PIII: 15287
PepBDB: 19131
total: 38226
```

Propedia is available only as a partial recoverable subset and should not be in
the default trusted V1 source table until a complete archive is available.

Raw PDB is not allowed as a V1 positive-label source.

---

## Audit Criteria

Each source was checked for:

```text
1. Whether it is a curated receptor-peptide annotation source.
2. Whether prepared records exist.
3. Whether structure paths exist for all prepared records.
4. Whether the first 100 structures per source are readable by gemmi.
5. Whether the V1 builder can produce positive_strong_bound anchors from source-specific smoke inputs.
6. Whether the local raw archive is complete or only partial.
```

---

## Source Verdicts

| Source | V1 status | Reason |
|---|---|---|
| BioLiP_peptide | Trusted default | Curated peptide-ligand records; all prepared structure paths exist; first 100 structures readable; V1 smoke passed. |
| Q-BioLiP_PIII | Trusted default | Curated quaternary-interaction records; separate receptor/peptide structures exist; first 100 structures readable; V1 smoke passed after builder support for separate files. |
| PepBDB | Trusted default, Tier-2 | Curated protein-peptide complexes; full local adapter produced 19,131 records; all prepared structure paths exist; first 100 structures readable; V1 smoke passed. |
| Propedia | Partial / experimental | Metadata is usable and a recoverable subset exists, but local structure archive is truncated and only 1,222 of 6,758 V1-canonical candidates could be recovered. Do not use by default. |
| raw PDB | Not allowed | PDB/mmCIF is a coordinate reservoir only, not a curated V1 label source. |

---

## Prepared Source Tables

Trusted default:

```text
E:\pep\phase3\data\phase3_v1_trusted_sources.jsonl
E:\pep\phase3\data\phase3_v1_trusted_sources.summary.json
E:\pep\phase3\data\phase3_v1_trusted_sources.coverage.json
```

All curated including partial Propedia:

```text
E:\pep\phase3\data\phase3_v1_curated_sources.jsonl
E:\pep\phase3\data\phase3_v1_curated_sources.summary.json
E:\pep\phase3\data\phase3_v1_curated_sources.coverage.json
```

Use `phase3_v1_curated_sources.jsonl` only for sensitivity experiments that
explicitly allow partial Propedia.

---

## Raw / Prepared Completeness

### BioLiP + Q-BioLiP

Merged Tier-1 table:

```text
E:\pep\phase3\data\phase3_v1_full_sources.jsonl
```

Summary:

```text
input_rows: 19096
written_records: 19095
dropped_exact_duplicates: 1
source_counts:
  BioLiP_peptide: 3808
  Q-BioLiP_PIII: 15287
```

Structure-path audit:

```text
BioLiP_peptide:
  records: 3808
  missing_structure_paths: 0
  sample_read_errors_first100: 0

Q-BioLiP_PIII:
  records: 15287
  separate_receptor_peptide_records: 15287
  missing_structure_paths: 0
  sample_read_errors_first100: 0
```

### PepBDB

Prepared table:

```text
E:\pep\phase3\data\pepbdb_full\pepbdb_phase3_records.jsonl
```

Summary:

```text
processed_entries: 13508
written_records: 19131
failed_entries: 457
```

The failed entries were excluded during source preparation, mainly because no
protein receptor chain or no peptide protein chain could be identified.

Structure-path audit:

```text
records: 19131
missing_structure_paths: 0
sample_read_errors_first100: 0
```

### Propedia

Prepared partial table:

```text
E:\pep\phase3\data\propedia_full\propedia_phase3_records.jsonl
```

Summary:

```text
canonical 8-20-aa metadata candidates: 6758
needed_complex_files: 6758
extracted_complex_files: 1222
written_records: 1222
missing_complex_files: 5536
```

The local `complex.zip` starts with valid ZIP local headers but lacks a central
directory and is truncated in the later part of the archive. The adapter can
recover the readable prefix, but this is not a complete Propedia source.

Structure-path audit for recovered subset:

```text
records: 1222
missing_structure_paths: 0
sample_read_errors_first100: 0
```

Verdict:

```text
Do not include Propedia in the default trusted V1 dataset.
Use only as partial-source sensitivity analysis unless a complete archive is supplied.
```

---

## Builder Smoke Tests

Each source was tested with 200 prepared rows.

```text
BioLiP_peptide:
  input: 200
  final_anchor_count: 69
  leakage same_peptide_sequence_cross_split_count: 0

Q-BioLiP_PIII:
  input: 200
  final_anchor_count: 97
  leakage same_peptide_sequence_cross_split_count: 0

PepBDB:
  input: 200
  final_anchor_count: 43
  leakage same_peptide_sequence_cross_split_count: 0

Propedia:
  input: 200
  final_anchor_count: 166
  leakage same_peptide_sequence_cross_split_count: 0
```

The Propedia recovered subset can produce anchors, but builder success does not
make the source complete. Completeness and archive integrity are the limiting
issues.

---

## Policy For V1 Runs

Use this for default V1:

```powershell
python -m phase3.v1 `
  --input_jsonl E:\pep\phase3\data\phase3_v1_trusted_sources.jsonl `
  --structure_root E:\pep `
  --output_dir E:\pep\phase3\runs\<run_name> `
  --split_mode peptide_exact_sequence
```

Use this only for partial-source sensitivity experiments:

```powershell
python -m phase3.v1 `
  --input_jsonl E:\pep\phase3\data\phase3_v1_curated_sources.jsonl `
  --structure_root E:\pep `
  --output_dir E:\pep\phase3\runs\<run_name> `
  --split_mode peptide_exact_sequence
```

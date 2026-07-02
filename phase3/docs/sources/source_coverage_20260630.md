# Phase-3 V1 Source Coverage Status

Date: 2026-06-30

## Source Policy

V1 positive labels must come from curated receptor-peptide annotation sources.

Allowed V1 label sources:

```text
Tier 1:
  BioLiP_peptide
  Q-BioLiP_PIII

Tier 2:
  PepBDB
  Propedia
```

Not allowed as V1 label source:

```text
raw PDB protein-peptide mining
```

PDB/mmCIF files are allowed only as coordinate reservoirs referenced by curated
sources.

## Current Source Tables

Default trusted source table:

```text
E:\pep\phase3\data\phase3_v1_trusted_sources.jsonl
```

Trusted coverage:

```text
BioLiP_peptide: 3808
Q-BioLiP_PIII: 15287
PepBDB: 19131
total: 38226
```

All curated source table, including partial Propedia:

```text
E:\pep\phase3\data\phase3_v1_curated_sources.jsonl
```

All-curated coverage:

```text
BioLiP_peptide: 3808
Q-BioLiP_PIII: 15287
PepBDB: 19131
Propedia: 1222
total: 39448
```

Coverage audit:

```text
E:\pep\phase3\data\phase3_v1_curated_sources.coverage.json
```

Merge summary:

```text
E:\pep\phase3\data\phase3_v1_curated_sources.summary.json
```

## PepBDB Integration

PepBDB was prepared from:

```text
E:\pep\phase3\data_sources_raw\PepBDB\raw\pepbdb-20200318\pepbdb
```

Output:

```text
E:\pep\phase3\data\pepbdb_full\pepbdb_phase3_records.jsonl
```

Summary:

```text
processed_entries: 13508
written_records: 19131
failed_entries: 457
```

PepBDB entries are written with absolute `complex_structure_file` paths so the
V1 builder can consume the merged curated table with a single `--structure_root`.

## Propedia Integration

Propedia metadata is present:

```text
E:\pep\phase3\data_sources_raw\Propedia\complex.csv
```

The downloaded `complex.zip` lacks a ZIP central directory and is truncated in
the later part of the archive. The adapter therefore streams local ZIP headers
and recovers the readable complex PDB subset.

```text
canonical 8-20-aa metadata candidates: 6758
recovered complex PDB files: 1222
written Propedia V1 records: 1222
missing due to unavailable/truncated complex files: 5536
```

Current Propedia records:

```text
E:\pep\phase3\data\propedia_full\propedia_phase3_records.jsonl
```

Summary:

```text
E:\pep\phase3\data\propedia_full\propedia_prepare_summary.json
```

Propedia is present only in `phase3_v1_curated_sources.jsonl`, not in the
default trusted V1 source table. It is a recoverable subset from a local
truncated archive.

## Mixed-Source Builder Smoke Test

Input:

```text
E:\pep\phase3\runs\v1_curated_sources_mixed_smoke_input.jsonl
```

The smoke input contains 50 rows from each active source:

```text
BioLiP_peptide: 50
Q-BioLiP_PIII: 50
PepBDB: 50
Propedia: 50
```

Output:

```text
E:\pep\phase3\runs\v1_curated_4source_smoke
```

Result:

```text
final_anchor_count: 99
kept_by_source:
  Propedia: 41
  Q-BioLiP_PIII: 26
  BioLiP_peptide: 17
  PepBDB: 15
leakage_checks:
  same_peptide_sequence_cross_split_count: 0
  same_pdb_cross_split_count: 0
  same_receptor_family_cross_split_count: 1
```

This confirms that the V1 builder can consume the merged BioLiP +
Q-BioLiP + PepBDB + Propedia source table and produce `positive_strong_bound`
anchors from all four curated sources. The receptor-family cross-split
count reflects the current peptide-exact split mode; strict receptor-family
split remains a separate V1 enhancement.

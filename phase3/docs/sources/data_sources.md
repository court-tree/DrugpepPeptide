# Phase 3 Paired Data Sources

Phase 3 V1 needs real paired structure supervision:

```text
receptor chain/interface <-> peptide-chain bound conformer
```

The source database must provide enough information to recover both chains from
a structure file and revalidate contacts locally. A peptide sequence list alone
is not enough.

## Required Source Fields

Each source record should provide:

- PDB ID or structure file path
- receptor chain ID
- peptide/source chain ID
- peptide residue range when the peptide is a window rather than a whole chain
- source database and source entry ID
- biological assembly or source-file provenance when available

Phase 3 then recomputes:

- peptide length and sequence
- bound peptide atoms/backbone conformer
- receptor-interface residues
- heavy-atom contact statistics
- split leakage keys

## V1 Sources

1. `BioLiP peptide` records

   Local files already available:

   ```text
   E:\pep\dow\BioLiP_nr.txt
   E:\pep\dow\BioLiP_download
   ```

   BioLiP contains mixed ligand types. V1 keeps only rows with
   `ligand_id = peptide`, then validates the receptor chain and peptide chain in
   the local structure file.

2. `Q-BioLiP PIII` records

   Q-BioLiP provides biological-assembly receptor coordinates and separate
   peptide coordinates. The adapter emits one sample per annotated interacting
   receptor chain, preserving the single-chain receptor baseline. Multiple
   receptor-chain positives for the same peptide remain in the same split.

The merged full-run source list is:

```text
E:\pep\phase3\data\phase3_v1_full_sources.jsonl
```

## Not A Default V1 Source

PepBDB is not treated as the default V1 source in this repository. It can be a
supplementary candidate source, but the current Phase-3 V1 baseline should start
from curated protein-peptide complex records with recoverable receptor and
peptide chains.

Unpaired peptide sequence databases are not training supervision. They may be
used later as candidate libraries for retrieval, but not as positive
`receptor-interface <-> peptide` fine-tuning pairs.

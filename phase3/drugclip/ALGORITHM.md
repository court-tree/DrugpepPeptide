# Phase-3 DrugCLIP algorithm contract

This directory implements the 2026-07-10 DrugCLIP-style random-conformer
fine-tuning design. This file is the development gate; code that violates these
rules must not be merged into this namespace.

## Training identity and augmentation

- A training unit is one unique interface--peptide positive,
  `interface_pair_id = receptor_interface_id + peptide_sequence`.
- Training weight is one per `interface_pair_id`, independent of the number of
  cached conformers. `biological_pair_id` is mapped exactly by
  `biological_receptor_id + peptide_sequence` to `biological_pairs.jsonl`, but
  is retained only for audit, statistics, logging, and later ablation.
- Generate at most `M=10` conformers by default, using only the peptide's own
  sequence, common generator parameters, and a random seed.
- Each epoch visits every retained `interface_pair_id` exactly once. The
  interface is fixed by that pair; sample one retained conformer of its peptide
  uniformly at each visit.
- Multiple interfaces in one biological relation remain distinct training
  positives because they can represent different receptor binding sites; they
  are never compressed by relation-level interface sampling.
- Other conformers of the same peptide are neither negatives nor additional
  positive relations.

## Structural evidence boundary

The true complex establishes the real positive relation and receptor interface.
The true-bound peptide may be used for contact/QC and an independent evaluation,
but never as a training conformer, template, constraint, priority, or RMSD-based
selection target.

The first version excludes PDB conformer search, similar-sequence templates,
RMSD clustering, source priority, conformer labels/ranking, hard conformer
negatives, consistency losses, residual pair heads, and multi-positive
conformer losses.

## Split and evaluation

- Deduplicate and build the full known-positive graph before splitting.
- No identical peptide, receptor/interface leakage, duplicate complex, or
  configured receptor homology may cross splits. Similar but non-identical
  peptides are ordinary independent peptides and do not create split edges.
- Peptide split identity is exact peptide sequence identity only; the historical
  80% peptide-similarity rule is not used.
- Generate augmentation independently inside each completed split.
- Training uses bidirectional in-batch contrastive loss with known-positive
  denominator masks in both directions.
- Fixed validation loss visits fixed `interface_pair_id` inputs with fixed
  conformers. It is an engineering-monitoring loss, not the final
  all-candidate retrieval metric.
- Report single-conformer retrieval, repeated/multi-conformer robustness,
  true-bound evaluation separately, and 1D-only/3D-ablation controls.

The model claims robust pair retrieval under uncertain peptide conformation. It
does not claim binding-pose prediction or conformer ranking.

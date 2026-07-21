# Phase-3 Random-Conformer Algorithm Contract

This package is isolated from `phase3.active_algorithm`, which retains the
historical PDB-conformer-pool algorithm. Neither package imports the other.

- A positive training unit is the unique `interface_pair_id =
  receptor_interface_id + peptide_sequence`, with direct experimental complex
  evidence and a coordinate-validated interface.
- The complex supplies only the positive relation and receptor interface.
- Peptide 3D views are generated from peptide sequence alone with the fixed
  `internal-coordinate-rama-v1` generator and stored split-local seeds. It uses
  standard peptide bond geometry and seeded Ramachandran basin sampling.
- The default cache target is `M=10`; fewer valid conformers are retained; no
  valid conformer excludes that pair from this algorithm's 3D fine-tuning.
- PDB true-bound coordinates, exact/similar templates, receptor context, and
  label-dependent filtering are prohibited from random-conformer generation.
- Split interface-positive relations before cache generation. Split uses shared
  receptor biology/homology, exact peptide identity, and shared experimental
  PDB sources. Similar but non-identical peptides do not create split edges,
  labels, known positives, or sampling rules. All generated conformers inherit
  their exact peptide's split. The historical 80% peptide-similarity rule is
  not used.
- Known-positive masks are interface-to-peptide tables in both directions.
- Training gives every unique `interface_pair_id` equal weight. Every epoch
  visits each retained interface--peptide pair once; its interface is fixed and
  one cache member is sampled uniformly. A `biological_pair_id` is audit and
  statistics metadata only: it must not define epoch length, compress
  interfaces, or sample the main validation input. No conformer is an extra
  positive or a conformer-ranking target.
- Multiple interfaces under one biological relation are preserved as separate
  positives because they may represent distinct binding sites.
- Fixed validation loss uses fixed interface-pair inputs and fixed conformers
  only for engineering monitoring; it is not the final all-candidate retrieval
  metric.

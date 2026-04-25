One PDB = one peptide interface.

Chain semantics:
  F = receptor full chain (only if --include_full_receptor)
  L = original peptide-source full chain (only if --include_source_full_chain)
  X = receptor local patch around the exported peptide window
  P = exported peptide window

Recommended first look in PyMOL:
  hide everything, all
  show cartoon, chain F
  color gray70, chain F
  show cartoon, chain L
  color yellow, chain L
  show sticks, chain X
  color cyan, chain X
  show sticks, chain P
  color orange, chain P
  orient
  zoom visible, 6

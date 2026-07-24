# PepCLIP Current Project State

Last verified: 2026-07-24

## Current Phase

Phase-1 teacher data is frozen, Phase-2 model development is frozen, and the
active path is `phase3.drugclip`. No training or evaluation process is
currently running. Formal v3 random-conformer fine-tuning, fixed single- and
ten-conformer evaluation, the epoch-0 balanced model selection, and its
machine-readable release contract are complete. A subsequent authorized
fixed-512 input-domain ablation completed as a read-only diagnostic against
the unchanged Phase-2 and selected Phase-3 epoch-0 checkpoints. It isolates
true-bound all-heavy peptide input, its exact N/CA/C subset, random conformer
0, and arithmetic-mean scores across random conformers 0-9. The result shows
that removal of full-heavy/side-chain information is the primary observed
input-domain loss; randomizing the already-backbone-only input is a smaller
effect. This does not establish that all-heavy random conformers work.

## Committed Implementation Baseline

The A-E implementation groups are committed as independent boundaries:

- `9dac8dad0 Add Phase 3 DrugCLIP foundation`
- `ec40e83fa Add random conformer v3 release tooling`
- `472ae8455 Add learned concat fusion model`
- `3928e00e6 Add Phase 3 v3 runtime contract`
- `7de86c0a3 Add bounded Phase 3 training and evaluation`

A clean detached worktree at HEAD `7de86c0a3` passed all 68 Phase-3 tests, all
6 Phase-2 concat contract tests, and the `--help` entry points for training,
full retrieval, and multi-conformer retrieval. The learned-concat checkpoint
was supplied to that worktree through a Windows hardlink. The temporary
worktree was safely removed after verification.

This earlier verification proves that the committed A-E code can run
independently from a clean worktree with the required external checkpoint
supplied. By itself it did not prove a real v3 GPU Pilot, full retrieval, or
ten-conformer evaluation; the formal bounded acceptance recorded below now
provides that real-run evidence.

## Frozen Upstream Baselines

### Phase-1 teacher data

- Canonical run: `E:\pep\phase1\runs\full_run_v9_resume`.
- The current `phase1/README.md`, `step3_window_candidates.py`, and
  `step5_sample_by_avg_contacts.py` have zero diff from tag `phase1-v9`.
- Final rows: 175,810; unique peptide sequences: 131,021; mean peptide length:
  11.1317 aa.
- Dual-track LMDB counts: 158,229 main-train and 17,581 monitor rows for both
  Track A and Track B.
- `full_run_v9` is an interrupted historical directory; do not use it instead
  of `full_run_v9_resume`. v10-v12 are controls, not the teacher baseline.

### Phase-2 model roles

Two different frozen artifacts serve different roles and must not be
conflated:

- Formal Phase-3 initialization chain: the learned 1D+3D concat implementation
  in `phase2/pepclip/train_concat_fusion.py` and its frozen dual-encoder
  checkpoint at
  `E:\pep\phase2\runs\v9_concat_fusion_partial_unfreeze_1d3d_last1_from3d_e40_v1\checkpoint_best.pt`.
  Its saved monitor metrics are best epoch 40, Recall@1 0.24771, Recall@5
  0.71793, Recall@10 0.79313, MRR 0.45394, and median rank 2. Track A/B
  sample IDs are aligned for all 158,229 train and 17,581 validation rows.
- Existing raw cross-set retrieval artifact: manual score fusion
  `0.45 * score_1d + 0.55 * score_3d`, with paired-monitor Recall@10 0.80638,
  at `E:\pep\phase2\runs\v10_crossset_score_fusion_045_055_raw_top500\top500.jsonl`.

`phase2/runs/summary.md`, `phase2_v9_teacher_frozen_report.md`, and the current
Phase-2 PPT still describe the manual fusion as the final/default model. That
narrative is stale for the learned-model role, but the raw Top500 artifact
itself remains valid historical output. Do not edit those business documents
as part of Phase-3 experiments.

## Active Phase-3 Contract

- Active namespace: `phase3.drugclip`; `phase3.active_algorithm` is historical.
- Formal data release:
  `E:\pep\phase3\runs\drugclip\random_conformer_v3`.
- Manifest SHA256:
  `043278F18EFC9B9C3238788D4C6B34C35641C9C26895E5045D8598FA99D5C309`.
- v3 contains 24,633 interface pairs, 6,979 caches, and 69,790 conformers.
  It replaced 516 clash-failing conformers while preserving pair identities,
  splits, interfaces, peptide sequences, mappings, and known-positive groups.
- Release validation is PASS: zero semantic differences from v2, zero missing
  cache references, zero independent clash15 failures, and zero determinism
  mismatches.
- The v3 data/model smoke visited all 19,707 train pairs exactly once across
  1,232 batches, with zero pair loss, pair duplication, or within-batch peptide
  duplication. CPU, CUDA, CUDA BF16 backward, and CUDA FP32 backward were
  finite; all four major trainable module groups received nonzero gradients.
- These checks satisfy the v3 data/model prerequisites. The formal bounded
  acceptance has now additionally proved the real training and evaluator
  execution chain described below.
- New work must explicitly select `--data-version v3` and provide the v3
  `--dataset-root`. Checkpoint resume must preserve the manifest/data contract.

Primary evidence:

- `E:\pep\phase3\runs\drugclip\random_conformer_v3\DATA_MANIFEST.json`
- `E:\pep\phase3\runs\drugclip\random_conformer_v3\VALIDATION_REPORT.md`
- `E:\pep\phase3\runs\drugclip\random_conformer_v3_smoke_20260721.json`

## Formal v3 Bounded End-To-End Acceptance

The clean-worktree acceptance at HEAD `a88f0c1f0` completed on 2026-07-22.
This is an engineering gate for the Phase-3 random-conformer fine-tuning main
chain, not a new step-window or hyperparameter-study branch.

- Training completed exactly 32 successful `optimizer.step()` calls over 32
  batches and 512 actual interface-pair visits. All 88 optimizer parameter
  states record step 32. Training and validation directional losses were
  finite.
- The 4,096-pair epoch plan assigned all conformer indices 0-9. The sampler
  reported zero interface-pair loss, zero interface-pair duplication, and
  zero within-batch peptide uniqueness violations. The successful finite
  bidirectional training path exercised the exact known-positive masks.
- `step_032.pt` is 178,079,323 bytes with SHA256
  `E37E2D225464E059003E19C0697B32D34F8CE2D9FE4725BDDB776A60061B06BE`.
  It records schema `pepclip-phase3-drugclip-training-v1`, `global_step=32`,
  the formal v3 manifest/data contract, model, optimizer, scheduler, scaler,
  RNG, and sampler state. It is the only `step_*.pt` produced.
- Single-conformer fixed-512 full-candidate retrieval completed. For
  receptor-to-peptide, baseline versus step 32 Recall@1/5/10 were
  0.08203/0.14063/0.17383 versus 0.08203/0.14063/0.17578, with MRR
  0.11893 versus 0.11896 and median rank 79 versus 79. For
  peptide-to-receptor, Recall@1/5/10 were 0.07617/0.12500/0.15039 for both,
  with MRR 0.10665 versus 0.10666 and median rank 115 versus 114.
- Ten-conformer full-candidate retrieval completed for conformer indices 0-9.
  Conformer-0 regression against the single-conformer output passed and the
  candidate bank matched. Across per-conformer results, baseline versus step
  32 mean Recall@10 was 0.18379 versus 0.18398 receptor-to-peptide and
  0.15508 versus 0.15469 peptide-to-receptor. Arithmetic-mean-score Recall@10
  was 0.19336 versus 0.19336 receptor-to-peptide and 0.15625 versus 0.15625
  peptide-to-receptor.
- The acceptance satisfies the minimum engineering conditions to consider
  formal Phase-3 fine-tuning. It does not establish a full useful-step window
  or authorize longer training by itself.

Evidence:

- `E:\pep\phase3\runs\drugclip\v3_bounded_step032_pilot_review_pending_v1`
- `E:\pep\phase3\runs\drugclip\v3_bounded_step032_full_retrieval_v1`
- `E:\pep\phase3\runs\drugclip\v3_bounded_step032_multi_conformer_v1`

## Formal Phase-3 v1 Training And Retrieval Evaluation

The single formal run at HEAD `a88f0c1f0` completed five full epochs with
`batch_size=16`, `lr=1e-6`, `tower_lr=2e-7`, AMP, and seed 1. Every epoch
recorded 19,707 unique training interface pairs in 1,232 batches, with zero
pair loss, pair duplication, or peptide-uniqueness violations. Global step is
6,160 and all 88 optimizer parameter states record step 6,160. Train loss fell
from 7.12005 to 5.31653 and fixed-validation loss from 6.96594 to 5.77681.

`checkpoint_best.pt` and `checkpoint_last.pt` both represent epoch 4 at step
6,160 and contain identical model tensors. The selected best checkpoint is
178,876,587 bytes with SHA256
`B334BE1FAFF2518F94ADBF3A89632FF6E102402CA5CE7F88E9B56328167BCC93`.

Evaluation used the bounded acceptance's fixed 512-query conformer-0 plan and
candidate banks. Plan, candidate, known-positive, baseline-regression, and
conformer-0 checks passed. Single-conformer formal versus Phase-2 baseline
Recall@10 was 0.18555 versus 0.17383 r2p, but 0.14844 versus 0.15039 p2r;
formal Recall@1 and MRR were lower in both directions. Across ten conformers,
mean Recall@10 was 0.19219 versus 0.18379 r2p and exactly 0.15508 versus
0.15508 p2r, while mean Recall@1 and MRR were lower in both directions.
Arithmetic-mean-score Recall@10 was 0.19922 versus 0.19336 r2p and 0.15234
versus 0.15625 p2r. Mean ranks improved in both directions, but the head-rank
metrics do not support a general bidirectional retrieval improvement.

Evidence:

- `E:\pep\phase3\runs\drugclip\v3_formal_full_finetune_5epoch_v1`
- `E:\pep\phase3\runs\drugclip\v3_formal_full_finetune_5epoch_v1_full_retrieval_best_v1`
- `E:\pep\phase3\runs\drugclip\v3_formal_full_finetune_5epoch_v1_multi_conformer_best_v1`

## Minimal Epoch-0 Model-Selection Recovery

One recovery run repeated the formal five-epoch command and changed only
`--stop_after_epoch 0`. It completed exactly one epoch and 1,232 optimizer
steps over all 19,707 training pairs, with zero pair loss, duplication, or
peptide-uniqueness violations. Its train/validation losses, sampling-plan SHA,
and directional losses exactly reproduce epoch 0 of the formal five-epoch
run. The trained epoch-0 `checkpoint_best.pt` is 178,874,731 bytes with SHA256
`5D2D326B634B38A4412950B08093F2152C974403A344AD8A7B2A59EAF8F33599`.

On fixed conformer 0, epoch 0 versus Phase-2 baseline Recall@1/5/10 and MRR
were 0.08398/0.14258/0.17773 and 0.12259 versus
0.08203/0.14063/0.17383 and 0.11893 r2p; p2r values were
0.07813/0.12891/0.15820 and 0.10894 versus
0.07617/0.12500/0.15039 and 0.10665. Across ten conformers, epoch-0 mean
Recall@10 was 0.18672 r2p and 0.15840 p2r, compared with baseline 0.18379 and
0.15508, bounded step 32 0.18398 and 0.15469, and formal epoch 4 0.19219 and
0.15508. Arithmetic-mean-score Recall@10 was 0.20117 r2p and 0.16016 p2r,
higher than all three comparison models in both directions.

The epoch-0 candidate is the best balanced model under this fixed evaluation
contract. Bootstrap intervals support mean-rank improvements, but the
Recall@10 difference intervals cross zero; do not reinterpret this bounded
model-selection result as universal statistical superiority.

Evidence:

- `E:\pep\phase3\runs\drugclip\v3_formal_epoch0_selection_recovery_v1`
- `E:\pep\phase3\runs\drugclip\v3_formal_epoch0_selection_recovery_full_retrieval_v1`
- `E:\pep\phase3\runs\drugclip\v3_formal_epoch0_selection_recovery_multi_conformer_v1`

## Phase-3 v1 Selected-Model Release Contract

The selected epoch-0 checkpoint is formalized as model version
`pepclip-phase3-v1-balanced-epoch0` by the machine-readable descriptor
`phase3/drugclip/releases/phase3_v1_selected_model.json`. The descriptor binds
the repository-relative checkpoint path and SHA256, `global_step=1232`, v3
Manifest SHA, frozen Phase-2 learned-concat initialization SHA, clean code
baseline, fixed single- and ten-conformer evaluation reports, and the bounded
conclusion language.

`python -m phase3.drugclip.validate_model_release` is a read-only release
validator. Against the retained formal artifacts it passed the checkpoint
byte/hash and `pepclip-phase3-drugclip-training-v1` schema checks, verified all
352 model tensors and 28,575,002 parameter/buffer elements with finite model
state, matched the `random_conformer_v3` Manifest and checkpoint data
contract, matched the Phase-2 initialization checkpoint, and verified five
recorded evaluation reports. Focused validator and training-state tests pass.
The checkpoint and all run/evaluation artifacts remain ignored outputs and
are not part of the tracked release contract.

## Fixed-512 Input-Domain Ablation

The read-only evaluator
`phase3/drugclip/evaluate_input_domain_ablation.py` reused the exact fixed
512-query plan, candidate banks, known-positive policy, v3 Manifest, Phase-2
checkpoint, and selected Phase-3 epoch-0 checkpoint. The input variants were:

- A: true-bound peptide all-heavy atoms.
- B: the exact residue-matched N/CA/C subset of the same A structure.
- C0: formal `random_conformer_v3` conformer 0.
- Cmean10: arithmetic mean of score matrices for conformers 0-9.

Input validation found 512/512 exact evidence matches, 512 existing structure
paths, zero peptide-sequence mismatches, zero missing structures, and zero A/B
subset failures. Four pairs contained one extra incomplete residue each; all
four residues were excluded only after the complete-residue sequence exactly
matched the expected peptide. The source distribution was BioLiP2 306,
BioLiP2_nr_peptide 29, Propedia26 29, and Q-BioLiP_PIII 148; the four excluded
residues were three BioLiP2 and one BioLiP2_nr_peptide. Two BioLiP2 pairs, one
residue each, required lossless source N/CA/C atom-order normalization. No A
input touched or exceeded the 192-atom cap.

The evaluation produced 24,576 per-query rank rows: two checkpoints, three
representations, four input variants, two directions, and 512 queries. All 60
pre-registered comparisons used the same paired query-resampling matrix with
seed 20260723 and 10,000 resamples. All scores were finite, every target was
present under the exact known-positive policy, both model-state hashes were
unchanged before versus after inference, the formal foreground command exited
0, stdout recorded PASS, and stderr was empty. The evaluator contains no
training, backward, optimizer, checkpoint write, or data/Manifest mutation
path.

The durable scientific interpretation is:

- A to B causes a large 3D-only and learned-fusion decline, showing that the
  removed full-heavy/side-chain input information is the primary observed
  missing factor under this contract.
- B to C0 is materially smaller. The Phase-3 v1 failure cannot primarily be
  assigned to randomizing the backbone itself.
- Backbone-only C0 and Cmean10 learned fusion remain below the corresponding
  1D-only result, so the deployed-input 3D branch does not provide reliable
  added value in this diagnostic.
- Epoch 0 does not materially repair the backbone-only 3D-only
  representation. Its small changes do not validate the backbone-only design.
- A uses target-bound coordinates and is an upper-bound diagnostic, not
  deployable screening performance.
- The next scientific step is a feasibility audit for chemically complete,
  candidate-independent random peptide conformers, not more Phase-3 v1
  training.

Evidence:

- `E:\pep\phase3\runs\drugclip\v3_input_domain_ablation_fixed512_v1`

## Full-Heavy Candidate-Independent Conformer Feasibility Audit

The 2026-07-23 read-only audit is classified `INSUFFICIENT_EVIDENCE` for a
fixed-512 all-heavy-random forward evaluation. This is not a model-contract
block: the selected Phase-2 3D tower is an EGNN whose existing element,
standard atom-name, and standard residue-name vocabularies cover N/CA/C/O,
OXT, and all standard side-chain heavy atoms. The generic atom tensor and
Phase-3 batching paths already accept those identities and coordinates, as
also demonstrated by the completed bound-all-heavy diagnostic. No model
architecture or vocabulary change is required.

The formal `random_conformer_v3` data contract itself remains strictly
backbone-only. Every conformer stores exactly three ordered atoms per residue
under `backbone_atoms`: N, CA, C. Its loader rejects any other atom count or
order, and its formal clash rule examines only N/CA/C. Therefore a future
prototype must use an independent schema/namespace; it must not overwrite or
reinterpret v3. A later formal release would require a new data schema and
contract, even though the current model can consume full-heavy tensors.

The current Python environment contains RDKit 2025.09.5 and Gemmi 0.7.5.
OpenMM, PDBFixer, PeptideBuilder, PyRosetta, Biopython, OpenBabel, MDTraj,
ParmEd, Modeller, Biotite, and MDAnalysis are unavailable, and no matching
local executable was found on PATH. Existing DrugCLIP/ProFSA source includes
generic RDKit ETKDG plus MMFF conformer helpers, but no peptide-specific
side-chain or rotamer builder.

An in-memory RDKit audit built all-heavy standard peptide topology for all 512
fixed-plan rows with zero failures and zero PepCLIP vocabulary mismatches;
heavy-atom counts were 44-175, below the 192-atom cap. A six-residue standard
peptide produced ten finite, distinct, seed-deterministic conformers with
stable atom identity/order and full MMFF parameters, although three of ten did
not converge within 500 MMFF iterations. A larger three-sequence,
ten-conformer, 1,000-iteration deterministic probe exceeded a five-minute
tool window and left no process behind. Thus topology compatibility is proved,
but robust fixed-512 ten-conformer generation throughput and convergence are
not.

The formal v3 sequence set contains 6,979 unique standard-alphabet peptides
of length 8-20. It contains 355 sequences with at least two cysteines; the
fixed-512 plan contains 31 such rows representing 22 unique sequences. The
metadata contains no explicit linear/cyclic, disulfide, modification,
crosslink, terminal-state, or other chemistry classification. Consequently,
multiple-Cys, cyclic, modified, covalent, and other chemistry-ambiguous
peptides cannot be silently treated as ordinary linear reduced peptides.
Only records with an explicit ordinary linear, unmodified, standard-residue,
defined-terminal chemistry contract are eligible for a first prototype.

RDKit `MolFromSequence` plus deterministic ETKDGv3, explicit-hydrogen
embedding, MMFF94s optimization, hydrogen removal, and strict convergence,
geometry, chirality, clash, identity/order, and repeatability rejection is the
single recommended minimal route. It is candidate-independent and requires no
network, but it has no explicit protein rotamer library and still needs a
separately authorized prototype to establish coverage and performance. No
prototype, data, training, or retrieval output was created by this audit.

## Full-Heavy Prototype Fixed-512 Chemistry And Length Panel

The subsequently authorized independent prototype is classified
`PERFORMANCE_BLOCKED`, not `FEASIBLE_WITH_EXISTING_LOCAL_STACK`. It did not
train, run GPU retrieval, publish a data version, overwrite
`random_conformer_v3`, or use target-bound coordinates for conformer
generation.

The read-only chemistry audit reproduced the exact 512-query plan and evidence
join, then inspected original residue names, adjacent peptide-bond geometry,
mmCIF/PDB connection records, S-S geometry, head-to-tail closure, and
peptide-receptor covalent geometry. Query classifications were 382
ordinary-linear-standard, 50 receptor-covalent, 45 modified/non-standard, 16
chemistry-insufficient, 13 known-disulfide, 5 unresolved multiple-Cys, and 1
cyclic/crosslinked. Because a sequence is eligible only when every fixed-plan
occurrence is ordinary, the conservative safe subset contains 373 queries and
265 unique sequences; 139/512 queries (27.1484%) are excluded. Safe-query
targets remain present in the unchanged 370-peptide/512-receptor candidate
banks.

The safe subset spans length 8-20 and theoretical heavy-atom counts 52-175.
No fixed-512 sequence reaches the 192-atom boundary, and the prototype never
truncates atoms. The known 197-heavy-atom sequence in the larger 6,979-sequence
set remains a future full-release atom-cap risk and was not generated here.

A deterministic nine-sequence panel covered the shortest, median, p75, p90,
p95, longest, maximum-heavy, closest-below-192, single-Cys, and three
composition-distinct length-13 criteria. With 25 fixed ETKDG attempts per
conformer, MMFF94s capped at 1,000 iterations, seed 20260723, and a 900-second
per-worker hard timeout:

- Six sequences of length 8-17 and 55-121 heavy atoms completed two independent
  10/10 runs. Atom-identity and canonical coordinate-set hashes matched
  exactly; every accepted conformer had MMFF status 0 and passed geometry,
  coordinate chirality, clash, and current EGNN CPU-forward checks.
- The length-19, 141-heavy-atom single-Cys sequence completed run 1 with 10/10
  in 525.06 seconds, but its identical run 2 exceeded 900 seconds. The process
  was killed and left no descendant or evaluator process.
- The p95 length-19/168-heavy and longest/max-heavy length-20/175-heavy
  sequences accepted 0 conformers. For conformer 0, all 25 fixed attempts
  embedded successfully but returned MMFF status 1 after 1,000 iterations.
  Structured evidence replays recorded all seeds, embedding/MMFF durations,
  rejection reasons, and zero accepted conformers.

The earlier 500-iteration prototype-v1 panel is retained as failed evidence.
One pre-registered diagnostic showed that 1,000 iterations can rescue a
length-13 case, so prototype-v2 used that single revised bound; no further
iteration, attempt, timeout, or parameter search was performed. Special
chemistry examples from every observed exclusion class were rejected before
generation. Structure coordinates were used only for eligibility
classification, never as generator input.

The permanent boundary is that ordinary linear short-peptide smoke succeeds,
while global ETKDGv3 followed by whole-molecule MMFF94s is blocked by
long-peptide convergence and runtime. This blocks the current generator
strategy, not the complete-heavy-atom representation itself, and it does not
authorize training or GPU retrieval. The sole next implementation action is a
separately authorized minimal prototype that starts from a fixed,
candidate-independent random peptide backbone and performs chemically local
side-chain completion without global whole-peptide ETKDG/MMFF optimization.

Evidence:

- `E:\pep\phase3\runs\drugclip\v2_full_atom_prototype_fixed512_audit_v1`
- `E:\pep\phase3\runs\drugclip\v2_full_atom_prototype_fixed512_audit_v2`

## Constrained Fixed-Backbone Side-Chain Completion Prototype

The formal-v3-input prototype is classified `PERFORMANCE_BLOCKED`, not
`CONSTRAINED_COMPLETION_PASS`. It used the unchanged
`random_conformer_v3` seed plan to regenerate all 10 candidate-independent
N/CA/C backbones for each of the five pre-registered sequences. All 50
regenerated backbone coordinate hashes matched the formal cache before
side-chain workers were launched.

The prototype built the full peptide topology with RDKit, supplied the formal
N/CA/C coordinates as an exact ETKDG coordinate map, fixed every N/CA/C atom
during MMFF94s optimization, and allowed only O/OXT, side-chain atoms, and
temporary hydrogens to move. It used at most 10 completion attempts, 500 MMFF
iterations, and a 300-second hard limit per sequence and repeat. It never read
receptor, interface, contact, evidence, or bound-coordinate data inside a
worker.

For the length-8, 55-heavy-atom control `SAVTTVVN`, two independent runs each
completed 10/10 accepted conformers in 4.56 and 4.73 seconds. The canonical
coordinate-set hash and atom-identity hash matched across repeats, all 10
conformers were distinct, maximum N/CA/C displacement was 0.0 angstrom, and
complete-heavy topology, MMFF convergence, geometry, chirality, clash,
vocabulary, and finite CPU EGNN-forward checks passed.

The remaining four sequences did not produce a first accepted conformer
within their first authorized 300-second run:

- `TLAPADGPTTDEVTLQV` (length 17, 121 heavy atoms)
- `KVSKAAADLMAYCEAHAKE` (length 19, 141 heavy atoms)
- `DDFTNELKAELDRYKRENQ` (length 19, 168 heavy atoms)
- `ENYFQAEAYNLDKVLDEFEQ` (length 20, 175 heavy atoms)

Each timed-out worker was terminated and left no child or evaluator process.
No timeout, attempt, or MMFF bound was increased, and no failed sequence was
retried. Because the formal N/CA/C hashes were reproduced before the workers
and the short control preserved them exactly, the evidence localizes the
remaining runtime block to the current whole-molecule RDKit constrained
completion/embedding route rather than formal-v3 backbone generation. It
does not establish long-peptide completion, determinism, geometry, or model
input success.

The earlier `v2_constrained_sidechain_completion_panel_v1` generated a new
backbone split and is retained only as invalid-input diagnostic evidence. The
decision-bearing run is
`v2_constrained_sidechain_completion_panel_v2`. All six non-ordinary chemistry
classes remain explicitly rejected. No training, GPU retrieval, full-data
generation, v4 publication, atom truncation, or target-bound conformer input
occurred. This result is specific to whole-molecule constrained ETKDG and does
not reject rotamer-based packing on an already fixed backbone.

Evidence:

- `E:\pep\phase3\runs\drugclip\v2_constrained_sidechain_completion_panel_v2\backbone_seed_plan.json`
- `E:\pep\phase3\runs\drugclip\v2_constrained_sidechain_completion_panel_v2\runs.json`
- `E:\pep\phase3\runs\drugclip\v2_constrained_sidechain_completion_panel_v2\summary.json`

## FASPR Fixed-Backbone Side-Chain Packing Prototype

The authorized official-source FASPR prototype is classified
`PACKING_COVERAGE_FAIL`, not `FASPR_FIXED_BACKBONE_PASS`. It did not train,
run GPU retrieval, generate a full dataset, publish a data version, or use an
online FASPR service.

FASPR was cloned only from
`https://github.com/tommyhuangthu/FASPR.git` at commit
`0d55732fd6307f373018c6bddd842291c355c5f7` into the external directory
`C:\Users\admin\.codex\tools\FASPR`. Its `LICENSE` is MIT. Existing Ubuntu
WSL g++ 13.3.0 compiled the source with the README command
`g++ -O3 --fast-math -o FASPR src/*.cpp`; no package was installed. The
external ELF binary SHA256 is
`EC5A10ACBDB97E377B0A6263CC4D94192A0E3F5D8189D8726C889C1BA935EFA3`.
The binary and `dun2010bbdep.bin` are in the same external tool directory.
Two official 1mol example runs were byte-identical, exited successfully, and
had empty stderr.

The panel orchestrator reproduced all 50/50 formal-v3 N/CA/C backbone hashes
before starting workers. Each backbone O was reconstructed deterministically
from local random-backbone geometry with a 1.231-angstrom C-O bond; the
C-terminal OXT was completed by a documented trigonal terminal contract.
FASPR received only N/CA/C/O peptide PDB records. Canonical output restored the
full-precision formal N/CA/C values after the unavoidable three-decimal PDB
round trip and required their coordinate hash to remain exact.

`SAVTTVVN` completed two independent 10/10 runs. Both runs had identical
atom-identity and coordinate-set hashes, ten distinct conformers, maximum
N/CA/C displacement 0.0 angstrom, FASPR exit code 0 for every conformer, and
complete-heavy, O/OXT, geometry, chirality, clash, vocabulary, and finite CPU
EGNN-forward PASS. Individual FASPR calls were below 0.54 seconds in the first
decision-bearing run.

The first length-17 control `TLAPADGPTTDEVTLQV` failed on formal conformer 0
despite FASPR exit code 0 and a reported 0.005-second packing computation.
The packed Pro8 ring placed the previous Gly C-Pro N-Pro CD angle at about
46.05 degrees; Pro4 showed the same incompatibility at about 57.75 degrees.
Both violate the pre-registered minimum 60-degree heavy-atom angle contract.
FASPR therefore removes the previous global-ETKDG long-peptide speed blocker
for the attempted length-17 control, but exposes a proline fixed-backbone
packing geometry failure instead. This is not an O/OXT reconstruction,
atom-cap, determinism, or model-input result.
The three remaining length-19-to-20 sequences were not run because the panel
stopped at the first failure. There were no timeouts or leftover FASPR
processes.

All six non-ordinary chemistry classes were rejected before generation.
Workers received no receptor, interface, contact, evidence, or bound
coordinates. The result shows that unmodified FASPR packing is not yet a
general fixed-512 generator for the formal random backbones; it does not
evaluate a proline-aware repair or another rotamer packer. Current evidence
cannot distinguish whether the formal-v3 random Pro backbone itself is
chemically incompatible with ring closure or whether this is a limitation of
FASPR on an extreme but potentially valid random backbone.

Evidence:

- `E:\pep\phase3\runs\drugclip\v2_faspr_fixed_backbone_panel_v1\faspr_tool_contract.json`
- `E:\pep\phase3\runs\drugclip\v2_faspr_fixed_backbone_panel_v1\runs.json`
- `E:\pep\phase3\runs\drugclip\v2_faspr_fixed_backbone_panel_v1\summary.json`

## Formal-v3 Proline Backbone Compatibility Audit

The authorized read-only audit classifies the tested formal backbone as
`FORMAL_V3_PROLINE_BACKBONE_INCOMPATIBLE`, resolving the ambiguity left by
the FASPR prototype. It did not modify the generator or packer, rerun the
stopped formal panel, train, run retrieval, or publish data.

`random_conformer_v3` reuses the generic internal-coordinate sampler.
Proline changes only the sampled phi term to `N(-65, 10)`; its psi remains
the current generic Ramachandran basin. Glycine changes only basin-switch
weights, not the basin means, standard deviations, or fixed backbone
geometry. There is no pre-Pro branch. Every peptide bond uses
`180 + N(0, 3)` degrees, so the contract is trans-like and has no cis-Pro
mode. The internal-coordinate placement maps the sampled Pro phi to the
opposite-signed standard coordinate dihedral.

All ten formal `TLAPADGPTTDEVTLQV` backbones were audited at Pro4 and Pro8.
The 20 sampled Pro phi values were negative, while all 20 standard
`Cprev-N-CA-C` coordinate dihedrals were positive. Bond lengths and the fixed
`Cprev-N-CA` and `N-CA-C` angles remained at their ideal generator values.
A fixed-backbone ring-only placement can preserve standard L-Pro ring bonds
and CA chirality, but it is not chemically sufficient because it permits a
pyramidal peptide amide N. After requiring the three angles about the Pro
peptide N to sum to within 10 degrees of 360, zero of 20 sites admitted any
of 200 independently generated, converged standard L-Pro templates. The best
site residuals were 20.18-37.03 degrees; all 200 standard controls were below
10 degrees.

The existing FASPR conformer-0 output preserved N/CA/C at PDB precision and
the residue-index/atom-name map was intact. Pro4 and Pro8 nevertheless had
illegal `Cprev-N-CD` angles of about 57.75 and 46.05 degrees. Reconstructed O
cannot change that angle because O is not one of its three defining atoms.
An independent candidate-only `AAPA` control exited FASPR with code 0 and
passed bond, angle, chirality, and clash QC, ruling out a universal inability
of the local FASPR binary to construct Pro.

Within the conservative 373-query ordinary-linear-standard subset, 168
queries and 118 unique sequences contain Pro; excluding all Pro would remove
45.04% of that subset. They contain 314 query-weighted Pro residues (218
across unique sequences), and 66 queries/49 unique sequences contain multiple
Pro residues. This is impact accounting only and is not authorization to
exclude Pro.

For Pro-inclusive Phase-3 v2 coverage, formal-v3 Pro backbone hashes cannot
be reused. A successor generator must be residue-aware, with Pro-specific
phi/psi, Gly-specific phi/psi, pre-Pro context, trans peptide bonds in its
first version, complete N/CA/C/O generation before packing, and a new
generator version and hash contract. Geometry QC must not be relaxed.

Evidence:

- `E:\pep\phase3\runs\drugclip\v3_proline_backbone_compatibility_audit_v1\summary.json`
- `E:\pep\phase3\runs\drugclip\v3_proline_backbone_compatibility_audit_v1\tlap_proline_numeric_audit.jsonl`
- `E:\pep\phase3\runs\drugclip\v3_proline_backbone_compatibility_audit_v1\ring_template_contract.json`
- `E:\pep\phase3\runs\drugclip\v3_proline_backbone_compatibility_audit_v1\safe373_proline_coverage.json`

## IDPConformerGenerator Read-Only Feasibility Audit

The official IDPConformerGenerator source was inspected read-only at commit
`e25a7d683b278532a3288f156edd8fba5f3a286c`, the dereferenced target of tag
`v0.8.2`. The clone is outside this repository at
`C:\Users\admin\.codex\tools\IDPConformerGenerator_audit`; no installation,
prepared database download, or conformer generation was run. The authoritative
repository `LICENSE` is Apache-2.0 with SHA256
`3DDF9BE5C28FE27DAD143A5DC76EEA25222AD1DD68934A047064E56ED2FA40C5`.
The stale `setup.py` license classifier says GPLv3+, so downstream packaging
must bind the copied Apache-2.0 license and commit rather than trust that
classifier.

The backbone algorithm is residue-aware in a useful but database-dependent
sense. It searches exact sequence fragments of length 1-5 with default relative
weights 1/1/3/3/2, samples contiguous empirical omega/phi/psi triples, and
falls back by shortening an unavailable fragment down to one residue. Pro and
Gly therefore use residue-matched records. A separate `_P` lookup requires the
sampled fragment to be followed by Pro, providing pre-Pro context. Omega is
sampled from the database with no built-in trans-only filter; a first-version
trans-only contract would have to reject or split source segments containing a
cis-like bond before database alignment. `--random-seed` seeds NumPy, while
worker seeds are derived as base seed plus worker index. `--nconfs 10` requests
ten outputs but does not make ten accepted conformers an atomic guarantee, so
a future wrapper must use one core, validate the output count, and bind each
conformer to a stable sequence/index seed namespace.

The generator can build N/CA/C/O and terminal carboxyl coordinates and write
standard PDB atom, residue, element, chain, and residue-number fields.
Residue-specific fixed bond lengths and angles are available without Int2Cart.
The first Phase-3 route can therefore use backbone-only output plus the already
compiled external FASPR binary; MCSCE, TensorFlow, Int2Cart, CUDA, and GPU are
not scientifically required. IDPConfGen's own FASPR integration instead uses a
compiled `idpcpp` extension and bundled `dun2010bbdep.bin`. The documented
wrong-stereochemistry bug affected `ldrs` coordinate alignment before v0.7.17
and was fixed in v0.7.17; the audited version is v0.8.2.

The decisive blocker is torsion-database provenance. `sscalc` records a
filename-derived PDB/chain/segment key plus `dssp`, `fasta`, and `resids`;
`torsions` adds phi/psi/omega. It does not record experimental method,
resolution, original source path, source hash, or an exclusion manifest.
`aligndb` temporarily creates a source-key-to-array-slice map, but `build`
discards that map before fragment sampling, so a generated fragment is not
traceable to a PDB/chain at runtime. The official prepared database is described
only as a 27,425-chain PISCES snapshot. The source repository contains no
manifest proving removal of fixed-512 evidence structures, all Phase-3
biological evidence, Phase-2/Phase-3 validation or test structures, exact
evaluation peptides, or highly similar peptide-length fragments. It is
therefore forbidden for PepCLIP evaluation.

The current local environment cannot yet build a replacement database under
the authorized no-install/no-download boundary. Windows has Python 3.13.5 but
no `pybind11`, `tox`, `idpconfgen`, compiler, or DSSP. WSL2 Ubuntu has Python
3.12.3 and g++ 13.3.0, but none of the required Python packages; the maintained
environment contract is Python >=3.8,<3.12. WSL has DSSP 4.2.2, whereas the
project explicitly supports only DSSP 2 or 3 for database construction. The
8,657 local targeted mmCIF files are Phase-3 evidence and belong on the
exclusion list; the local PDB snapshot has sequence indexes but no coordinate
corpus. No manifest-qualified, non-overlapping local PDB corpus was found.
Consequently a sanitized database would require a separately authorized,
pinned offline coordinate corpus and a compatible Python/DSSP environment (or
new downloads/installations).

PepCLIP compatibility is not the blocker. The current tensor path consumes
coordinates, element, atom name, and residue name; the safe 373-query
ordinary-linear-standard subset has 52-175 theoretical heavy atoms, below the
192-atom cap. A future prototype must filter IDPConfGen output to N/CA/C/O,
run external FASPR, add/validate OXT, restore canonical atom order, reject all
existing special-chemistry classes, and rerun geometry/chirality/clash,
vocabulary, atom-cap, and CPU-EGNN checks.

The sole audit classification is `TORSION_DATABASE_PROVENANCE_BLOCKED`.
IDPConformerGenerator is a plausible residue-aware candidate-only backbone
engine, but it is not currently an evaluation-safe generator. The minimum
unblock is a pre-sampling sanitized database whose manifest pins every source
PDB/chain and SHA256, method/resolution metadata, the generator and DSSP
versions, all project PDB/sequence exclusion sets and their hashes, an
ungapped peptide-length similarity exclusion rule (at least 80% identity over
at least 8 residues), trans-only segment filtering, deterministic build
commands, and final JSON hashes. The exclusion must happen before `aligndb`;
post-hoc runtime filtering is not auditable after source mappings are
discarded. No five-sequence prototype is authorized or specified until that
database and compatible local environment exist.

## Train-Only Torsion And Deterministic QC Rejection-Sampling Prototype

The separately authorized local alternative completed its source audit before
generation. It used only formal Phase-3 train-split evidence structures and
excluded validation/test/fixed-512 PDBs and structure files, exact evaluation
sequences, source 8-mers with at least 80% ungapped identity to any evaluation
8-mer, and non-ordinary chemistry. No receptor coordinate, interface, contact,
or bound pose is a generation input.

The audit began with 19,707 train pairs, 5,020 train-unique peptide sequences,
13,620 candidate train sources, and 7,297 candidate PDB IDs. It resolved every
train evidence ID. The exclusive filters removed 2,863 sources for chemistry
and 1,163 for the evaluation-window rule; split/fixed-512 PDB overlap and exact
evaluation-sequence overlap were zero. The accepted prior source contains
9,594 peptide sources, 5,050 PDB IDs, 6,097 structure files, and 3,461 unique
peptide sequences.

After filtering 397 non-trans observations, the joint residue-context prior
contains 95,345 trans phi/psi/omega observations. Pro, Gly, and pre-Pro have
6,360, 5,973, and 6,428 observations respectively, all above the required 500.
Every observation retains PDB, chain, residue, source path, and source-file
SHA256. The canonical prior manifest SHA256 is
`E93B24E59D5C18D7CC4213BC82D38C789CB32A279A3078AED738477246E80F94`.
Sampling is joint in phi/psi/omega and distinguishes Pro, Gly, residue-specific
pre-Pro, and all other residue identities. The torsion prior and its manifest
remain byte-for-byte unchanged.

The new internal-coordinate implementation fixes the formal-v3 sign
convention defect: sampled and coordinate-recomputed phi/psi/omega agree to
less than `1.14e-13` degrees in the executed panel. It directly creates
N/CA/C/O, is trans-only, and uses a new generator/hash namespace rather than
claiming formal-v3 backbone reproduction.

The authorized extension is classified
`DETERMINISTIC_REJECTION_SAMPLING_PASS`. It uses logical conformer slots 0-9
and attempt indices 0-24. Each attempt seed hashes the new generator version,
unchanged prior-manifest SHA, peptide sequence, logical conformer index, and
attempt index. One seed produces exactly one complete backbone sample. Each
distinct backbone is sent to deterministic FASPR at most once; a failed packed
geometry advances to the next attempt rather than rerunning FASPR on the same
backbone. A slot is occupied only after the unchanged geometry, chirality,
Pro-planarity, atom identity/vocabulary/cap, finite-coordinate, and single-
conformer CPU-EGNN checks all pass.

All five fixed panel sequences completed two independent 10/10 runs. The two
runs for every sequence have identical accepted-attempt vectors,
atom-identity hashes, coordinate-set hashes, and byte-identical deterministic
rejection logs. The complete per-attempt logs separately retain elapsed time.
Across each one of the two panel passes there were 59 attempts, 50 accepts,
and 9 strict-QC rejections; the maximum accepted attempt index was 3. All 18
double-run rejections were nonlocal heavy-atom clashes from 0.289988 to
0.666087 angstrom and remained rejected under the unchanged 0.75-angstrom
threshold. The previous 0.325159-angstrom clash remains invalid under the same
contract. Thus rejection sampling in this executed panel was driven only by
candidate-independent chemical/geometry QC; atom-schema, vocabulary, cap, and
CPU-EGNN compatibility checks all passed and caused zero retries. No receptor,
contact, retrieval score, or model-performance signal selected an attempt.

The maximum one-run peptide time was 16.081 seconds including first-use tool
startup, far below 300 seconds. Accepted atom counts were 55, 121, 141, 168,
and 175, below the 192 cap. Maximum sampled-versus-measured torsion error was
`1.71e-13` degrees. The Pro control passed all Pro ring/bond/chirality and
amide-nitrogen planarity checks with a maximum planarity residual of 1.279
degrees. Every accepted conformer had FASPR exit code 0, finite full-heavy
coordinates, zero fixed-backbone movement, no vocabulary unknown, ten unique
coordinates per peptide, and finite per-attempt and ten-conformer CPU EGNN
forward. All six special-chemistry classes remain explicitly rejected, no
target-bound input was used, and no process remained.

This five-sequence PASS proves the bounded deterministic rejection-sampling
prototype contract, not fixed-512 readiness or retrieval benefit. It does not
prove coverage for all 265 unique sequences in the 373-query safe subset. It
does not authorize GPU retrieval, training, a new data release, or any
relaxation of clash or geometry QC.

The subsequent authorized safe-candidate CPU coverage run originally stopped
with `SAFE265_GENERATION_COVERAGE_FAIL` under the then-current QC
implementation. Before generation, the fixed-512 chemistry audit was
re-derived as 370 unique peptide candidates: 265
`ordinary_linear_standard` sequences and 105 explicit exclusions
(`chemistry_insufficient` 15, `cyclic_or_crosslinked` 1,
`known_disulfide` 9, `modified_or_nonstandard` 38,
`multiple_cys_unknown` 4, and `receptor_covalent` 38). All 373 safe-query
targets are present in the safe265 sequence set. This establishes the input
boundary only; it does not give all 370 candidates full-atom coordinates.

Generation stopped at the first exhausted logical slot, as required. The first
two sequences in canonical order, `AAAPAGAAAA` and `AAASLYEKKAA`, completed
10/10 and were atomically cached. The next sequence, `AERKRILPTWML`, obtained
zero acceptable structures for logical conformer slot 0 across the fixed 25
attempts. Every attempt reached FASPR output and was rejected by unchanged
geometry QC as `illegal_heavy_bond_length_range`; the reported maximum heavy
bond length ranged from 3.028067 to 3.125133 angstrom. No attempt was accepted,
the attempt limit and QC were not changed, no whole-audit retry or independent
second pass was run, and no evaluator/training process remained.

The output directory is an intentionally incomplete prototype failure scene,
not a cache release. It contains the canonical candidate and explicit-rejection
records, two completed sequence caches, all attempted FASPR work files, and
`build_failure.json` plus a `generation_progress.json` whose status is `FAIL`.
It has no final cache manifest and no validation report. Generation seeds and
the generator API remain limited to generator version, prior-manifest SHA,
peptide sequence, conformer index, and attempt index; query, receptor,
evidence, contact, and bound coordinates are absent.

A separately authorized diagnosis retained that original failure scene and
did not invoke FASPR or sample a new backbone. It established that all 25
reported 3.028067-3.125133-angstrom “bonds” were ILE6 `CG2-CD1`. RDKit
2025.03 gives `MolFromSequence` the standard ILE atom names but connects CD1
to CG2. FASPR source and all saved coordinates follow standard PDB ILE
topology: CB connects CG1 and CG2, and CG1 connects CD1; saved `CG1-CD1`
distances are 1.515531-1.516706 angstrom. The same erroneous RDKit graph also
made the legacy coordinate chirality path label standard FASPR ILE6 CB as R
instead of its RDKit-expected S.

The repaired QC uses explicit standard-PDB heavy-atom templates for all 20
standard amino acids, explicit inter-residue `C-N` peptide bonds, and an
explicit terminal `C-OXT` bond. Bond lengths, bond angles, and graph-distance
greater-than-two clash exclusions use only this canonical graph. Chirality is
checked independently for every non-Gly L-CA plus ILE CB and THR CB; the
existing Pro ring and peptide-amide-nitrogen planarity check remains separate.
The 20-residue comparison found matching RDKit and canonical graphs for 19
residues and only the ILE `CG1/CD1` versus `CG2/CD1` mismatch; FASPR atom names
match the canonical standard names for all 20.

Read-only replay of the 25 saved FASPR PDBs passed 24 and retained one true
rejection: attempt 13 has a 0.629184-angstrom nonlocal clash between residue 1
CB and residue 9 OG1, below the unchanged 0.75-angstrom limit. The largest
correct covalent bond among passing replays is 1.808035 angstrom; the closest
nonlocal distance among passing replays is 0.750387 angstrom. All replayed
passing conformers have valid L-CA, ILE/THR CB chirality, Pro planarity
(maximum residual 0.859 degrees), no vocabulary unknowns, and finite batched
CPU EGNN output. Therefore slot 0 had 24 acceptable saved attempts and was not
actually exhausted.

Withdraw `SAFE265_GENERATION_COVERAGE_FAIL` as the scientific coverage
classification. The original run still stopped correctly under its
implemented contract and remains retained as historical diagnostic evidence.
At the end of the topology-only diagnosis the interim classification was
`QC_TOPOLOGY_MAPPING_FAIL / SAFE265_COVERAGE_INCONCLUSIVE`.

A subsequently authorized build then started from HEAD `3867f38d9`, used the
repaired canonical topology, and wrote only to the new
`v3_safe265_full_atom_conformer_coverage_prototype_v2` directory. The
historical v1 failure file remains byte-identical with SHA256
`E2F82024919B33E2330058F55CFA4CD43E4937857454EEEDF1F21DBA563EC6BD`;
neither its two caches nor any saved v1 attempt was reused.

The v2 first pass generated 10/10 accepted conformers for all 265 safe unique
sequences: 2,650 conformers from 2,933 total attempts, with 283 strict-QC
rejections. The largest attempt index used was 4; no slot exhausted the
25-attempt limit. Rejections comprise 270 nonlocal clashes below the unchanged
0.75-angstrom threshold and 13 Pro amide-planarity failures. The maximum
per-sequence generation time was 3.976 seconds and the generation wall time
was 408.800 seconds, both within the fixed 300-second per-sequence contract.

Every accepted conformer passed the standard-PDB 20-residue topology contract,
sampled/measured torsion agreement, FASPR exit 0, complete atom identity,
canonical ordering, bond/angle/chirality, Pro, nonlocal-clash, vocabulary,
atom-cap, fixed-backbone, finite-coordinate, and CPU-EGNN checks. All 2,650
backbone deviations are zero, the maximum atom count is 175, the minimum
accepted nonlocal distance is 0.758091 angstrom, and each sequence has ten
distinct coordinate hashes. Generation uses no query, receptor, evidence,
contact, or bound-pose input.

The independent second pass regenerated all 265 sequences in temporary
directories and retained no second coordinate cache. It matched all 2,650
accepted attempt indices, rejection sequences, atom identities, backbone
hashes, coordinate hashes, FASPR hashes, per-sequence aggregate hashes, and
the global deterministic semantic manifest SHA256
`E213B177A98E484BAC9F3516B899EB0FF167B6A7A28A97FF0998F48C1B1C84F8`.
The slowest full regeneration plus validation was 89.037 seconds, below 300
seconds. The durable classification is therefore
`SAFE265_FULL_ATOM_CACHE_PASS`.

This is a CPU prototype cache and coverage proof, not a formal data release or
retrieval result. It covers 265/370 peptide candidates; the remaining 105
special-chemistry candidates retain their explicit rejection classifications
and have no fabricated coordinates. All 373/373 safe-query targets occur in
the safe265 set. Any future receptor-to-peptide evaluation must use a new
265-peptide bank and recompute every baseline. Peptide-to-receptor may retain
the 512-receptor bank, but must recompute baselines on the exact 373-query safe
contract. Neither result is directly comparable with the old 512-query /
370-peptide-bank metrics.

Evidence:

- `E:\pep\phase3\runs\drugclip\v3_train_only_torsion_prior_prototype_v1\train_source_audit_summary.json`
- `E:\pep\phase3\runs\drugclip\v3_train_only_torsion_prior_prototype_v1\torsion_prior_manifest.json`
- `E:\pep\phase3\runs\drugclip\v3_train_only_torsion_rejection_sampling_prototype_v1\train_only_torsion_panel_summary.json`
- `E:\pep\phase3\runs\drugclip\v3_train_only_torsion_rejection_sampling_prototype_v1\panel.log`
- `E:\pep\phase3\runs\drugclip\v3_safe265_full_atom_conformer_coverage_prototype_v1\build_failure.json`
- `E:\pep\phase3\runs\drugclip\v3_safe265_full_atom_conformer_coverage_prototype_v1\generation_progress.json`
- `E:\pep\phase3\runs\drugclip\v3_safe265_full_atom_topology_qc_replay_v1\topology_qc_replay_report.json`
- `E:\pep\phase3\runs\drugclip\v3_safe265_full_atom_conformer_coverage_prototype_v2\cache_manifest.json`
- `E:\pep\phase3\runs\drugclip\v3_safe265_full_atom_conformer_coverage_prototype_v2\deterministic_generation_manifest.json`
- `E:\pep\phase3\runs\drugclip\v3_safe265_full_atom_conformer_coverage_prototype_v2\validation_report.json`

## Safe373 Frozen-Model Full-Atom Retrieval Evaluator Preflight

The independent read-only evaluator
`phase3/drugclip/evaluate_safe_full_atom_retrieval.py` is implemented without
changing the prior fixed-512 input-domain evaluator. It derives a safe373 plan
by filtering the original fixed-512 plan while preserving query-relative
order. The plan contains 373 queries, a lexicographically ordered 265-peptide
r2p bank, and the original lexicographically ordered 512-receptor p2r bank.
Its canonical SHA256 is
`A32FF671CFEA0D1B858C8EFC58AD0E30D6F3170C670089238127B637FCC64310`;
it binds the original plan file/canonical hashes, chemistry-audit hash, query
IDs, both candidate-bank orders, and the bank-intersected known-positive
policy.

The CPU contract preflight is
`SAFE373_FULL_ATOM_RETRIEVAL_PREFLIGHT_PASS`. It read all 265 sequence caches
and 2,650 full-heavy conformers, verified the required cache, deterministic,
and validation manifest hashes, and found zero vocabulary unknowns. A/B were
constructed by running the unchanged exact-evidence selection over all 512
queries first and only then filtering safe IDs. Thus safe373 has 373/373 exact
evidence matches, zero sequence mismatches, zero target missing, and zero A/B
subset failures without changing the original candidate-representative or
bound-structure selection logic. A remains a target-bound, non-deployable
diagnostic upper bound.

Both allowed frozen models were loaded on CPU. Phase-2 checkpoint SHA256 is
`9FB16C48BA715C6273341609D60725AE796AD4A78771744E19ECF2C13D38AE20`;
selected Phase-3 epoch-0 SHA256 is
`5D2D326B634B38A4412950B08093F2152C974403A344AD8A7B2A59EAF8F33599`.
All model state was finite and each model's state hash was identical before
and after a two-query forward smoke. D0 produced the required 2x2 score
matrix, and Dmean10 exactly matched the arithmetic mean of ten independently
scored 2x2 matrices; coordinates were not averaged.

The evaluator emits 1D-only, 3D-only, and learned-fusion metrics for A, B, C0,
Cmean10, D0, and Dmean10 as applicable; per-query target scores/ranks,
known-positive exclusions, target-missing audit, rank dispersion/worst rank,
and the preregistered paired comparisons with 10,000-query bootstraps. No
training, backward, optimizer, formal GPU evaluation, checkpoint mutation, or
retrieval result was produced in this work unit.

Evidence:

- `E:\pep\phase3\runs\drugclip\v3_safe373_full_atom_retrieval_preflight_v1\safe373_evaluation_plan.json`
- `E:\pep\phase3\runs\drugclip\v3_safe373_full_atom_retrieval_preflight_v1\input_variant_audit.jsonl`
- `E:\pep\phase3\runs\drugclip\v3_safe373_full_atom_retrieval_preflight_v1\preflight_report.json`

## Formal Safe373 Frozen-Model Full-Atom Retrieval

The single authorized frozen-model GPU evaluation completed at detached HEAD
`02e160449c1ec1c2324c01deef4d8f6748ca329b` with exit code 0. It used plan
SHA256 `A32FF671CFEA0D1B858C8EFC58AD0E30D6F3170C670089238127B637FCC64310`,
373 queries, the 265-peptide r2p bank, and the 512-receptor p2r bank. The
safe265 cache supplied 265/265 sequences and 2,650/2,650 conformers. Exact
A/B evidence and sequence coverage were 373/373, target missing was zero,
and no A input was truncated. The output contains 19,396 per-query rank rows
and all 40 preregistered paired-bootstrap groups at seed 20260724 and 10,000
resamples. All metrics are finite. Both checkpoint files and their in-memory
model-state hashes were unchanged before and after evaluation.

Full-heavy input restores substantial frozen 3D-only retrieval signal relative
to backbone-only input. For Dmean10 minus Cmean10, Recall@10 improvements were
0.13405 r2p and 0.14209 p2r for the Phase-2 model, and 0.13137 r2p and
0.14477 p2r for Phase-3 epoch 0; every 95% paired interval excludes zero.
MRR and mean rank support the same 3D-only conclusion in both directions.
D0 minus C0 gives the same qualitative result, so the finding is not dependent
on ten-conformer averaging.

Learned fusion also improves from Cmean10 to Dmean10 in both directions and
both frozen models for Recall@10 and MRR, with all corresponding 95% intervals
excluding zero. However, p2r mean rank worsens by 11.43 for Phase 2 and 12.51
for Phase-3 epoch 0, while r2p mean-rank intervals cross zero. Relative to
1D-only, learned-fusion Dmean10 changes Recall@10 by only 0.00536/0.00804
(r2p/p2r) for Phase 2 and 0.01072/0.00268 for Phase-3 epoch 0; all four
Recall@10 intervals cross zero. MRR improves significantly in Phase-2 p2r and
both Phase-3 directions, but the full metric set does not establish a stable
overall learned-fusion advantage over 1D-only.

Ten-conformer Dmean10 does not show a general paired advantage over D0:
Recall@10 and mean-rank intervals cross zero for both models, representations,
and directions; only Phase-3 3D-only p2r MRR excludes zero. Bound-heavy A
remains a target-bound, non-deployable diagnostic upper bound. Dmean10
learned fusion retains worse mean rank than A in both directions and both
models; Recall@10 gaps are mixed in significance but range from 0.01341 to
0.03485 in magnitude.

Phase-3 epoch 0 is not generally superior to the Phase-2 baseline on
full-heavy random input. Their Dmean10 Recall@10 differences are at most
0.00536 in magnitude and all paired intervals cross zero. Phase-3 has a small
significant p2r 3D-only MRR gain and a small significant r2p learned-fusion
mean-rank gain, but the remaining primary comparisons are mixed or
indistinguishable.

This result provides a scientific basis to consider a separately authorized,
strictly bounded Phase-3 v2 adaptation experiment: candidate-independent
full-heavy input has restored useful frozen 3D signal, while the existing
learned fusion does not reliably exceed 1D-only. It is not evidence that
fine-tuning will succeed, does not authorize training, and is not directly
comparable with the old 512-query / 370-peptide-bank evaluation.

Evidence:

- `E:\pep\phase3\runs\drugclip\v3_safe373_full_atom_retrieval_v1\retrieval_metrics.json`
- `E:\pep\phase3\runs\drugclip\v3_safe373_full_atom_retrieval_v1\bootstrap_confidence_intervals.json`
- `E:\pep\phase3\runs\drugclip\v3_safe373_full_atom_retrieval_v1\per_query_ranks.jsonl`
- `E:\pep\phase3\runs\drugclip\v3_safe373_full_atom_retrieval_v1\checkpoint_audit.json`
- `E:\pep\phase3\runs\drugclip\v3_safe373_full_atom_retrieval_v1\stdout_stderr.log`

## Phase-3 v2 Bounded Full-Heavy Adaptation Contract

The existing Phase-3 runner now has an opt-in
`--full-heavy-adaptation-manifest` path. It does not alter the model
architecture, temperature, bidirectional known-positive loss, batching, or
masking. The mode freezes receptor 1D, receptor 3D, receptor fusion, and
peptide 1D; it trains only the peptide EGNN final block, peptide
`final_norm`, peptide `project`, and peptide fusion. Against the registered
Phase-2 checkpoint state this is 26 parameter tensors and 2,843,265 scalar
parameters: 2,448,001 peptide-3D and 395,264 peptide-fusion parameters. The
canonical trainable-name SHA256 is
`ACC0E5C1AC2FC5EA2C27DC559795B55BD5FCD1D351821529F72B4D9AC6414774`.

The manifest validator requires the Phase-2 learned-concat initialization SHA
`9FB16C48BA715C6273341609D60725AE796AD4A78771744E19ECF2C13D38AE20`;
Phase-3 epoch 0 is not accepted as initialization. It binds exact bounded
train/valid interface-pair lists and hashes, limits them to at most 4,096/512,
requires disjoint formal split IDs, and validates that every required unique
peptide has exactly ten full-heavy conformers. Cache validation requires
`ordinary_linear_standard`, canonical standard-PDB atom identity/order,
complete standard-residue topology, finite coordinates, no vocabulary UNK,
strict atom count below 192, distinct conformers, accepted FASPR/QC/CPU-EGNN
records, and the unchanged 25-attempt and 0.75-angstrom contracts. Generation
seed inputs are restricted to generator/prior identity, sequence, conformer
index, and attempt index; receptor, contact, interface, evidence, and bound
coordinates are forbidden. The safe373 evaluation cache is explicitly
invalid as training input.

Checkpoint save/resume now embeds and compares both the full-heavy data
contract and the exact freeze contract. A synthetic CPU optimizer-step audit
changed only allowed peptide-3D/peptide-fusion parameters; receptor 1D,
receptor 3D, receptor fusion, and peptide 1D weights and recomputed embeddings
were bitwise unchanged. A deliberately changed receptor-fusion parameter was
correctly rejected. Focused contract tests pass 9/9, existing training-state
and bounded-runner tests pass 6/6, Phase-3 tests pass 187/187, `py_compile`,
runner `--help`, and `git diff --check` pass. No formal cache, optimizer
trajectory, training run, GPU retrieval, or parameter search was executed.

During final environment checking, the prior detached-worktree cleanup was
found to have followed Windows junctions and emptied the Phase-2 learned-concat
run directory and `E:\pep\models`. The complete Phase-2 run directory was
restored from an existing local nested copy; every restored file matches its
source SHA and `checkpoint_best.pt` again has the registered SHA above.
The exact `E:\pep\models\esm2_t6_8M_UR50D` asset was subsequently restored
from a trusted server copy. Its Hugging Face metadata binds revision
`c731040fcd8d73dceaa04b0a8e6329b345b0f5df`, and a strict-offline real-model
load now passes: 352 state tensors, 28,575,002 elements, all finite, unchanged
state hash, and the expected 26-tensor / 2,843,265-parameter trainable set with
name SHA
`ACC0E5C1AC2FC5EA2C27DC559795B55BD5FCD1D351821529F72B4D9AC6414774`.
No optimizer, backward, or training path was executed.

## Phase-3 v2 Real Bounded-Plan Preflight

The read-only audit at
`E:\pep\phase3\runs\drugclip\v2_bounded_full_heavy_plan_preflight_v1`
loaded the actual formal-v3 runner Dataset contract and reproduced 19,707
train and 2,463 valid interface pairs. The current runner selections are
exactly `sorted(train_base.interface_pair_ids)[:4096]` and
`sorted(valid_base.interface_pair_ids)[:512]`, yielding 2,101 train and 370
valid unique peptide sequences, with no train/valid interface-pair,
biological-relation, or peptide-sequence overlap. Ordered pair-list SHA256
values are
`1802DB4F1148101E209AF9ED1DA3E1E7573B359A5BDC4FC899326262768A890B`
for train and
`6BF3206C391BEB590D2C9ED033D947E489CFBBEC5219D33B0F0383AA7D466BE4`
for valid.

The exact-evidence chemistry classifier rejects 1,110 of the 4,608 prefix
pairs, so the durable classification is `PLAN_CONTRACT_FAIL`. Train contains
3,125 ordinary and 971 non-ordinary pairs; valid contains the expected 373
ordinary safe queries and 139 non-ordinary pairs. Across both prefixes the
non-ordinary counts are 489 modified/non-standard, 441 receptor-covalent, 92
known-disulfide, 55 chemistry-insufficient, 19 cyclic/crosslinked, and 14
multiple-Cys-unknown pairs. All 2,471 union sequences have train-only
torsion-prior context coverage, but the maximum theoretical heavy-atom count
is 197; `WASLWNWFDITNWLWYIRKK` also violates the strict `<192` atom-cap
contract. A future separately authorized cache would require 21,010 train and
3,700 valid conformers, or 24,710 across the disjoint sequence union.

The train prefix has zero interface-pair, biological-relation, or peptide-
sequence overlap with safe373 evaluation queries/candidates, and no safe373
cache path or coordinate was reused. No cache was generated. Evidence files
are `bounded_pair_plan.json`, `chemistry_eligibility_audit.jsonl`,
`bounded_plan_preflight_report.json`, `esm_real_model_preflight.json`, and
`summary.md` in the audit directory.

## Verified Historical Phase-3 Diagnosis

The existing 4096/512 Pilot and early-step diagnostics used data version v2.
They are historical diagnostic evidence only, not a v3 runner and not
authorization to continue v2 training.

- Zero-update regression: two independent model objects reproduced embeddings,
  score matrices, complete candidate orders, target ranks, and aggregate
  metrics exactly. The 12 missing and 2 unexpected `inv_freq` keys are a
  fixed-buffer schema difference; classification is `KEY_ONLY_FAIL`, while
  model behavior is PASS.
- First optimizer step: the frozen Phase-2 reference stayed bitwise unchanged;
  all expected modules changed; no frozen/temperature parameter changed; no
  optimizer parameter was missing or unexpected; no NaN/Inf was found.
- Step 1 did not break fixed 512-query, ten-conformer retrieval. Recall@1,
  Recall@10, and median rank were unchanged in both directions; 14 r2p and 28
  p2r target ranks changed, with no Top-10 entries or exits.
- The v2 useful window peaked as a balanced head-metric candidate at step 32:
  r2p Recall@10 0.19922 and p2r Recall@10 0.15820. At step 64 these fell to
  0.19336 and 0.15625, and p2r MRR fell from 0.11457 to 0.11375. The durable
  classification is `SHORT_USEFUL_WINDOW`, with the first regression at step
  64.
- The first-step audit saved step 0 and step 1 checkpoints plus retrieval JSON
  through step 64. It did not save a step-32 checkpoint. A separate historical
  v2 epoch-strength diagnostic has step checkpoints, but it is not a formal v3
  bounded Pilot.
- `phase3/drugclip/audit_first_optimizer_step.py` hard-codes the historical v2
  audit contract and is not the runner for a formal v3 bounded Pilot.

Evidence:

- `E:\pep\phase3\runs\drugclip\pilot_interface_pair_4096_512_v1\zero_update_regression\KEY_VS_BEHAVIOR_REPORT.md`
- `E:\pep\phase3\runs\drugclip\pilot_interface_pair_4096_512_v1\first_optimizer_step_audit\audit_summary.json`
- `E:\pep\phase3\runs\drugclip\pilot_interface_pair_4096_512_v1\first_optimizer_step_audit\retrieval_step032.json`
- `E:\pep\phase3\runs\drugclip\pilot_interface_pair_4096_512_v1\first_optimizer_step_audit\retrieval_step064.json`

## Incomplete Or Superseded Work

- `lr5e5_epoch0_diagnostic` is incomplete: it has checkpoints only through
  step 128 and retrieval rows through step 160, with no completed trajectory
  or final selection. It cannot support a learning-rate conclusion and must
  not be resumed as the next experiment.
- The proposed lower-LR B comparison was never started.
- The completed five-epoch v2 Pilot and its ten-conformer evaluations remain
  useful history but do not replace a formal v3 bounded run.
- The original Phase-2 Conda environment and exact package versions were not
  preserved and must remain unknown; do not guess or reconstruct them from the
  model config.

## Current Problem

Formal v3 fine-tuning, selected-model release, fixed-512 input-domain ablation,
canonical-topology safe265 generation, and the authorized safe373 frozen-model
GPU evaluation are complete. Full-heavy random input clearly restores frozen
3D-only signal relative to backbone-only input, but the learned-fusion result
does not establish a stable overall advantage over 1D-only, ten-conformer
averaging is not generally superior to D0, and neither frozen checkpoint is
clearly best. The safe265 prototype still excludes 105 special-chemistry
candidates and is not a formal data release. The bounded full-heavy adaptation
contract implementation, synthetic one-step audit, and strict-offline
real-model preflight pass. The current hard-coded 4,096/512 sorted prefix
remains invalid, but the subsequent full formal-split sequence-level audit is
`CORE_LINEAR_SUBSET_SUFFICIENT`: 13,831/19,707 train pairs (3,337 sequences;
5,988 biological relations) and 1,711/2,463 valid pairs (663 sequences; 826
relations) are conservatively ordinary-linear, strictly below 192 heavy atoms,
and covered by the frozen torsion prior. Combined immediate coverage is
15,542/22,170 pairs (70.1037%). The ordinary chemical class contains one
additional 197-heavy-atom sequence, `WASLWNWFDITNWLWYIRKK`; it is excluded
rather than truncated.

The excluded sequence-level classes are 2,776 modified/nonstandard pairs,
3,075 receptor-covalent pairs, 335 known-disulfide pairs, 86
cyclic/crosslinked pairs, 51 multiple-Cys-unknown pairs, and 304
chemistry-insufficient pairs. All 2,070 receptor-covalent structure-evidence
instances have explicit connection and/or covalent-distance evidence, so they
are outside the current non-covalent retrieval task rather than ordinary
samples recoverable by sequence-only conformer generation. Adding fully
specified disulfide and cyclic generators would raise theoretical coverage
only to 15,963/22,170 (72.0027%); 355/22,170 pairs (1.6013%) remain blocked by
insufficient chemistry metadata or unresolved multiple-Cys state.

The separately authorized explicit plan is now frozen as
`phase3-v2-bounded-full-heavy-plan-v1` with classification
`EXPLICIT_BOUNDED_PLAN_PASS`. It selects eligible pairs by SHA256 of the
schema namespace, split, and interface-pair ID, with interface-pair ID as the
tie breaker. The descriptor has file SHA256
`1894F635E352D127AC79DF226E4F50A7451B8E47C43D6388239A23752721957D`
and canonical SHA256
`2F8FF55185DE5E87861687CA564EC4851E186C16C4C8158B9C1168D8E32D8DE0`.
It binds 4,096 train pairs / 1,748 sequences and 512 valid pairs / 337
sequences. Ordered pair SHA256 values are
`397FE1822FF3C1D5CD6CAAE812AB6B32ADE81FE82B65E6DE54954A340044CE26`
train and
`4ADADC28DC9ECD75B6DD9BC67BBEF7856434EB1358041A3C71A01E9CFACF81C8`
valid. Train/valid and train/safe373 pair, sequence, and relation overlaps are
all zero. Valid/safe373 overlap is explicitly retained and reported as 92
query pairs, 150 peptide sequences, and 150 biological relations.

The descriptor requires a future bounded cache for 2,085 unique sequences and
20,850 conformers, but records generation/cache status as `NOT_BUILT`.
No cache manifest or final adaptation manifest exists, and the runner rejects
descriptor-only training before constructing an optimizer. Future training
requires separately validated plan descriptor, materialized cache manifest,
and final adaptation manifest binding the plan/cache SHA values, Phase-2
checkpoint SHA, and unchanged freeze contract.

Evidence:

- `E:\pep\phase3\runs\drugclip\v2_full_split_full_heavy_eligibility_audit_v1\sequence_eligibility_registry.jsonl`
- `E:\pep\phase3\runs\drugclip\v2_full_split_full_heavy_eligibility_audit_v1\full_split_eligibility_report.json`
- `E:\pep\phase3\runs\drugclip\v2_full_split_full_heavy_eligibility_audit_v1\summary.md`
- `E:\pep\phase3\runs\drugclip\v2_bounded_full_heavy_explicit_plan_v1\bounded_plan_descriptor.json`
- `E:\pep\phase3\runs\drugclip\v2_bounded_full_heavy_explicit_plan_v1\plan_distribution_audit.json`
- `E:\pep\phase3\runs\drugclip\v2_bounded_full_heavy_explicit_plan_v1\plan_validation_report.json`

## Single Next Action

Review the frozen explicit descriptor and distribution audit before separately
authorizing generation of its dedicated 2,085-sequence bounded train/valid
cache. Do not reuse safe265/safe373 evaluation caches, change selected IDs,
silently replace failed samples, fabricate special chemistry, create a final
adaptation manifest before a cache exists, start training, or publish a data
release without separate authorization.

## Workspace Safety

- Current branch: `codex-phase3-v1-full-mmseqs`.
- The committed A-E implementation baseline ends at
  `7de86c0a3e2b3f52a026f31fb70c0aa8d61de79f`.
- The Session Bridge is maintained as a separate F-group commit on top of
  that implementation baseline. The bounded-acceptance state is maintained as
  a separate follow-up Session Bridge commit on top of the F-group baseline.
- The selected-model release descriptor, validator, and focused test are
  committed as `5c34cdbbb Add Phase 3 v1 model release contract`. This
  state-bridge update is maintained as a separate commit on top of that
  implementation boundary.
- The working tree contains extensive pre-existing modified, deleted, and
  untracked files. Phase-2 queue work, historical diagnostic sources,
  generated artifacts, and other dirty content remain outside the selected-
  model release scope.
- The bounded acceptance temporary worktree and its junction were removed by
  ordinary `git worktree remove`; `--force` was not used. The three formal run
  directories were verified after junction and worktree removal.
- The formal five-epoch training and both best-checkpoint evaluation outputs
  are retained under `E:\pep\phase3\runs\drugclip`. Their temporary detached
  worktree and output-root junction were removed after the successful state
  handoff; launcher logs were copied into the corresponding formal output
  directories before removal.
- The epoch-0 recovery training and both evaluation outputs are also retained
  under the same formal run root. Their launcher logs were copied into the
  corresponding outputs before the temporary worktree and junction were
  removed.
- The fixed-512 input-domain ablation output is retained at
  `E:\pep\phase3\runs\drugclip\v3_input_domain_ablation_fixed512_v1`. It is
  ignored result evidence and must not be staged with the evaluator sources or
  Session Bridge.
- The formal safe373 frozen-model output and launcher logs are retained at
  `E:\pep\phase3\runs\drugclip\v3_safe373_full_atom_retrieval_v1`. Its clean
  detached worktree and two exact dependency junctions were removed only after
  full result validation.
- Do not clean, revert, stage, commit, or infer file ownership from Git diff
  alone. Generated `runs/` artifacts are ignored and require direct evidence
  inspection.

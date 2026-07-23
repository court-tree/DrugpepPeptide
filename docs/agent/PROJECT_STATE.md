# PepCLIP Current Project State

Last verified: 2026-07-23

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

Formal v3 fine-tuning, epoch-4 evaluation, the one authorized epoch-0
model-selection recovery, the selected-model release contract, and the
fixed-512 input-domain ablation are complete. Epoch 0 remains the released
balanced fixed-contract checkpoint, but the ablation shows that the current
backbone-only random peptide input removes the full-heavy/side-chain signal
that drives most of the observed 3D retrieval difference. The feasibility
audit confirms that the current model can encode standard full-heavy atoms and
that local RDKit can construct compatible topology, but chemistry metadata and
robust ten-conformer generation remain unproved. Full-heavy-random retrieval
is therefore not authorized or technically ready.

## Single Next Action

Have the review session decide whether to authorize a bounded RDKit prototype
for explicitly ordinary linear, unmodified standard peptides, with
deterministic retry and strict chemistry/geometry rejection. Do not implement
the prototype or run fixed-512 all-heavy-random retrieval without that
separate authorization, and do not continue Phase-3 v1 training.

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
- Do not clean, revert, stage, commit, or infer file ownership from Git diff
  alone. Generated `runs/` artifacts are ignored and require direct evidence
  inspection.

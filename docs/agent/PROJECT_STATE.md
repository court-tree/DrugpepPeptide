# PepCLIP Current Project State

Last verified: 2026-07-22

## Current Phase

Phase-1 teacher data is frozen, Phase-2 model development is frozen, and the
active path is `phase3.drugclip`. No training or diagnostic process is
currently running. The v3 data contract, release validation, real-model smoke,
and bounded runner/evaluator code gate are complete. The runner now supports a
strict successful-optimizer-step cap, step checkpoints, and mid-epoch resume;
both retrieval evaluators accept an explicit checkpoint and model label. The
committed code passes from a clean detached worktree. No formal v3 Pilot has
been started, and full-data or multi-epoch training remains paused pending
review of a separately launched bounded step-32 Pilot.

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

This proves that the committed A-E code can run independently from a clean
worktree with the required external checkpoint supplied. It does not prove
that a formal v3 GPU Pilot, full retrieval, or ten-conformer evaluation has
been executed.

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
- These checks satisfy the v3 data/model prerequisites for a bounded Pilot.
  The separately committed and tested bounded runner enforces the 32-step
  contract, but no real v3 Pilot is authorized or complete.
- New work must explicitly select `--data-version v3` and provide the v3
  `--dataset-root`. Checkpoint resume must preserve the manifest/data contract.

Primary evidence:

- `E:\pep\phase3\runs\drugclip\random_conformer_v3\DATA_MANIFEST.json`
- `E:\pep\phase3\runs\drugclip\random_conformer_v3\VALIDATION_REPORT.md`
- `E:\pep\phase3\runs\drugclip\random_conformer_v3_smoke_20260721.json`

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

The bounded code contract is implemented and unit-tested, but no formal v3
step-32 Pilot exists yet. The remaining uncertainty is operational: a real
bounded run must confirm `step_032.pt` and both explicit-checkpoint evaluators
under the fixed 512-query, ten-conformer, full-candidate rules without
expanding into longer training.

## Single Next Action

Only after explicit user authorization, a separately claimed session may
launch one fresh formal v3 bounded Pilot with `--max_steps 32`, verify the
resulting `step_032.pt`, and run both evaluators with `--checkpoint` and
`--model-label`. Do not expand into full-data, multi-epoch, lower-learning-rate,
or Phase-2 retraining work before that bounded result is reviewed.

## Workspace Safety

- Current branch: `codex-phase3-v1-full-mmseqs`.
- The committed A-E implementation baseline ends at
  `7de86c0a3e2b3f52a026f31fb70c0aa8d61de79f`.
- The Session Bridge is maintained as a separate F-group commit on top of
  that implementation baseline.
- The working tree contains extensive pre-existing modified, deleted, and
  untracked files. The A-E paths above are tracked by the current HEAD; the
  Session Bridge closeout is the only F-group scope, while Phase-2 queue work,
  historical diagnostic sources, generated artifacts, and other dirty content
  remain outside A-E.
- Do not clean, revert, stage, commit, or infer file ownership from Git diff
  alone. Generated `runs/` artifacts are ignored and require direct evidence
  inspection.

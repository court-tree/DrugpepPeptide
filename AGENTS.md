# AGENTS.md

Repository: `E:\pep`

This file exists to keep future Codex/agent sessions aligned. If the
conversation context is unclear, read this file first, then read
`docs/agent/PROJECT_STATE.md`, `docs/agent/DECISIONS.md`, and
`docs/agent/ACTIVE_WORK.md` before consulting the historical
`PROJECT_CONTEXT.md` log or task-specific files.

## Project Identity

- This repository is the PepCLIP workspace.
- The work is split into:
  - `phase1/`: dataset construction pipeline.
  - `phase2/pepclip/`: PepCLIP model training, evaluation, embedding export,
    and retrieval code.
  - `phase3/drugclip/`: active Phase-3 v3 data contract, bounded training,
    checkpoint recovery, and retrieval evaluation code.
  - `project/`: earlier or auxiliary pipeline scripts, reports, PPT assets,
    visualizations, and teacher-facing artifacts.
  - `models/`: local model weights, including local HuggingFace/ESM assets.
  - `phase1/runs/` and `phase2/runs/`: generated experiment outputs.
- Treat `phase1/runs/full_run_v9_resume` as the frozen Phase-1 teacher-data
  baseline unless the user explicitly changes it. Older `full_run_v4` through
  `full_run_v8` directories are history or controls, not the current default.

## Session Synchronization

Before investigating, editing, or running an experiment, every session must:

1. Read `docs/agent/PROJECT_STATE.md`.
2. Read the relevant entries in `docs/agent/DECISIONS.md`.
3. Read `docs/agent/ACTIVE_WORK.md` and check for overlapping file or run
   ownership.
4. Run `git status --short` and inspect recent commits relevant to the task.
5. Confirm whether another session has already proved, rejected, or claimed
   the work. Do not infer current state from chat history alone.

Before modifying files or starting a run, add a row to
`docs/agent/ACTIVE_WORK.md`. Use an independent Git branch/worktree for
parallel code changes whenever possible. Sessions sharing this dirty working
tree must use disjoint file and output scopes.

At the end of every verified unit of work:

1. Re-check the conclusion against the actual diff and generated evidence.
2. Replace stale content in `docs/agent/PROJECT_STATE.md` with the new current
   state; do not append an unbounded diary.
3. Append only durable decisions, rejected causes, and do-not-repeat results to
   `docs/agent/DECISIONS.md`.
4. Remove the completed claim from `docs/agent/ACTIVE_WORK.md`.
5. Run `python scripts/check_agent_sync.py`.

The repository state files are the cross-session handoff mechanism. Codex
Memories and conversation summaries are auxiliary context only.

When sources conflict, use this priority:

1. Current code, data, checkpoints, hashes, and reproducible run evidence.
2. `docs/agent/PROJECT_STATE.md`.
3. `docs/agent/DECISIONS.md`.
4. Phase-specific current READMEs and reports.
5. `PROJECT_CONTEXT.md` historical override log.
6. Git history, memories, and chat history.

## Operating Rules

- Do not revert or delete unrelated local changes. This workspace often has
  active generated artifacts and uncommitted work.
- Prefer small, scoped edits that match the existing scripts and README style.
- Use UTF-8 for any new markdown files. Some older Chinese markdown may render
  as mojibake; do not infer project facts from garbled text without checking
  other sources.
- Large files and generated artifacts live in the repo. Avoid recursive scans
  of `.venv`, `model_weights.zip`, and large run directories unless the task
  requires them.
- Prefer `Get-ChildItem`, `Get-Content`, and targeted path reads on Windows
  PowerShell if `rg` is unavailable or blocked.
- Use `apply_patch` for manual source or markdown edits.
- Do not run long full-data pipelines casually. Start with smoke or pilot runs
  unless the user explicitly asks for a full run.
- Internet may be restricted. Prefer local model directories, especially under
  `E:\pep\models`, over downloading from HuggingFace at runtime.

## Common Commands

PowerShell from repo root:

```powershell
python -m py_compile phase1\step1_structure_qc.py
python -m py_compile phase2\pepclip\train_1d.py
python -m unittest discover -s tests
```

WSL Phase-1 runs:

```bash
cd /mnt/e/pep/phase1
bash run_phase1_wsl.sh
bash run_full_v2_wsl.sh
```

Phase-2 1D training entry point:

```powershell
python -m phase2.pepclip.train_1d `
  --train_track_a E:\pep\phase1\runs\full_run_v9_resume\step8_stage2_lmdb\track_a_main_train.lmdb `
  --valid_track_a E:\pep\phase1\runs\full_run_v9_resume\step8_stage2_lmdb\track_a_monitor.lmdb `
  --data_format lmdb `
  --output_dir E:\pep\phase2\runs\<run_name>
```

Phase-2 3D training entry point:

```powershell
python -m phase2.pepclip.train_3d `
  --train_track_b E:\pep\phase1\runs\full_run_v9_resume\step8_stage2_lmdb\track_b_main_train.lmdb `
  --valid_track_b E:\pep\phase1\runs\full_run_v9_resume\step8_stage2_lmdb\track_b_monitor.lmdb `
  --data_format lmdb `
  --output_dir E:\pep\phase2\runs\<run_name>
```

## Git And Workspace State

- The repo may already be dirty. Check `git status --short` before editing.
- The Phase-3 foundation, v3 release tooling, learned concat dependency,
  runtime contract, and bounded training/evaluator groups are committed through
  `7de86c0a3`. Consult `docs/agent/PROJECT_STATE.md` for the exact commit chain
  and clean-worktree verification evidence.
- Known unrelated local changes still include Phase-2 queue/training edits,
  historical Phase-3 diagnostic sources, generated/untracked PPT and report
  artifacts under `project/`, and other pre-existing dirty content.
- Do not stage, commit, or push unless the user explicitly asks.

## Where To Look First

- Current cross-session snapshot: `docs/agent/PROJECT_STATE.md`
- Durable decisions and rejected paths: `docs/agent/DECISIONS.md`
- Parallel-session ownership: `docs/agent/ACTIVE_WORK.md`
- Historical context and overrides: `PROJECT_CONTEXT.md`
- Phase-1 defaults and step behavior: `phase1/README.md`
- Phase-1 tuning history: `phase1/tuning_history.md`
- Phase-1 comparison notes: `phase1/v4_v5_v6_summary.md`
- Phase-2 commands and model contracts: `phase2/pepclip/README.md`
- Phase-2 result summary: `phase2/runs/summary.md`
- Next Phase-1 experiment ideas: `project/next_experiments_after_stable_run.md`

# PepCLIP Session Bridge

This directory is the repository-owned synchronization layer for Codex
sessions. It exists because open sessions do not share uncommitted reasoning in
real time, and automatic memories are neither immediate nor authoritative.

## File Roles

- `PROJECT_STATE.md`: replace-in-place snapshot of what is true now and the
  single next action.
- `DECISIONS.md`: append-only durable conclusions, rejected causes, and
  do-not-repeat constraints.
- `ACTIVE_WORK.md`: short-lived ownership claims for parallel sessions.
- `HANDOFF_TEMPLATE.md`: checklist used to finish a verified work unit.

Do not turn `PROJECT_STATE.md` into a chronological log. Do not place tentative
hypotheses in `DECISIONS.md`. Do not leave completed claims in
`ACTIVE_WORK.md`.

## Start Of A Session

1. Read the root `AGENTS.md`.
2. Read `PROJECT_STATE.md` and the relevant sections of `DECISIONS.md`.
3. Read `ACTIVE_WORK.md` and inspect `git status --short`.
4. Summarize confirmed facts, excluded directions, and the single next action.
5. Add an ownership row before editing or launching a run.

An ownership row must name a bounded scope. Examples are
`phase3/drugclip/audit_first_optimizer_step.py` or
`phase3/runs/drugclip/pilot_step32_v1/`; `phase3/` is too broad.

## End Of A Verified Work Unit

Use `HANDOFF_TEMPLATE.md`, then:

1. Verify claims against the diff, logs, metrics, hashes, or tests.
2. Update the current snapshot.
3. Append a decision only when it should constrain future sessions.
4. Release the ownership claim.
5. Run:

```powershell
python scripts\check_agent_sync.py
```

When a result is claimed to work independently of the dirty main workspace,
verify the committed revision in a clean detached worktree. Record the exact
HEAD, tests, required external artifact provisioning, and whether the temporary
worktree was safely removed. Passing a clean-worktree code gate does not imply
that a real training or evaluation run occurred.

## Parallel Work

Independent worktrees are the safe default for overlapping code changes. If
sessions must share this working tree, their claimed source files and output
directories must be disjoint. A session must not edit a claimed path or reuse
another session's run directory.

Writing a result here is the synchronization event. Another already-open
session learns it only after re-reading these files or the corresponding Git
commit.

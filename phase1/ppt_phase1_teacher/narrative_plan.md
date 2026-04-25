# Phase-1 teacher-report deck plan

Audience: thesis advisor / lab meeting teacher.
Objective: explain the new simplified Phase-1 pipeline, why it is trustworthy, what tuning evidence led to current defaults, and what remaining limitation is known.
Narrative arc: from real PPI structures -> candidate peptide windows -> physical filtering -> representative sampling -> cautious joint homology dedup -> final 1D/3D dataset; then show tuning evidence and recommended defaults.
Slide list:
1. Title and one-sentence thesis.
2. End-to-end 7-step pipeline map.
3. Candidate generation and filtering logic: why local average contact matters.
4. Tuning history: what failed and what changed.
5. Current defaults and why they are defensible.
6. Length bias diagnosis and selected fix: per-task 8-mer cap.
7. Final teacher-facing conclusion and next checkpoint.
Visual system: clean scientific report, light background, deep navy text, teal/orange accents, card-based diagrams, simple native charts/tables.
Source plan: phase1 README, tuning_history.md, length_balance_experiment.md, len8cap_experiment.md, user-provided phase1 description.
Editability plan: all text, flow labels, tables, and charts authored as editable PowerPoint objects.

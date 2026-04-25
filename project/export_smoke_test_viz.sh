#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/e/pep/project"
RUN_NAME="smoke_test"
PYTHON_BIN="python"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) PROJECT_ROOT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash export_smoke_test_viz.sh [options]

Options:
  --project-root PATH
  --run-name NAME
  --python BIN
EOF
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

cd "${PROJECT_ROOT}"

RUN_DIR="${PROJECT_ROOT}/runs/${RUN_NAME}"
OUT_DIR="${RUN_DIR}/viz_review"
mkdir -p "${OUT_DIR}"

STEP5_JSONL="${RUN_DIR}/step5/step5_final.jsonl"
STEP6C_JSONL="${RUN_DIR}/step6c_align/step6c_main_survived.jsonl"
PDB_DIR="/mnt/e/pep/download"

echo "================================================================================"
echo "[START] Export smoke-test visualization PDBs"
echo "[INPUT] step5  = ${STEP5_JSONL}"
echo "[INPUT] step6c = ${STEP6C_JSONL}"
echo "[OUTPUT] dir   = ${OUT_DIR}"
echo "================================================================================"

# ------------------------------------------------------------------
# Step 5 task-level review: check whether windows stay on one interface
# ------------------------------------------------------------------
"${PYTHON_BIN}" visualize_export_pdb.py \
  --input_jsonl "${STEP5_JSONL}" \
  --pdb_dir "${PDB_DIR}" \
  --parent_task_id "fc2ab51f-be33-53c9-b37b-3b99a58d6110" \
  --receptor_scope full \
  --no_source_full \
  --output_pdb "${OUT_DIR}/step5_task_11ba_context.pdb"

"${PYTHON_BIN}" visualize_export_pdb.py \
  --input_jsonl "${STEP5_JSONL}" \
  --pdb_dir "${PDB_DIR}" \
  --parent_task_id "fc2ab51f-be33-53c9-b37b-3b99a58d6110" \
  --receptor_scope patch \
  --no_source_full \
  --output_pdb "${OUT_DIR}/step5_task_11ba_closeup.pdb"

"${PYTHON_BIN}" visualize_export_pdb.py \
  --input_jsonl "${STEP5_JSONL}" \
  --pdb_dir "${PDB_DIR}" \
  --parent_task_id "cdb0e6bf-8690-5261-ad3b-d0b9a5fe4245" \
  --receptor_scope full \
  --no_source_full \
  --output_pdb "${OUT_DIR}/step5_task_1a3x_context.pdb"

"${PYTHON_BIN}" visualize_export_pdb.py \
  --input_jsonl "${STEP5_JSONL}" \
  --pdb_dir "${PDB_DIR}" \
  --parent_task_id "cdb0e6bf-8690-5261-ad3b-d0b9a5fe4245" \
  --receptor_scope patch \
  --no_source_full \
  --output_pdb "${OUT_DIR}/step5_task_1a3x_closeup.pdb"

"${PYTHON_BIN}" visualize_export_pdb.py \
  --input_jsonl "${STEP5_JSONL}" \
  --pdb_dir "${PDB_DIR}" \
  --parent_task_id "a8329f72-cef5-50ba-93f3-cd47f2b59add" \
  --receptor_scope full \
  --no_source_full \
  --output_pdb "${OUT_DIR}/step5_task_1adb_context.pdb"

"${PYTHON_BIN}" visualize_export_pdb.py \
  --input_jsonl "${STEP5_JSONL}" \
  --pdb_dir "${PDB_DIR}" \
  --parent_task_id "a8329f72-cef5-50ba-93f3-cd47f2b59add" \
  --receptor_scope patch \
  --no_source_full \
  --output_pdb "${OUT_DIR}/step5_task_1adb_closeup.pdb"

# ------------------------------------------------------------------
# Step 6C final sample review: check whether survivors look train-worthy
# ------------------------------------------------------------------
"${PYTHON_BIN}" visualize_export_pdb.py \
  --input_jsonl "${STEP6C_JSONL}" \
  --pdb_dir "${PDB_DIR}" \
  --candidate_id "23a94d74-f075-4850-a206-61e45bb5989f" \
  --receptor_scope patch \
  --output_pdb "${OUT_DIR}/step6c_candidate_10lg_high_rbsa.pdb"

"${PYTHON_BIN}" visualize_export_pdb.py \
  --input_jsonl "${STEP6C_JSONL}" \
  --pdb_dir "${PDB_DIR}" \
  --candidate_id "50a18ebf-f013-4615-a61d-f9a7920472a9" \
  --receptor_scope patch \
  --output_pdb "${OUT_DIR}/step6c_candidate_1a9m_high_rbsa.pdb"

"${PYTHON_BIN}" visualize_export_pdb.py \
  --input_jsonl "${STEP6C_JSONL}" \
  --pdb_dir "${PDB_DIR}" \
  --candidate_id "57644247-1927-4dcb-ab8c-7b2570f8ac29" \
  --receptor_scope patch \
  --output_pdb "${OUT_DIR}/step6c_candidate_1a8g_high_rbsa.pdb"

"${PYTHON_BIN}" visualize_export_pdb.py \
  --input_jsonl "${STEP6C_JSONL}" \
  --pdb_dir "${PDB_DIR}" \
  --candidate_id "a51f9950-862b-4234-b07a-928fdcdb5b10" \
  --receptor_scope patch \
  --output_pdb "${OUT_DIR}/step6c_candidate_1a7a_high_rbsa.pdb"

cat > "${OUT_DIR}/README.txt" <<'EOF'
Suggested review order:

1. step5_task_11ba_context.pdb
   Use this first. Check whether all candidate windows fall on the same receptor-side interface region.

2. step5_task_11ba_closeup.pdb
   Then zoom in. Check whether 8/9/16/17/20 aa windows all stay near the same local patch.

3. step5_task_1a3x_context.pdb
4. step5_task_1a3x_closeup.pdb
   This task looked odd in the earlier export; compare context vs closeup before judging.

5. step5_task_1adb_context.pdb
6. step5_task_1adb_closeup.pdb
   Includes a weaker short-window case; useful for spotting whether low-rBSA short cuts still hug the interface.

7. step6c_candidate_10lg_high_rbsa.pdb
8. step6c_candidate_1a9m_high_rbsa.pdb
9. step6c_candidate_1a8g_high_rbsa.pdb
10. step6c_candidate_1a7a_high_rbsa.pdb
   These are strong final survivors from Step 6C. Check peptide-patch compactness and whether they look train-worthy.

Chain semantics inside exported PDBs:
  R = receptor or receptor patch union
  S = peptide source full chain
  P = selected candidate peptide window
  X = candidate-local receptor patch
  A/B/C/... = multiple candidate windows in task-level exports
EOF

cat > "${OUT_DIR}/task_review_style.pml" <<'EOF'
# Basic styling for Step 5 task-level review files
# Usage in PyMOL:
#   load step5_task_1a3x_context.pdb
#   @task_review_style.pml

hide everything, all
bg_color black
set ray_opaque_background, off

show cartoon, chain R
color gray70, chain R
set cartoon_transparency, 0.35, chain R

show sticks, not chain R
util.cnc not chain R

color tv_red, chain A
color tv_orange, chain B
color yellow, chain C
color green, chain D
color cyan, chain E
color marine, chain F
color magenta, chain G
color salmon, chain H
color violet, chain I
color limon, chain J

orient
zoom visible, 6
EOF

cat > "${OUT_DIR}/single_review_style.pml" <<'EOF'
# Basic styling for Step 6C single-candidate review files
# Usage in PyMOL:
#   load step6c_candidate_10lg_high_rbsa.pdb
#   @single_review_style.pml

hide everything, all
bg_color black
set ray_opaque_background, off

show cartoon, chain R
color gray70, chain R
set cartoon_transparency, 0.40, chain R

show sticks, chain P
color tv_orange, chain P

show sticks, chain X
color cyan, chain X

show line, chain S
color green, chain S

orient
zoom visible, 6
EOF

echo "================================================================================"
echo "[DONE] Exported visualization PDBs into: ${OUT_DIR}"
echo "[DONE] Open the files in PyMOL/Chimera and start with README.txt"
echo "[DONE] Optional PyMOL styles: task_review_style.pml / single_review_style.pml"
echo "================================================================================"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/e/pep/project}"
PHASE1_ROOT="${PHASE1_ROOT:-/mnt/e/pep/phase1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-${PHASE1_ROOT}/runs/dev}"
PDB_DIR="${PDB_DIR:-/mnt/e/pep/download}"

STEP1_DIR="${RUN_ROOT}/step1"
STEP2_DIR="${RUN_ROOT}/step2"
TASKS_JSONL="${TASKS_JSONL:-${STEP2_DIR}/step2_tasks.jsonl}"
STEP3_OUT="${RUN_ROOT}/step3/step3_candidates.jsonl"
STEP4_OUT="${RUN_ROOT}/step4/step4_features.jsonl"
STEP5_OUT="${RUN_ROOT}/step5/step5_final.jsonl"
STEP6_MAIN="${RUN_ROOT}/step6/step6_main.jsonl"
STEP6_MONITOR="${RUN_ROOT}/step6/step6_monitor.jsonl"
STEP6_DROPPED="${RUN_ROOT}/step6/step6_dropped.jsonl"
STEP7_OUT="${RUN_ROOT}/step7/final_metadata.jsonl"

mkdir -p "${RUN_ROOT}/step1" "${RUN_ROOT}/step2" "${RUN_ROOT}/step3" "${RUN_ROOT}/step4" "${RUN_ROOT}/step5" "${RUN_ROOT}/step6" "${RUN_ROOT}/step7"

echo "[PHASE1] Step1 structure QC"
"${PYTHON_BIN}" "${PHASE1_ROOT}/step1_structure_qc.py" \
  --pdb_dir "${PDB_DIR}" \
  --output_dir "${STEP1_DIR}"

echo "[PHASE1] Step2 bidirectional task generation"
"${PYTHON_BIN}" "${PHASE1_ROOT}/step2_generate_tasks.py" \
  --step1_dir "${STEP1_DIR}" \
  --pdb_dir "${PDB_DIR}" \
  --output_jsonl "${TASKS_JSONL}" \
  --error_jsonl "${STEP2_DIR}/step2_errors.jsonl" \
  --workers 4

echo "[PHASE1] Step3 local-window candidate generation"
"${PYTHON_BIN}" "${PHASE1_ROOT}/step3_window_candidates.py" \
  --tasks_jsonl "${TASKS_JSONL}" \
  --pdb_dir "${PDB_DIR}" \
  --output_jsonl "${STEP3_OUT}" \
  --error_jsonl "${RUN_ROOT}/step3/step3_errors.jsonl" \
  --workers 4

echo "[PHASE1] Step4 physical sanity + dedup"
"${PYTHON_BIN}" "${PHASE1_ROOT}/step4_filter_dedup.py" \
  --candidate_jsonl "${STEP3_OUT}" \
  --pdb_dir "${PDB_DIR}" \
  --output_jsonl "${STEP4_OUT}" \
  --error_jsonl "${RUN_ROOT}/step4/step4_errors.jsonl" \
  --workers 4

echo "[PHASE1] Step5 sampling by average contact count"
"${PYTHON_BIN}" "${PHASE1_ROOT}/step5_sample_by_avg_contacts.py" \
  --input_jsonl "${STEP4_OUT}" \
  --output_jsonl "${STEP5_OUT}" \
  --max_keep_per_task 4

echo "[PHASE1] Step6 joint receptor+peptide homology dedup"
"${PYTHON_BIN}" "${PHASE1_ROOT}/step6_joint_homology_dedup.py" \
  --input_jsonl "${STEP5_OUT}" \
  --pdb_dir "${PDB_DIR}" \
  --main_output_jsonl "${STEP6_MAIN}" \
  --monitor_output_jsonl "${STEP6_MONITOR}" \
  --dropped_output_jsonl "${STEP6_DROPPED}"

echo "[PHASE1] Step7 finalize dataset metadata"
"${PYTHON_BIN}" "${PHASE1_ROOT}/step7_finalize_dataset.py" \
  --main_jsonl "${STEP6_MAIN}" \
  --monitor_jsonl "${STEP6_MONITOR}" \
  --pdb_dir "${PDB_DIR}" \
  --output_jsonl "${STEP7_OUT}"

echo "[PHASE1] done"

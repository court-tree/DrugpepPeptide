#!/usr/bin/env bash
set -euo pipefail

PHASE1_ROOT="${PHASE1_ROOT:-/mnt/e/pep/phase1}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/e/pep/.venv/bin/python}"
PDB_DIR="${PDB_DIR:-/mnt/e/pep/phase1/test_input_1000}"
TASKS_JSONL="${TASKS_JSONL:-${PHASE1_ROOT}/runs/pilot_1000/step2/step2_tasks.jsonl}"
RUN_ROOT="${RUN_ROOT:-${PHASE1_ROOT}/runs/pilot_1000_logic_v2}"

mkdir -p "${RUN_ROOT}/step3" "${RUN_ROOT}/step4" "${RUN_ROOT}/step5" "${RUN_ROOT}/step6" "${RUN_ROOT}/step7"

echo "[TEST] Step3"
"${PYTHON_BIN}" "${PHASE1_ROOT}/step3_window_candidates.py" \
  --tasks_jsonl "${TASKS_JSONL}" \
  --pdb_dir "${PDB_DIR}" \
  --output_jsonl "${RUN_ROOT}/step3/step3_candidates.jsonl" \
  --error_jsonl "${RUN_ROOT}/step3/step3_errors.jsonl" \
  --workers 4

echo "[TEST] Step4"
"${PYTHON_BIN}" "${PHASE1_ROOT}/step4_filter_dedup.py" \
  --candidate_jsonl "${RUN_ROOT}/step3/step3_candidates.jsonl" \
  --pdb_dir "${PDB_DIR}" \
  --output_jsonl "${RUN_ROOT}/step4/step4_features.jsonl" \
  --error_jsonl "${RUN_ROOT}/step4/step4_errors.jsonl" \
  --workers 4

echo "[TEST] Step5"
"${PYTHON_BIN}" "${PHASE1_ROOT}/step5_sample_by_avg_contacts.py" \
  --input_jsonl "${RUN_ROOT}/step4/step4_features.jsonl" \
  --output_jsonl "${RUN_ROOT}/step5/step5_final.jsonl" \
  --max_keep_per_task 4 \
  --max_len8_per_task 2

echo "[TEST] Step6"
"${PYTHON_BIN}" "${PHASE1_ROOT}/step6_joint_homology_dedup.py" \
  --input_jsonl "${RUN_ROOT}/step5/step5_final.jsonl" \
  --pdb_dir "${PDB_DIR}" \
  --main_output_jsonl "${RUN_ROOT}/step6/step6_main.jsonl" \
  --monitor_output_jsonl "${RUN_ROOT}/step6/step6_monitor.jsonl" \
  --dropped_output_jsonl "${RUN_ROOT}/step6/step6_dropped.jsonl" \
  --progress_every 10000

echo "[TEST] Step7"
"${PYTHON_BIN}" "${PHASE1_ROOT}/step7_finalize_dataset.py" \
  --main_jsonl "${RUN_ROOT}/step6/step6_main.jsonl" \
  --monitor_jsonl "${RUN_ROOT}/step6/step6_monitor.jsonl" \
  --pdb_dir "${PDB_DIR}" \
  --output_jsonl "${RUN_ROOT}/step7/final_metadata.jsonl"

echo "[TEST] Summary"
"${PYTHON_BIN}" "${PHASE1_ROOT}/summarize_final_metadata.py" \
  --input_jsonl "${RUN_ROOT}/step7/final_metadata.jsonl" \
  --output_json "${RUN_ROOT}/step7/final_metadata_summary.json" \
  > "${RUN_ROOT}/step7/final_metadata_summary.stdout.json"

echo "[TEST] done: ${RUN_ROOT}"

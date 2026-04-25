#!/usr/bin/env bash
set -euo pipefail

PHASE1_ROOT="${PHASE1_ROOT:-/mnt/e/pep/phase1}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/e/pep/.venv/bin/python}"
PDB_DIR="${PDB_DIR:-/mnt/e/pep/phase1/test_input_1000}"
SOURCE_RUN="${SOURCE_RUN:-${PHASE1_ROOT}/runs/pilot_1000}"
OUT_ROOT="${OUT_ROOT:-${PHASE1_ROOT}/runs}"

if [ "$#" -gt 0 ]; then
  ALPHAS="$*"
else
  ALPHAS="${ALPHAS:-0 0.25 0.5 0.75}"
fi

for alpha in ${ALPHAS}; do
  tag="${alpha/./p}"
  run="${OUT_ROOT}/pilot_1000_alpha_${tag}"
  mkdir -p "${run}/step5" "${run}/step6" "${run}/step7"

  echo "== alpha=${alpha} step5 =="
  "${PYTHON_BIN}" "${PHASE1_ROOT}/step5_sample_by_avg_contacts.py" \
    --input_jsonl "${SOURCE_RUN}/step4/step4_features.jsonl" \
    --output_jsonl "${run}/step5/step5_final.jsonl" \
    --max_keep_per_task 4 \
    --length_balance_alpha "${alpha}"

  echo "== alpha=${alpha} step6 =="
  "${PYTHON_BIN}" "${PHASE1_ROOT}/step6_joint_homology_dedup.py" \
    --input_jsonl "${run}/step5/step5_final.jsonl" \
    --pdb_dir "${PDB_DIR}" \
    --main_output_jsonl "${run}/step6/step6_main.jsonl" \
    --monitor_output_jsonl "${run}/step6/step6_monitor.jsonl" \
    --dropped_output_jsonl "${run}/step6/step6_dropped.jsonl" \
    --progress_every 10000

  echo "== alpha=${alpha} step7 =="
  "${PYTHON_BIN}" "${PHASE1_ROOT}/step7_finalize_dataset.py" \
    --main_jsonl "${run}/step6/step6_main.jsonl" \
    --monitor_jsonl "${run}/step6/step6_monitor.jsonl" \
    --pdb_dir "${PDB_DIR}" \
    --output_jsonl "${run}/step7/final_metadata.jsonl"

  echo "== alpha=${alpha} summarize =="
  "${PYTHON_BIN}" "${PHASE1_ROOT}/summarize_final_metadata.py" \
    --input_jsonl "${run}/step7/final_metadata.jsonl" \
    --output_json "${run}/step7/final_metadata_summary.json" \
    > "${run}/step7/final_metadata_summary.stdout.json"
done

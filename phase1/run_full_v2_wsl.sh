#!/usr/bin/env bash
set -euo pipefail

PHASE1_ROOT="${PHASE1_ROOT:-/mnt/e/pep/phase1}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/e/pep/.venv/bin/python}"
PDB_DIR="${PDB_DIR:-/mnt/e/pep/download}"
RUN_ROOT="${RUN_ROOT:-${PHASE1_ROOT}/runs/full_run_v2}"
WORKERS="${WORKERS:-4}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-300}"
STEP6_PROGRESS_EVERY="${STEP6_PROGRESS_EVERY:-10000}"

# Optional shortcut for full reruns. Step1/Step2 are unchanged by the latest
# algorithm edits, so this can reuse/copy their outputs from a previous full run
# while still writing everything into the new full_run_v2 directory.
REUSE_STEP12_FROM="${REUSE_STEP12_FROM:-}"
REUSE_STEP3_FROM="${REUSE_STEP3_FROM:-}"

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
RUN_LOG="${RUN_ROOT}/full_run_v2.log"

mkdir -p \
  "${RUN_ROOT}/step1" "${RUN_ROOT}/step2" "${RUN_ROOT}/step3" \
  "${RUN_ROOT}/step4" "${RUN_ROOT}/step5" "${RUN_ROOT}/step6" \
  "${RUN_ROOT}/step7"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

file_size() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    du -h "${path}" | awk '{print $1}'
  else
    echo "missing"
  fi
}

line_count() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    wc -l < "${path}"
  else
    echo "0"
  fi
}

log_msg() {
  echo "[$(timestamp)] $*" | tee -a "${RUN_LOG}"
}

show_summary_if_exists() {
  local summary_path="$1"
  if [[ -f "${summary_path}" ]]; then
    log_msg "summary: ${summary_path}"
    cat "${summary_path}" | tee -a "${RUN_LOG}"
    echo | tee -a "${RUN_LOG}" >/dev/null
  fi
}

run_step() {
  local step_name="$1"
  local output_path="$2"
  local summary_path="$3"
  local step_log="$4"
  shift 4

  mkdir -p "$(dirname "${step_log}")"
  : > "${step_log}"

  log_msg "START ${step_name}"
  log_msg "log=${step_log}"
  log_msg "output=${output_path}"

  local start_ts
  start_ts="$(date +%s)"

  (
    echo "[$(timestamp)] command: $*"
    "$@"
  ) >> "${step_log}" 2>&1 &

  local pid=$!
  log_msg "${step_name} pid=${pid}"

  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${HEARTBEAT_SEC}"
    if kill -0 "${pid}" 2>/dev/null; then
      local now elapsed
      now="$(date +%s)"
      elapsed=$((now - start_ts))
      log_msg "HEARTBEAT ${step_name} | elapsed=$((elapsed / 60)) min | output_size=$(file_size "${output_path}") | output_lines=$(line_count "${output_path}") | log_size=$(file_size "${step_log}")"
      tail -n 5 "${step_log}" | sed 's/^/[tail] /' | tee -a "${RUN_LOG}" || true
    fi
  done

  wait "${pid}"
  local exit_code=$?
  local end_ts elapsed
  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))

  if [[ "${exit_code}" -ne 0 ]]; then
    log_msg "FAILED ${step_name} | exit_code=${exit_code} | elapsed=$((elapsed / 60)) min"
    log_msg "last log lines:"
    tail -n 80 "${step_log}" | sed 's/^/[tail] /' | tee -a "${RUN_LOG}" || true
    exit "${exit_code}"
  fi

  log_msg "DONE ${step_name} | elapsed=$((elapsed / 60)) min | output_size=$(file_size "${output_path}") | output_lines=$(line_count "${output_path}")"
  show_summary_if_exists "${summary_path}"
}

copy_step_outputs() {
  local src_root="$1"
  local step_name="$2"
  local src_dir="${src_root}/${step_name}"
  local dst_dir="${RUN_ROOT}/${step_name}"

  if [[ ! -d "${src_dir}" ]]; then
    log_msg "Cannot reuse ${step_name}: missing ${src_dir}"
    exit 1
  fi

  log_msg "REUSE ${step_name}: ${src_dir} -> ${dst_dir}"
  mkdir -p "${dst_dir}"
  cp -a "${src_dir}/." "${dst_dir}/"
}

log_msg "PeptideCLIP Phase1 full_run_v2"
log_msg "PHASE1_ROOT=${PHASE1_ROOT}"
log_msg "PYTHON_BIN=${PYTHON_BIN}"
log_msg "PDB_DIR=${PDB_DIR}"
log_msg "RUN_ROOT=${RUN_ROOT}"
log_msg "WORKERS=${WORKERS}"
log_msg "HEARTBEAT_SEC=${HEARTBEAT_SEC}"
log_msg "REUSE_STEP12_FROM=${REUSE_STEP12_FROM:-none}"
log_msg "REUSE_STEP3_FROM=${REUSE_STEP3_FROM:-none}"

if [[ -n "${REUSE_STEP12_FROM}" ]]; then
  copy_step_outputs "${REUSE_STEP12_FROM}" "step1"
  copy_step_outputs "${REUSE_STEP12_FROM}" "step2"
  show_summary_if_exists "${STEP1_DIR}/step1_summary.json"
  show_summary_if_exists "${STEP2_DIR}/step2_summary.json"
else
  run_step \
    "Step1 structure QC" \
    "${STEP1_DIR}/step1_results.jsonl" \
    "${STEP1_DIR}/step1_summary.json" \
    "${STEP1_DIR}/step1.log" \
    "${PYTHON_BIN}" "${PHASE1_ROOT}/step1_structure_qc.py" \
      --pdb_dir "${PDB_DIR}" \
      --output_dir "${STEP1_DIR}"

  run_step \
    "Step2 bidirectional task generation" \
    "${TASKS_JSONL}" \
    "${STEP2_DIR}/step2_summary.json" \
    "${STEP2_DIR}/step2.log" \
    "${PYTHON_BIN}" "${PHASE1_ROOT}/step2_generate_tasks.py" \
      --step1_dir "${STEP1_DIR}" \
      --pdb_dir "${PDB_DIR}" \
      --output_jsonl "${TASKS_JSONL}" \
      --error_jsonl "${STEP2_DIR}/step2_errors.jsonl" \
      --workers "${WORKERS}"
fi

if [[ -n "${REUSE_STEP3_FROM}" ]]; then
  copy_step_outputs "${REUSE_STEP3_FROM}" "step3"
  show_summary_if_exists "${RUN_ROOT}/step3/step3_summary.json"
else
  run_step \
    "Step3 hotspot-window candidate generation" \
    "${STEP3_OUT}" \
    "${RUN_ROOT}/step3/step3_summary.json" \
    "${RUN_ROOT}/step3/step3.log" \
    "${PYTHON_BIN}" "${PHASE1_ROOT}/step3_window_candidates.py" \
      --tasks_jsonl "${TASKS_JSONL}" \
      --pdb_dir "${PDB_DIR}" \
      --output_jsonl "${STEP3_OUT}" \
      --error_jsonl "${RUN_ROOT}/step3/step3_errors.jsonl" \
      --workers "${WORKERS}"
fi

run_step \
  "Step4 single-candidate physical filter" \
  "${STEP4_OUT}" \
  "${RUN_ROOT}/step4/step4_summary.json" \
  "${RUN_ROOT}/step4/step4.log" \
  "${PYTHON_BIN}" "${PHASE1_ROOT}/step4_filter_dedup.py" \
    --candidate_jsonl "${STEP3_OUT}" \
    --pdb_dir "${PDB_DIR}" \
    --output_jsonl "${STEP4_OUT}" \
    --error_jsonl "${RUN_ROOT}/step4/step4_errors.jsonl" \
    --workers "${WORKERS}"

run_step \
  "Step5 weighted sampling by avg_contact_count" \
  "${STEP5_OUT}" \
  "${RUN_ROOT}/step5/step5_summary.json" \
  "${RUN_ROOT}/step5/step5.log" \
  "${PYTHON_BIN}" "${PHASE1_ROOT}/step5_sample_by_avg_contacts.py" \
    --input_jsonl "${STEP4_OUT}" \
    --output_jsonl "${STEP5_OUT}" \
    --max_keep_per_task 4

run_step \
  "Step6 joint receptor+peptide homology dedup" \
  "${STEP6_MAIN}" \
  "${RUN_ROOT}/step6/step6_summary.json" \
  "${RUN_ROOT}/step6/step6.log" \
  "${PYTHON_BIN}" "${PHASE1_ROOT}/step6_joint_homology_dedup.py" \
    --input_jsonl "${STEP5_OUT}" \
    --pdb_dir "${PDB_DIR}" \
    --main_output_jsonl "${STEP6_MAIN}" \
    --monitor_output_jsonl "${STEP6_MONITOR}" \
    --dropped_output_jsonl "${STEP6_DROPPED}" \
    --progress_every "${STEP6_PROGRESS_EVERY}"

run_step \
  "Step7 finalize dataset metadata" \
  "${STEP7_OUT}" \
  "${RUN_ROOT}/step7/step7_summary.json" \
  "${RUN_ROOT}/step7/step7.log" \
  "${PYTHON_BIN}" "${PHASE1_ROOT}/step7_finalize_dataset.py" \
    --main_jsonl "${STEP6_MAIN}" \
    --monitor_jsonl "${STEP6_MONITOR}" \
    --pdb_dir "${PDB_DIR}" \
    --output_jsonl "${STEP7_OUT}"

run_step \
  "Final metadata summary" \
  "${RUN_ROOT}/step7/final_metadata_summary.json" \
  "${RUN_ROOT}/step7/final_metadata_summary.json" \
  "${RUN_ROOT}/step7/final_metadata_summary.log" \
  "${PYTHON_BIN}" "${PHASE1_ROOT}/summarize_final_metadata.py" \
    --input_jsonl "${STEP7_OUT}" \
    --output_json "${RUN_ROOT}/step7/final_metadata_summary.json"

log_msg "ALL DONE full_run_v2"
log_msg "Final metadata: ${STEP7_OUT}"
log_msg "Final summary : ${RUN_ROOT}/step7/final_metadata_summary.json"

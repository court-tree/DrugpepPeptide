#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# PeptideCLIP Phase-1 Pipeline Runner
# =========================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DATA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_ROOT="${SCRIPT_DIR}"
CIF_DIR="${DEFAULT_DATA_ROOT}/download"
PEPBDB_ROOT="${DEFAULT_DATA_ROOT}/PepBDB_download/pepbdb-20200318/pepbdb"
RUN_ROOT="${PROJECT_ROOT}/runs"
RUN_NAME="smoke_test"

PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKERS="2"
CHUNKSIZE="20"
PROGRESS_EVERY="20"
MAX_TASKS="0"
MAX_CANDIDATES="20000"
MAP_SIZE_GB="4"
RESUME_FROM=""
FORCE_STEP=""

# -------------------------------
# 闂傚倸鍊烽懗鍫曞储瑜旈幃娲Ω瑜嶉弸鍫⑩偓骞垮劚閹锋垿鎳撻崹顔氬綊宕楅崗鑲╃▏濠碘槅鍋呴敃銏ゅ蓟濞戙垹唯妞ゆ牜鍋為宥咁渻閵堝棗濮冪紒顔界懇瀵鈽夊鍡樺兊濡炪倖甯掗崐鐢稿几瀹€鈧槐鎾存媴缁嬭法绉梺鍛婎殕婵炲﹪鎮伴鈧獮鍥敇閻愮數鐛┑鐘灱濞夋盯鎯夋總绋跨叀濠㈣埖鍔栭埛鎴︽煕濠靛棗顏褔浜堕弻娑㈠箻鐎靛憡鍣ч梺鎸庢磸閸ㄤ粙鐛鈧、娆撴寠婢跺﹤顥愰梻鍌欒兌缁垰螞閸愵啟澶愬箻鐠囪尙楠囨繛瀵稿Т椤戝棝鎮″☉銏＄厱閻忕偛澧介。鏌ユ煟椤撶噥娈曠紒缁樼洴瀹曢亶骞囬鍌欑礄闁诲氦顫夊ú蹇涘磿闁稁鏁囧┑鍌滎焾濡炰粙鏌涢幇鍏哥敖闁崇鍊濆?# -------------------------------
STEP1_SCRIPT="step1_xray_qc_mmcif.py"
STEP2_SCRIPT="step2_generate_tasks_improved.py"
STEP3_SCRIPT="step3_method_a_multi_anchor.py"
STEP4_SCRIPT="step4_postscore_candidates.py"
STEP5_SCRIPT="step5_method_a_probabilistic_sampling.py"
STEP6_REF_SCRIPT="build_reference_dataset_from_pepbdb.py"
STEP6_ALIGN_SCRIPT="step6_stratified_sampling.py"
STEP6X_SCRIPT="step6x_receptor_sequence_clusters.py"
STEP6C_SCRIPT="step6c_postcap_realign.py"
STEP7_SCRIPT="step7_family_capping_monitor_split_v2.py"
STEP8_SCRIPT="step8_finalize_dataset_v2.py"

# -------------------------------
# Step 闂傚倸鍊风粈渚€骞夐敓鐘冲仭闁靛鏅涚壕鍦喐閻楀牆绗掓慨?# -------------------------------
STEP1_MAX_RESOLUTION="3.5"
STEP1_MIN_INTERFACE_BSA="200.0"
STEP1_CONTACT_PREFILTER_CUTOFF="6.0"

STEP2_CHAIN_CONTACT_CUTOFF="6.0"
STEP2_DIRECTED_CONTACT_CUTOFF="6.0"
STEP2_MIN_SOURCE_CONTACT_RESIDUES="2"
STEP2_MIN_SOURCE_CHAIN_RESIDUES="8"

# Step 3: multi-anchor candidate generation
STEP3_ANCHOR_CUTOFF="5.0"
STEP3_CONTACT_CUTOFF="6.0"
STEP3_NMS_MIN_SEQ_GAP="3"
STEP3_MAX_ANCHORS_PER_TASK="5"
STEP3_MIN_LEN="8"
STEP3_MAX_LEN="20"
STEP3_MIN_CONTACT_RESIDUES_6A="4"
STEP3_MIN_CONTACT_RATIO_6A="0.40"
STEP3_MAX_WINDOWS_PER_ANCHOR="6"
STEP3_LIMIT_TASKS="200"   # smoke test default; set to 0 for a full run

STEP4_CONTACT_CUTOFF_6A="6.0"
STEP4_POCKET_CUTOFF_6A="6.0"
STEP4_MIN_CONTACT_RESIDUES_6A="2"
STEP4_MIN_RBSA_RAW="0.05"

# Step 5
STEP5_TOP_K="5"
STEP5_DENSITY_THRESHOLD="0.50"
STEP5_MIN_LEN="8"
STEP5_MAX_LEN="20"
STEP5_MIN_CONTACT_RESIDUES_6A="4"
STEP5_MIN_RBSA_RAW="0.05"
STEP5_SAMPLING_POWER="1.0"
STEP5_LENGTH_BONUS_STRENGTH="0.25"

# Step 6A / 6B
STEP6_REF_LIMIT="500"               # smoke test default; set to 0 for a full reference build
STEP6_POCKET_CUTOFF="6.0"

# Step 6B
STEP6_ALPHA="1.0"
STEP6_EPSILON_GEN="1e-5"
STEP6_EPSILON_REF="1e-6"
STEP6_RANDOM_STATE="42"
STEP6_TARGET_KEEP_RATIO="0.40"
STEP6_ENSURE_TASK_FLOOR="1"

# Step 6X: receptor sequence cluster annotation
STEP6X_CLUSTER_MODE="mmseqs"
STEP6X_MMSEQS_BIN=""
STEP6X_MIN_SEQ_ID="0.70"
STEP6X_COVERAGE="0.80"

# Step 6C: post-Step7 re-alignment
STEP6C_ALPHA="1.0"
STEP6C_EPSILON_GEN="1e-5"
STEP6C_EPSILON_REF="1e-6"
STEP6C_RANDOM_STATE="42"
STEP6C_TARGET_KEEP_RATIO="1.0"
STEP6C_ENSURE_TASK_FLOOR="1"

STEP7_MAX_PER_GROUP="200"
STEP7_MONITOR_RATIO="0.01"
STEP7_RANDOM_STATE="42"

STEP8_POCKET_CUTOFF="6.0"

print_help() {
  cat <<'EOF'
Usage:
  bash run_pipeline_wsl.sh [options]

Options:
  --project-root PATH
  --cif-dir PATH
  --pepbdb-root PATH
  --run-root PATH
  --run-name NAME
  --python BIN
  --workers N
  --chunksize N
  --progress-every N
  --max-tasks N
  --max-candidates N
  --step3-limit-tasks N
  --step6-ref-limit N
  --step6x-cluster-mode MODE
  --step6x-mmseqs-bin PATH
  --map-size-gb N
  --resume-from STEP
  --force-step STEP
  -h, --help

Default profile is a small smoke test:
  workers=2
  progress-every=20
  step3 limit_tasks=200
  step3 max_windows_per_anchor=6
  step4 max_candidates=20000
  step6 ref limit=500
  map_size_gb=4

For a larger run, override the CLI values and edit these defaults if needed:
  STEP3_LIMIT_TASKS=0
  MAX_CANDIDATES=0
  STEP6_REF_LIMIT=0

Useful full-run example:
  bash run_pipeline_wsl.sh --run-name full_run --python /path/to/python --workers 8 --max-candidates 0 --step3-limit-tasks 0 --step6-ref-limit 0 --map-size-gb 64

Step 6X cluster mode:
  --step6x-cluster-mode mmseqs   # current preferred default: MMseqs2 receptor clustering
  --step6x-cluster-mode exact    # safe fallback: exact receptor sequence identity clustering

Supported step keys:
  step1 step2 step3 step4 step5 step6a step6b step6x step7 step6c step8
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) PROJECT_ROOT="$2"; shift 2 ;;
    --cif-dir) CIF_DIR="$2"; shift 2 ;;
    --pepbdb-root) PEPBDB_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --chunksize) CHUNKSIZE="$2"; shift 2 ;;
    --progress-every) PROGRESS_EVERY="$2"; shift 2 ;;
    --max-tasks) MAX_TASKS="$2"; shift 2 ;;
    --max-candidates) MAX_CANDIDATES="$2"; shift 2 ;;
    --step3-limit-tasks) STEP3_LIMIT_TASKS="$2"; shift 2 ;;
    --step6-ref-limit) STEP6_REF_LIMIT="$2"; shift 2 ;;
    --step6x-cluster-mode) STEP6X_CLUSTER_MODE="$2"; shift 2 ;;
    --step6x-mmseqs-bin) STEP6X_MMSEQS_BIN="$2"; shift 2 ;;
    --map-size-gb) MAP_SIZE_GB="$2"; shift 2 ;;
    --resume-from) RESUME_FROM="$2"; shift 2 ;;
    --force-step) FORCE_STEP="$2"; shift 2 ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1" >&2; exit 1 ;;
  esac
done

cd "${PROJECT_ROOT}"

require_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing file: $f" >&2
    exit 1
  fi
}

require_python_module() {
  local module="$1"
  "${PYTHON_BIN}" - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("${module}")
PY
}

echo "================================================================================"
echo "[CHECK] PROJECT_ROOT = ${PROJECT_ROOT}"
echo "[CHECK] CIF_DIR      = ${CIF_DIR}"
echo "[CHECK] PEPBDB_ROOT  = ${PEPBDB_ROOT}"
echo "[CHECK] RUN_ROOT     = ${RUN_ROOT}"
echo "[CHECK] RUN_NAME     = ${RUN_NAME}"
echo "[CHECK] PYTHON       = ${PYTHON_BIN}"
echo "[CHECK] Smoke knobs  : workers=${WORKERS}, step3_limit=${STEP3_LIMIT_TASKS}, step3_max_windows=${STEP3_MAX_WINDOWS_PER_ANCHOR}, step4_max_candidates=${MAX_CANDIDATES}, step6_ref_limit=${STEP6_REF_LIMIT}, map_size_gb=${MAP_SIZE_GB}"
echo "[CHECK] STEP6X_MODE  = ${STEP6X_CLUSTER_MODE}"
echo "[CHECK] RESUME_FROM  = ${RESUME_FROM}"
echo "[CHECK] FORCE_STEP   = ${FORCE_STEP}"
echo "================================================================================"

[[ -d "${CIF_DIR}" ]] || { echo "[ERROR] CIF dir not found: ${CIF_DIR}" >&2; exit 1; }
[[ -d "${PEPBDB_ROOT}" ]] || { echo "[ERROR] PepBDB root not found: ${PEPBDB_ROOT}" >&2; exit 1; }

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "[ERROR] Python binary not found: ${PYTHON_BIN}" >&2; exit 1; }
for module in tqdm gemmi numpy pandas scipy lmdb freesasa; do
  require_python_module "${module}" || {
    echo "[ERROR] Missing Python module: ${module}" >&2
    echo "[ERROR] Please install dependencies into: ${PYTHON_BIN}" >&2
    exit 1
  }
done

if [[ "${STEP6X_CLUSTER_MODE}" == "mmseqs" ]]; then
  if [[ -z "${STEP6X_MMSEQS_BIN}" ]]; then
    echo "[WARN] STEP6X requested mmseqs mode, but no mmseqs binary was provided." >&2
    echo "[WARN] Falling back to exact receptor-sequence clustering for this run." >&2
    STEP6X_CLUSTER_MODE="exact"
  elif ! command -v "${STEP6X_MMSEQS_BIN}" >/dev/null 2>&1 && [[ ! -x "${STEP6X_MMSEQS_BIN}" ]]; then
    echo "[WARN] STEP6X requested mmseqs mode, but binary is not found/executable: ${STEP6X_MMSEQS_BIN}" >&2
    echo "[WARN] Falling back to exact receptor-sequence clustering for this run." >&2
    STEP6X_CLUSTER_MODE="exact"
  fi
fi

echo "[CHECK] STEP6X_EFFECTIVE_MODE = ${STEP6X_CLUSTER_MODE}"

require_file "${STEP1_SCRIPT}"
require_file "${STEP2_SCRIPT}"
require_file "${STEP3_SCRIPT}"
require_file "${STEP4_SCRIPT}"
require_file "${STEP5_SCRIPT}"
require_file "${STEP6_REF_SCRIPT}"
require_file "${STEP6_ALIGN_SCRIPT}"
require_file "${STEP6X_SCRIPT}"
require_file "${STEP6C_SCRIPT}"
require_file "${STEP7_SCRIPT}"
require_file "${STEP8_SCRIPT}"

RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
STEP1_DIR="${RUN_DIR}/step1"
STEP2_DIR="${RUN_DIR}/step2"
STEP3_DIR="${RUN_DIR}/step3"
STEP4_DIR="${RUN_DIR}/step4"
STEP5_DIR="${RUN_DIR}/step5"
STEP6A_DIR="${RUN_DIR}/step6_ref"
STEP6B_DIR="${RUN_DIR}/step6_align"
STEP6X_DIR="${RUN_DIR}/step6_seqcluster"
STEP6C_DIR="${RUN_DIR}/step6c_align"
STEP7_DIR="${RUN_DIR}/step7"
STEP8_DIR="${RUN_DIR}/step8"

mkdir -p \
  "${STEP1_DIR}" "${STEP2_DIR}" "${STEP3_DIR}" "${STEP4_DIR}" \
  "${STEP5_DIR}" "${STEP6A_DIR}" "${STEP6B_DIR}" "${STEP6X_DIR}" "${STEP6C_DIR}" "${STEP7_DIR}" "${STEP8_DIR}"

step_done() {
  local path="$1"
  [[ -s "$path" ]]
}

step_rank() {
  case "$1" in
    step1) echo 1 ;;
    step2) echo 2 ;;
    step3) echo 3 ;;
    step4) echo 4 ;;
    step5) echo 5 ;;
    step6a) echo 6 ;;
    step6b) echo 7 ;;
    step6x) echo 8 ;;
    step7) echo 9 ;;
    step6c) echo 10 ;;
    step8) echo 11 ;;
    *) return 1 ;;
  esac
}

validate_step_key() {
  local key="$1"
  if [[ -n "$key" ]]; then
    step_rank "$key" >/dev/null || {
      echo "[ERROR] Invalid step key: ${key}" >&2
      echo "[ERROR] Supported keys: step1 step2 step3 step4 step5 step6a step6b step6x step7 step6c step8" >&2
      exit 1
    }
  fi
}

should_skip_before_resume() {
  local step_key="$1"
  if [[ -z "${RESUME_FROM}" ]]; then
    return 1
  fi
  local current_rank
  local resume_rank
  current_rank="$(step_rank "${step_key}")"
  resume_rank="$(step_rank "${RESUME_FROM}")"
  [[ "${current_rank}" -lt "${resume_rank}" ]]
}

should_force_from_step() {
  local step_key="$1"
  if [[ -z "${FORCE_STEP}" ]]; then
    return 1
  fi
  local current_rank
  local force_rank
  current_rank="$(step_rank "${step_key}")"
  force_rank="$(step_rank "${FORCE_STEP}")"
  [[ "${current_rank}" -ge "${force_rank}" ]]
}

skip_or_run_msg() {
  local step_name="$1"
  local target="$2"
  local step_key="$3"
  if should_skip_before_resume "${step_key}"; then
    echo "[SKIP] ${step_name} skipped due to --resume-from ${RESUME_FROM}"
    return 0
  fi
  if should_force_from_step "${step_key}"; then
    echo "[RUN ] ${step_name} forced by --force-step ${FORCE_STEP}"
    return 1
  fi
  if step_done "$target"; then
    echo "[SKIP] ${step_name} already completed: ${target}"
    return 0
  fi
  return 1
}

validate_step_key "${RESUME_FROM}"
validate_step_key "${FORCE_STEP}"

# =========================================================
# STEP 1
# =========================================================
echo "================================================================================"
echo "[STEP 1] X-ray QC"
echo "================================================================================"
if ! skip_or_run_msg "STEP 1" "${STEP1_DIR}/step1_summary.json" "step1"; then
  "${PYTHON_BIN}" "${STEP1_SCRIPT}" \
    --input_dir "${CIF_DIR}" \
    --output_dir "${STEP1_DIR}" \
    --max_resolution "${STEP1_MAX_RESOLUTION}" \
    --min_interface_bsa "${STEP1_MIN_INTERFACE_BSA}" \
    --contact_prefilter_cutoff "${STEP1_CONTACT_PREFILTER_CUTOFF}"
fi

# =========================================================
# STEP 2
# =========================================================
echo "================================================================================"
echo "[STEP 2] Directed task generation"
echo "================================================================================"
if ! skip_or_run_msg "STEP 2" "${STEP2_DIR}/step2_tasks.jsonl" "step2"; then
  "${PYTHON_BIN}" "${STEP2_SCRIPT}" \
    --step1_dir "${STEP1_DIR}" \
    --pdb_dir "${CIF_DIR}" \
    --output_jsonl "${STEP2_DIR}/step2_tasks.jsonl" \
    --error_jsonl "${STEP2_DIR}/step2_errors.jsonl" \
    --chain_contact_cutoff "${STEP2_CHAIN_CONTACT_CUTOFF}" \
    --directed_contact_cutoff "${STEP2_DIRECTED_CONTACT_CUTOFF}" \
    --min_source_contact_residues "${STEP2_MIN_SOURCE_CONTACT_RESIDUES}" \
    --min_source_chain_residues "${STEP2_MIN_SOURCE_CHAIN_RESIDUES}" \
    --workers "${WORKERS}" \
    --chunksize "${CHUNKSIZE}" \
    --progress_every "${PROGRESS_EVERY}"
fi

# =========================================================
# STEP 3
# =========================================================
echo "================================================================================"
echo "[STEP 3] Method A multi-anchor candidate pool (updated)"
echo "================================================================================"
STEP3_ARGS=(
  "${STEP3_SCRIPT}"
  --cif_dir "${CIF_DIR}"
  --task_jsonl "${STEP2_DIR}/step2_tasks.jsonl"
  --output_jsonl "${STEP3_DIR}/step3_candidates.jsonl"
  --error_jsonl "${STEP3_DIR}/step3_errors.jsonl"
  --workers "${WORKERS}"
  --chunksize "${CHUNKSIZE}"
  --progress_every "${PROGRESS_EVERY}"
  --anchor_cutoff "${STEP3_ANCHOR_CUTOFF}"
  --contact_cutoff "${STEP3_CONTACT_CUTOFF}"
  --nms_min_seq_gap "${STEP3_NMS_MIN_SEQ_GAP}"
  --max_anchors_per_task "${STEP3_MAX_ANCHORS_PER_TASK}"
  --min_len "${STEP3_MIN_LEN}"
  --max_len "${STEP3_MAX_LEN}"
  --min_contact_residues_6a "${STEP3_MIN_CONTACT_RESIDUES_6A}"
  --min_contact_ratio_6a "${STEP3_MIN_CONTACT_RATIO_6A}"
  --max_windows_per_anchor "${STEP3_MAX_WINDOWS_PER_ANCHOR}"
)
if [[ "${STEP3_LIMIT_TASKS}" != "0" ]]; then
  STEP3_ARGS+=( --limit_tasks "${STEP3_LIMIT_TASKS}" )
fi
if ! skip_or_run_msg "STEP 3" "${STEP3_DIR}/step3_candidates.jsonl" "step3"; then
  "${PYTHON_BIN}" "${STEP3_ARGS[@]}"
fi

# =========================================================
# STEP 4
# =========================================================
echo "================================================================================"
echo "[STEP 4] Postscore candidates"
echo "================================================================================"
STEP4_ARGS=(
  "${STEP4_SCRIPT}"
  --candidate_jsonl "${STEP3_DIR}/step3_candidates.jsonl"
  --pdb_dir "${CIF_DIR}"
  --output_jsonl "${STEP4_DIR}/step4_features.jsonl"
  --error_jsonl "${STEP4_DIR}/step4_errors.jsonl"
  --workers "${WORKERS}"
  --chunksize "${CHUNKSIZE}"
  --progress_every "${PROGRESS_EVERY}"
  --contact_cutoff_6a "${STEP4_CONTACT_CUTOFF_6A}"
  --pocket_cutoff_6a "${STEP4_POCKET_CUTOFF_6A}"
  --min_contact_residues_6a "${STEP4_MIN_CONTACT_RESIDUES_6A}"
  --min_rbsa_raw "${STEP4_MIN_RBSA_RAW}"
)
if [[ "${MAX_CANDIDATES}" != "0" ]]; then
  STEP4_ARGS+=( --max_candidates "${MAX_CANDIDATES}" )
fi
if ! skip_or_run_msg "STEP 4" "${STEP4_DIR}/step4_features.jsonl" "step4"; then
  "${PYTHON_BIN}" "${STEP4_ARGS[@]}"
fi

# =========================================================
# STEP 5
# =========================================================
echo "================================================================================"
echo "[STEP 5] Probabilistic selection"
echo "================================================================================"
if ! skip_or_run_msg "STEP 5" "${STEP5_DIR}/step5_final.jsonl" "step5"; then
  "${PYTHON_BIN}" "${STEP5_SCRIPT}" \
    --step3_jsonl "${STEP3_DIR}/step3_candidates.jsonl" \
    --step4_jsonl "${STEP4_DIR}/step4_features.jsonl" \
    --output_jsonl "${STEP5_DIR}/step5_final.jsonl" \
    --summary_json "${STEP5_DIR}/step5_summary.json" \
    --top_k "${STEP5_TOP_K}" \
    --density_threshold "${STEP5_DENSITY_THRESHOLD}" \
    --min_len "${STEP5_MIN_LEN}" \
    --max_len "${STEP5_MAX_LEN}" \
    --min_contact_residues_6a "${STEP5_MIN_CONTACT_RESIDUES_6A}" \
    --min_rbsa_raw "${STEP5_MIN_RBSA_RAW}" \
    --sampling_power "${STEP5_SAMPLING_POWER}" \
    --length_bonus_strength "${STEP5_LENGTH_BONUS_STRENGTH}"
fi

# =========================================================
# STEP 6A - Build reference dataset from PepBDB
# =========================================================
echo "================================================================================"
echo "[STEP 6A] Build reference dataset from PepBDB"
echo "================================================================================"
STEP6A_ARGS=(
  "${STEP6_REF_SCRIPT}"
  --pepbdb_root "${PEPBDB_ROOT}"
  --out_csv "${STEP6A_DIR}/reference_dataset.csv"
  --out_errors_jsonl "${STEP6A_DIR}/reference_errors.jsonl"
  --workers "${WORKERS}"
  --pocket_cutoff "${STEP6_POCKET_CUTOFF}"
)
if [[ "${STEP6_REF_LIMIT}" != "0" ]]; then
  STEP6A_ARGS+=( --limit "${STEP6_REF_LIMIT}" )
fi
if ! skip_or_run_msg "STEP 6A" "${STEP6A_DIR}/reference_dataset.csv" "step6a"; then
  "${PYTHON_BIN}" "${STEP6A_ARGS[@]}"
fi

# =========================================================
# STEP 6B - Global stratified sampling / alignment
# =========================================================
echo "================================================================================"
echo "[STEP 6B] Global stratified sampling (updated deterministic quota version)"
echo "================================================================================"
STEP6B_ARGS=(
  "${STEP6_ALIGN_SCRIPT}"
  --gen_file "${STEP5_DIR}/step5_final.jsonl"
  --ref_file "${STEP6A_DIR}/reference_dataset.csv"
  --out_survived "${STEP6B_DIR}/step6_survived.jsonl"
  --out_lookup "${STEP6B_DIR}/step6_lookup.csv"
  --out_scored "${STEP6B_DIR}/step6_scored.jsonl"
  --out_summary "${STEP6B_DIR}/step6_summary.json"
  --alpha "${STEP6_ALPHA}"
  --epsilon_gen "${STEP6_EPSILON_GEN}"
  --epsilon_ref "${STEP6_EPSILON_REF}"
  --random_state "${STEP6_RANDOM_STATE}"
  --bins_mode manual
  --target_keep_ratio "${STEP6_TARGET_KEEP_RATIO}"
)
if [[ "${STEP6_ENSURE_TASK_FLOOR}" == "1" ]]; then
  STEP6B_ARGS+=( --ensure_task_floor )
fi
if ! skip_or_run_msg "STEP 6B" "${STEP6B_DIR}/step6_survived.jsonl" "step6b"; then
  "${PYTHON_BIN}" "${STEP6B_ARGS[@]}"
fi

# =========================================================
# STEP 6X - Receptor sequence clustering
# =========================================================
echo "================================================================================"
echo "[STEP 6X] Receptor sequence cluster annotation"
echo "================================================================================"
if ! skip_or_run_msg "STEP 6X" "${STEP6X_DIR}/step6_survived_annotated.jsonl" "step6x"; then
  "${PYTHON_BIN}" "${STEP6X_SCRIPT}" \
    --input_jsonl "${STEP6B_DIR}/step6_survived.jsonl" \
    --cif_dir "${CIF_DIR}" \
    --out_receptors_csv "${STEP6X_DIR}/receptor_sequences.csv" \
    --out_fasta "${STEP6X_DIR}/receptor_sequences.fasta" \
    --out_clusters_csv "${STEP6X_DIR}/receptor_seq_clusters.csv" \
    --out_annotated_jsonl "${STEP6X_DIR}/step6_survived_annotated.jsonl" \
    --out_summary_json "${STEP6X_DIR}/step6x_summary.json" \
    --cluster_mode "${STEP6X_CLUSTER_MODE}" \
    --mmseqs_bin "${STEP6X_MMSEQS_BIN}" \
    --min_seq_id "${STEP6X_MIN_SEQ_ID}" \
    --coverage "${STEP6X_COVERAGE}"
fi

# =========================================================
# STEP 7
# =========================================================
echo "================================================================================"
echo "[STEP 7] Family capping + monitor split"
echo "================================================================================"
if ! skip_or_run_msg "STEP 7" "${STEP7_DIR}/step7_main.jsonl" "step7"; then
  "${PYTHON_BIN}" "${STEP7_SCRIPT}" \
    --input_jsonl "${STEP6X_DIR}/step6_survived_annotated.jsonl" \
    --out_main_jsonl "${STEP7_DIR}/step7_main.jsonl" \
    --out_monitor_jsonl "${STEP7_DIR}/step7_monitor.jsonl" \
    --out_dropped_jsonl "${STEP7_DIR}/step7_dropped.jsonl" \
    --out_summary_json "${STEP7_DIR}/step7_summary.json" \
    --max_per_group "${STEP7_MAX_PER_GROUP}" \
    --monitor_ratio "${STEP7_MONITOR_RATIO}" \
    --random_state "${STEP7_RANDOM_STATE}"
fi

# =========================================================
# STEP 6C - Post-Step7 Global Re-alignment
# =========================================================
echo "================================================================================"
echo "[STEP 6C] Post-Step7 global stratified re-alignment"
echo "================================================================================"
STEP6C_ARGS=(
  "${STEP6C_SCRIPT}"
  --gen_file "${STEP7_DIR}/step7_main.jsonl"
  --donor_file "${STEP7_DIR}/step7_dropped.jsonl"
  --ref_file "${STEP6A_DIR}/reference_dataset.csv"
  --out_survived "${STEP6C_DIR}/step6c_main_survived.jsonl"
  --out_lookup "${STEP6C_DIR}/step6c_main_lookup.csv"
  --out_scored "${STEP6C_DIR}/step6c_main_scored.jsonl"
  --out_summary "${STEP6C_DIR}/step6c_main_summary.json"
  --alpha "${STEP6C_ALPHA}"
  --epsilon_gen "${STEP6C_EPSILON_GEN}"
  --epsilon_ref "${STEP6C_EPSILON_REF}"
  --random_state "${STEP6C_RANDOM_STATE}"
  --bins_mode manual
  --target_keep_ratio "${STEP6C_TARGET_KEEP_RATIO}"
  --max_per_group "${STEP7_MAX_PER_GROUP}"
)
if [[ "${STEP6C_ENSURE_TASK_FLOOR}" == "1" ]]; then
  STEP6C_ARGS+=( --ensure_task_floor )
fi
if ! skip_or_run_msg "STEP 6C" "${STEP6C_DIR}/step6c_main_survived.jsonl" "step6c"; then
  "${PYTHON_BIN}" "${STEP6C_ARGS[@]}"
fi

# =========================================================
# STEP 8
# =========================================================
echo "================================================================================"
echo "[STEP 8] Finalize LMDB"
echo "================================================================================"
if ! skip_or_run_msg "STEP 8" "${STEP8_DIR}/lmdb/final_metadata.jsonl" "step8"; then
  "${PYTHON_BIN}" "${STEP8_SCRIPT}" \
    --input_jsonls "${STEP6C_DIR}/step6c_main_survived.jsonl" "${STEP7_DIR}/step7_monitor.jsonl" \
    --pdb_dir "${CIF_DIR}" \
    --lmdb_dir "${STEP8_DIR}/lmdb" \
    --pocket_cutoff "${STEP8_POCKET_CUTOFF}" \
    --map_size_gb "${MAP_SIZE_GB}" \
    --workers "${WORKERS}" \
    --chunksize "${CHUNKSIZE}" \
    --commit_every 1000
fi

echo "================================================================================"
echo "[DONE] Pipeline finished"
echo "[DONE] Run dir: ${RUN_DIR}"
echo "[DONE] Key outputs:"
echo "  Step5 final        : ${STEP5_DIR}/step5_final.jsonl"
echo "  Step6 ref csv      : ${STEP6A_DIR}/reference_dataset.csv"
echo "  Step6 survived     : ${STEP6B_DIR}/step6_survived.jsonl"
echo "  Step6 scored       : ${STEP6B_DIR}/step6_scored.jsonl"
echo "  Step7 main         : ${STEP7_DIR}/step7_main.jsonl"
echo "  Step7 monitor      : ${STEP7_DIR}/step7_monitor.jsonl"
echo "  Step6C main        : ${STEP6C_DIR}/step6c_main_survived.jsonl"
echo "  Step8 metadata     : ${STEP8_DIR}/lmdb/final_metadata.jsonl"
echo "  Step8 summary      : ${STEP8_DIR}/lmdb/step8_summary.json"
echo "================================================================================"

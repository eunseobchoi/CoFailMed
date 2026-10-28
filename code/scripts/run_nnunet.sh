#!/usr/bin/env bash
# nnU-Net v2 launch driver: five training folds; each fold is evaluated on the fixed held-out test set.
#
# Usage:
#   bash code/scripts/run_nnunet.sh verify              # check install + env
#   bash code/scripts/run_nnunet.sh prepare             # convert datasets + plan_and_preprocess
#   bash code/scripts/run_nnunet.sh prepare_riga        # RIGA Cup only: convert + plan_and_preprocess
#   bash code/scripts/run_nnunet.sh riga_2d_100ep_cv    # 2D 100ep   RIGA Cup   (5x2=10 cells)
#   bash code/scripts/run_nnunet.sh riga_2d_1000ep_cv   # 2D 1000ep  RIGA Cup   (5x2=10 cells)
#   bash code/scripts/run_nnunet.sh acdc_3d_1000ep_cv   # 3D fullres 1000ep ACDC LV (5x2=10 cells)
#   bash code/scripts/run_nnunet.sh all_1000ep_cv       # riga_2d_1000ep_cv then acdc_3d_1000ep_cv
#   bash code/scripts/run_nnunet.sh case_riga_100ep     # RIGA primary split/Magrabia holdout
#
# The paper reports only the 100-epoch case-identical runs (case_riga_100ep and the
# ACDC analogue of run_nnunet_case_worker.sh). The *_1000ep_cv modes are the nnU-Net
# default-length companions and are not reported. prepare_riga exports dataset
# RIGA_DATASET_ID (default 501) with prepare_nnunet_data.py, while case_riga_100ep trains
# dataset RIGA_CASE_DATASET_ID (default 551, the study's id): export and train with the
# same id (code/README.md uses 501 for both).
#
# Environment overrides (defaults use local workspace paths):
#   nnUNet_raw           nnunet_workspace/raw
#   nnUNet_preprocessed  nnunet_workspace/preprocessed
#   nnUNet_results       nnunet_workspace/results
#   FOLDS                "0 1 2 3 4"
#   SEEDS                "13 37"        (5 folds x 2 seeds = 10 cells)
#   PYTHON               python3
#   RIGA_DATASET_ID      501
#   ACDC_DATASET_ID      502
#   RIGA_CUP_INPUT_MANIFEST / ACDC_LV_INPUT_MANIFEST: pinned v2 manifests
#   RIGA_CUP_PREPROCESSING_EVIDENCE / ACDC_LV_PREPROCESSING_EVIDENCE: fresh receipts
#   NNUNET_OUT_ROOT: fresh result root; existing cells are refused
#
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
case "${1:-verify}" in
  verify|prepare|prepare_riga|riga_2d_100ep_cv|case_riga_100ep|riga_2d_1000ep_cv|acdc_3d_1000ep_cv|all_1000ep_cv) ;;
  *) echo "Usage: $0 {verify|prepare|prepare_riga|riga_2d_100ep_cv|case_riga_100ep|riga_2d_1000ep_cv|acdc_3d_1000ep_cv|all_1000ep_cv}" >&2; exit 2 ;;
esac

# --- Locate repo ------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
# Resolve caller-relative paths before changing directory.
for var in RIGA_CUP_INPUT_MANIFEST RIGA_CUP_PREPROCESSING_EVIDENCE ACDC_LV_INPUT_MANIFEST \
    ACDC_LV_PREPROCESSING_EVIDENCE NNUNET_OUT_ROOT NNUNET_LOG_ROOT nnUNet_raw nnUNet_preprocessed \
    nnUNet_results FMPOOL_DATA_ROOT; do
  if [ -n "${!var:-}" ]; then export "${var}=$(realpath -m -- "${!var}")"; fi
done
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-python3}"

# --- nnU-Net workspace ------------------------------------------------------
export nnUNet_raw="${nnUNet_raw:-nnunet_workspace/raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-nnunet_workspace/preprocessed}"
export nnUNet_results="${nnUNet_results:-nnunet_workspace/results}"
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

FOLDS="${FOLDS:-0 1 2 3 4}"
SEEDS="${SEEDS:-13 37}"
RIGA_DATASET_ID="${RIGA_DATASET_ID:-501}"
ACDC_DATASET_ID="${ACDC_DATASET_ID:-502}"

log() { printf '[run_nnunet %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

verify_install() {
  log "verifying nnunetv2 install + env vars"
  if ! "${PYTHON}" -c "import nnunetv2; print('nnunetv2', nnunetv2.__file__)"; then
    echo "ERROR: nnunetv2 not importable. Install with: pip install nnunetv2" >&2
    return 2
  fi
  for var in nnUNet_raw nnUNet_preprocessed nnUNet_results; do
    log "  ${var}=${!var}"
    [ -d "${!var}" ] || { echo "ERROR: ${var} dir missing" >&2; return 3; }
  done
  for cli in nnUNetv2_plan_and_preprocess nnUNetv2_train nnUNetv2_predict; do
    if ! command -v "${cli}" >/dev/null; then
      echo "ERROR: ${cli} not on PATH" >&2
      return 4
    fi
    log "  ${cli} -> $(command -v "${cli}")"
  done
  log "check existing FMPool seeded trainer discovery shim (read-only)"
  "${PYTHON}" -c 'from fmpool.nnunet_seeded import ensure_nnunet_discovery; ensure_nnunet_discovery(write=False)'
  log "verify OK"
}

prepare_dataset() {
  local task="$1" did="$2"
  log "prepare_nnunet_data --task ${task} --dataset-id ${did}"
  "${PYTHON}" "${REPO_ROOT}/scripts/prepare_nnunet_data.py" \
      --task "${task}" --dataset-id "${did}"
}

plan_and_preprocess() {
  local task="$1" did="$2"
  local receipt_var
  receipt_var="$(echo "${task}" | tr '[:lower:]' '[:upper:]')_PREPROCESSING_EVIDENCE"
  : "${!receipt_var:?task preprocessing evidence path required}"
  "${PYTHON}" "${REPO_ROOT}/scripts/prepare_nnunet_data.py" \
    --task "${task}" --dataset-id "${did}" --preprocess \
    --preprocessing-evidence "${!receipt_var}"
}

train_cell() {
  local task="$1" did="$2" cfg="$3" tr="$4" fold="$5" seed="$6" exp="$7"
  local prefix manifest_var receipt_var
  prefix="$(echo "${task}" | tr '[:lower:]' '[:upper:]')"
  manifest_var="${prefix}_INPUT_MANIFEST"
  receipt_var="${prefix}_PREPROCESSING_EVIDENCE"
  : "${!manifest_var:?task input manifest required}"
  : "${!receipt_var:?task preprocessing receipt required}"
  local out_json="${NNUNET_OUT_ROOT:-${REPO_ROOT}/../results/nnunet}/${exp}/${task}_fold${fold}_seed${seed}.json"
  "${PYTHON}" "${REPO_ROOT}/scripts/train_nnunet_baseline.py" \
    --task "${task}" --dataset-id "${did}" --config "${cfg}" \
    --fold "${fold}" --seed "${seed}" --base-trainer "${tr}" \
    --input-manifest "${!manifest_var}" --preprocessing-evidence "${!receipt_var}" \
    --evaluation holdout --out "$(dirname "${out_json}")" --out-json "${out_json}"
}

run_matrix() {
  local task="$1" did="$2" cfg="$3" tr="$4" exp="$5"
  log "==== matrix ${exp}: task=${task} cfg=${cfg} trainer=${tr} ===="
  for fold in ${FOLDS}; do
    for seed in ${SEEDS}; do
      train_cell "${task}" "${did}" "${cfg}" "${tr}" "${fold}" "${seed}" "${exp}"
    done
  done
}

cmd="${1:-verify}"
case "${cmd}" in
  verify)
    verify_install
    ;;
  prepare_riga)
    verify_install
    prepare_dataset riga_cup "${RIGA_DATASET_ID}"
    plan_and_preprocess riga_cup "${RIGA_DATASET_ID}"
    ;;
  prepare)
    verify_install
    prepare_dataset riga_cup "${RIGA_DATASET_ID}"
    prepare_dataset acdc_lv  "${ACDC_DATASET_ID}"
    plan_and_preprocess riga_cup "${RIGA_DATASET_ID}"
    plan_and_preprocess acdc_lv "${ACDC_DATASET_ID}"
    ;;
  riga_2d_100ep_cv)
    run_matrix riga_cup "${RIGA_DATASET_ID}" 2d \
        nnUNetTrainer_100epochs nnunet_2d_100ep
    ;;
  case_riga_100ep)
    verify_install
    RIGA_DATASET_ID="${RIGA_CASE_DATASET_ID:-551}"
    run_matrix riga_cup "${RIGA_DATASET_ID}" 2d \
        nnUNetTrainer_100epochs nnunet_case_riga_2d_100ep
    ;;
  riga_2d_1000ep_cv)
    run_matrix riga_cup "${RIGA_DATASET_ID}" 2d \
        nnUNetTrainer_1000epochs nnunet_2d_1000ep
    ;;
  acdc_3d_1000ep_cv)
    run_matrix acdc_lv "${ACDC_DATASET_ID}" 3d_fullres \
        nnUNetTrainer_1000epochs nnunet_3d_fullres
    ;;
  all_1000ep_cv)
    run_matrix riga_cup "${RIGA_DATASET_ID}" 2d \
        nnUNetTrainer_1000epochs nnunet_2d_1000ep
    run_matrix acdc_lv "${ACDC_DATASET_ID}" 3d_fullres \
        nnUNetTrainer_1000epochs nnunet_3d_fullres
    ;;
  *)
    echo "Usage: $0 {verify|prepare|prepare_riga|riga_2d_100ep_cv|case_riga_100ep|riga_2d_1000ep_cv|acdc_3d_1000ep_cv|all_1000ep_cv}" >&2
    exit 1
    ;;
esac

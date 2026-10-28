#!/usr/bin/env bash
# Distributed worker for the case-identical nnU-Net scope check (RIGA Cup 2D;
# CASE_TASK=acdc_lv for the ACDC LV 3D analogue).
#
# Prepare once per host:
#   PYTHON=/path/to/python bash code/scripts/run_nnunet_case_worker.sh prepare
#
# Then launch one worker per GPU:
#   CUDA_VISIBLE_DEVICES=0 WORKER_ID=0 N_WORKERS=4 PYTHON=/path/to/python \
#     bash code/scripts/run_nnunet_case_worker.sh worker
#
# The worker grid is 5 folds x 2 seeds (13, 37) using the nnU-Net v2 2D
# seeded trainer. By default this runs the 100-epoch scope check; set
# TRAINER_EPOCHS=1000 and EXP_NAME=nnunet_case_riga_2d_1000ep for the
# default-length companion (not reported in the paper). Training folds use BinRushed+MESSIDOR; held-out
# scoring uses the same Magrabia cases as the primary RIGA frozen-encoder row.
# CASE_TASK=acdc_lv runs the ACDC 3d_fullres analogue (inner CV over the primary
# training patients, every cell scored on the primary test patients).
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
case "${1:-worker}" in
  prepare) ;;
  worker) : "${WORKER_ID:?WORKER_ID required}" "${N_WORKERS:?N_WORKERS required}" ;;
  *) echo "Usage: $0 {prepare|worker}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
# Resolve caller-relative paths before changing directory.
for var in RIGA_CUP_INPUT_MANIFEST RIGA_CUP_PREPROCESSING_EVIDENCE ACDC_LV_INPUT_MANIFEST \
    ACDC_LV_PREPROCESSING_EVIDENCE NNUNET_OUT_ROOT NNUNET_LOG_ROOT nnUNet_raw nnUNet_preprocessed \
    nnUNet_results FMPOOL_DATA_ROOT; do
  if [ -n "${!var:-}" ]; then export "${var}=$(realpath -m -- "${!var}")"; fi
done
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"
PYBIN="$( dirname "${PYTHON}" )"
export PATH="${PYBIN}:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export FMPOOL_DATA_ROOT="${FMPOOL_DATA_ROOT:-${REPO_ROOT}/data/raw}"
export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnunet_case_workspace/raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnunet_case_workspace/preprocessed}"
export nnUNet_results="${nnUNet_results:-${REPO_ROOT}/nnunet_case_workspace/results}"

RIGA_CASE_DATASET_ID="${RIGA_CASE_DATASET_ID:-551}"
CASE_TASK="${CASE_TASK:-riga_cup}"
case "${CASE_TASK}" in
  riga_cup) CASE_DATASET_ID="${RIGA_CASE_DATASET_ID}"; CASE_CONFIG=2d; CASE_TAG=riga_2d ;;
  acdc_lv) CASE_DATASET_ID="${ACDC_CASE_DATASET_ID:-502}"; CASE_CONFIG=3d_fullres; CASE_TAG=acdc_3d ;;
  *) echo "CASE_TASK must be riga_cup or acdc_lv, got ${CASE_TASK}" >&2; exit 2 ;;
esac
TRAINER_EPOCHS="${TRAINER_EPOCHS:-100}"
if [ "${TRAINER_EPOCHS}" != "100" ] && [ "${TRAINER_EPOCHS}" != "1000" ]; then
  echo "TRAINER_EPOCHS must be 100 or 1000, got ${TRAINER_EPOCHS}" >&2
  exit 2
fi
BASE_TRAINER="nnUNetTrainer_${TRAINER_EPOCHS}epochs"
EXP_NAME="${EXP_NAME:-nnunet_case_${CASE_TAG}_${TRAINER_EPOCHS}ep}"
OUT_DIR="${NNUNET_OUT_ROOT:-${REPO_ROOT}/../results/nnunet}/${EXP_NAME}"
LOG_DIR="${NNUNET_LOG_ROOT:-${REPO_ROOT}/../results/logs}/${EXP_NAME}"
if [ "${1:-worker}" = worker ]; then
  prefix="$(echo "${CASE_TASK}" | tr '[:lower:]' '[:upper:]')"
  manifest_var="${prefix}_INPUT_MANIFEST"; receipt_var="${prefix}_PREPROCESSING_EVIDENCE"
  : "${!manifest_var:?pinned input manifest required}" "${!receipt_var:?preprocessing receipt required}"
fi
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}" "${OUT_DIR}" "${LOG_DIR}"

log() {
  printf '[nnunet_case %s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

verify_env() {
  log "python=${PYTHON}"
  "${PYTHON}" - <<'PY'
import nnunetv2
import torch
print("nnunetv2", nnunetv2.__file__)
print("torch_cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY
  for cli in nnUNetv2_plan_and_preprocess nnUNetv2_train nnUNetv2_predict; do
    command -v "${cli}" >/dev/null
    log "${cli}=$(command -v "${cli}")"
  done
}

prepare_case_dataset() {
  if [ "${CASE_TASK}" = riga_cup ]; then
    RIGA_DATASET_ID="${CASE_DATASET_ID}" \
      bash "${SCRIPT_DIR}/run_nnunet.sh" prepare_riga
  else
    : "${ACDC_LV_PREPROCESSING_EVIDENCE:?preprocessing receipt path required}"
    verify_env
    "${PYTHON}" "${SCRIPT_DIR}/prepare_nnunet_data.py" \
      --task "${CASE_TASK}" --dataset-id "${CASE_DATASET_ID}" \
      --split-json "${REPO_ROOT}/data/splits/acdc_lv_nnunet_official_split.json"
    "${PYTHON}" "${SCRIPT_DIR}/prepare_nnunet_data.py" \
      --task "${CASE_TASK}" --dataset-id "${CASE_DATASET_ID}" --preprocess \
      --preprocessing-evidence "${ACDC_LV_PREPROCESSING_EVIDENCE}"
  fi
}

run_cell() {
  local fold="$1" seed="$2"
  local prefix manifest_var receipt_var
  prefix="$(echo "${CASE_TASK}" | tr '[:lower:]' '[:upper:]')"
  manifest_var="${prefix}_INPUT_MANIFEST"
  receipt_var="${prefix}_PREPROCESSING_EVIDENCE"
  : "${!manifest_var:?pinned input manifest required}"
  : "${!receipt_var:?preprocessing receipt required}"
  local out_json="${OUT_DIR}/${CASE_TASK}_fold${fold}_seed${seed}.json"
  "${PYTHON}" "${REPO_ROOT}/scripts/train_nnunet_baseline.py" \
    --task "${CASE_TASK}" --dataset-id "${CASE_DATASET_ID}" --config "${CASE_CONFIG}" \
    --fold "${fold}" --seed "${seed}" --base-trainer "${BASE_TRAINER}" \
    --input-manifest "${!manifest_var}" \
    --preprocessing-evidence "${!receipt_var}" \
    --evaluation holdout --out "${OUT_DIR}" --out-json "${out_json}"
}

run_worker() {
  verify_env
  local worker_id="${WORKER_ID:?WORKER_ID required}"
  local n_workers="${N_WORKERS:?N_WORKERS required}"
  log "worker ${worker_id}/${n_workers} starting"
  local idx=0
  local fold seed
  for fold in 0 1 2 3 4; do
    for seed in 13 37; do
      if [ $(( idx % n_workers )) -eq "${worker_id}" ]; then
        run_cell "${fold}" "${seed}" 2>&1 | tee -a "${LOG_DIR}/worker_${worker_id}.log"
      fi
      idx=$((idx + 1))
    done
  done
  log "worker ${worker_id}/${n_workers} done"
}

cmd="${1:-worker}"
case "${cmd}" in
  prepare)
    prepare_case_dataset
    ;;
  worker)
    run_worker
    ;;
  *)
    echo "Usage: $0 {prepare|worker}" >&2
    exit 2
    ;;
esac

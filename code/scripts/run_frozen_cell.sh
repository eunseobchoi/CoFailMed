#!/usr/bin/env bash
# One frozen-encoder cell in FP32: fresh run directory, verified cache, all four fixed seeds.
# Usage: bash run_frozen_cell.sh TASK FM INPUT_MANIFEST NEW_RUN_DIR
# FMPOOL_CACHE_ROOT (feature cache) and FMPOOL_DATA_ROOT (raw data) must be set.
set -euo pipefail
TASK=${1:?'TASK required'}
FM=${2:?'FM required'}
INPUT_MANIFEST=$(realpath "${3:?'INPUT_MANIFEST required'}")
RUN_ROOT=$(realpath -m "${4:?'NEW_RUN_DIR required'}")
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${FMPOOL_PYTHON:-python3}
CACHE_ROOT=${FMPOOL_CACHE_ROOT:?'FMPOOL_CACHE_ROOT required'}
[[ -f "$INPUT_MANIFEST" ]] || exit 2
mkdir -p "$(dirname "$RUN_ROOT")"
mkdir "$RUN_ROOT" # No overwriting, resuming into, or skipping an existing run.
printf 'running\n' > "$RUN_ROOT/status"
trap 'rc=$?; if ((rc != 0)); then printf "failed rc=%s\n" "$rc" > "$RUN_ROOT/status"; fi' EXIT
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Preserve the scheduler's CUDA_VISIBLE_DEVICES; never replace its UUID with an index.
printf 'task=%s\nfm=%s\ninput_manifest=%s\ncuda_visible_devices=%s\n' \
  "$TASK" "$FM" "$INPUT_MANIFEST" "${CUDA_VISIBLE_DEVICES:-unset}" > "$RUN_ROOT/launch.txt"
"$PYTHON" "$REPO_ROOT/scripts/extract_features.py" \
  --task "$TASK" --fm "$FM" --input-manifest "$INPUT_MANIFEST" \
  --cache-root "$CACHE_ROOT" --batch-size 16 --num-workers 4 --device cuda \
  > "$RUN_ROOT/extract.log" 2>&1
manifest="$CACHE_ROOT/$TASK/$FM/manifest.json"
manifest_sha=$(sha256sum "$manifest")
manifest_sha=${manifest_sha%% *}
for seed in 42 43 44 45; do
  "$PYTHON" "$REPO_ROOT/scripts/train_head_cached.py" \
    --task "$TASK" --fm "$FM" --seed "$seed" --device cuda \
    --cache-root "$CACHE_ROOT" --cache-manifest-sha256 "$manifest_sha" \
    --out "$RUN_ROOT/per_case_dice/$TASK/$FM" \
    --checkpoint-dir "$RUN_ROOT/checkpoints/$TASK/$FM" \
    --epochs 30 --batch-size 16 > "$RUN_ROOT/seed_${seed}.log" 2>&1
done
printf 'completed\n' > "$RUN_ROOT/status"

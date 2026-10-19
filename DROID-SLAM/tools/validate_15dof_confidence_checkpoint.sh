#!/usr/bin/env bash
set -euo pipefail

# Evaluate exactly one saved training checkpoint on the fixed V1_02 validation
# sequence, then update the state used by the segmented early-stop controller.

if (( $# != 1 )); then
  echo "usage: $0 CHECKPOINT" >&2
  exit 2
fi

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
CHECKPOINT="$1"
RUN_NAME="${EARLY_STOP_RUN_NAME:?Set EARLY_STOP_RUN_NAME to the training run name}"
VALIDATION_ROOT="${EARLY_STOP_VALIDATION_ROOT:-/home/ubuntu/SLAM/datasets/euroc_validation/prepared/V1_02_medium_droid_cam0}"
OUT_ROOT="${EARLY_STOP_OUT_ROOT:-$ROOT/outputs/$RUN_NAME/early_stop_v1_02}"
STATE_PATH="${EARLY_STOP_STATE_PATH:-$OUT_ROOT/state.json}"
MIN_COVERAGE_RATIO="${EARLY_STOP_MIN_COVERAGE_RATIO:-0.95}"
DETERIORATION_TOLERANCE_M="${EARLY_STOP_DETERIORATION_TOLERANCE_M:-0.0}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-sim3_ate_rmse_m}"

export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "checkpoint does not exist: $CHECKPOINT" >&2
  exit 2
fi
for required in data calib.txt frames.csv imu_prior_fd_jacobian.csv cam0_sensor.yaml imu0_sensor.yaml; do
  if [[ ! -e "$VALIDATION_ROOT/$required" ]]; then
    echo "validation input is missing: $VALIDATION_ROOT/$required" >&2
    exit 2
  fi
done

tag="$(basename "$CHECKPOINT" .pth)"
evaluation_dir="$OUT_ROOT/$tag"
if [[ -e "$evaluation_dir" ]]; then
  echo "refusing to overwrite existing checkpoint evaluation: $evaluation_dir" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT"

DATA_ROOT="$VALIDATION_ROOT" \
CHECKPOINTS="$CHECKPOINT" \
STRIDE=2 \
IMU_CONFIDENCE_ONLY_WEIGHTING=1 \
DVI_IMU_QUALITY_FUSION=1 \
OUT_DIR="$evaluation_dir" \
bash tools/select_dvi_stage3_checkpoint.sh

"$PYTHON" tools/monitor_checkpoint_early_stop.py \
  --summary-common "$evaluation_dir/summary_common.csv" \
  --summary-full "$evaluation_dir/summary_full.csv" \
  --checkpoint "$CHECKPOINT" \
  --state-path "$STATE_PATH" \
  --metric "$EARLY_STOP_METRIC" \
  --min-coverage-ratio "$MIN_COVERAGE_RATIO" \
  --deterioration-tolerance-m "$DETERIORATION_TOLERANCE_M"

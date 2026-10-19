#!/usr/bin/env bash
set -euo pipefail

# Matched DVI-SLAM IMU-only four-way ablation.  Feature-metric is deliberately
# absent.  All four arms use the exact same trained checkpoint and sequence;
# only the IMU confidence multiplier w^u differs.
#
#   visual : no IMU factor
#   fixed  : w^u = 1
#   tuned  : manually supplied w^u schedule per outer DBA update
#   learned: DVI scalar 1x1 -> spatial-average confidence head

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to a prepared EuRoC *_droid_cam0 directory}"
IMAGEDIR="${IMAGEDIR:-${DATA_ROOT}/data}"
CALIB="${CALIB:-${DATA_ROOT}/calib.txt}"
IMU_PRIOR="${IMU_PRIOR:-${DATA_ROOT}/imu_prior_fd_jacobian.csv}"
FRAMES_CSV="${FRAMES_CSV:-${DATA_ROOT}/frames.csv}"
CAMERA_SENSOR_YAML="${CAMERA_SENSOR_YAML:-${DATA_ROOT}/cam0_sensor.yaml}"
IMU_SENSOR_YAML="${IMU_SENSOR_YAML:-${DATA_ROOT}/imu0_sensor.yaml}"
GT_CSV="${GT_CSV:-}"

# This is the output of tools/run_dvi_scalar_imu_confidence_finetune.sh.
WEIGHTS="${WEIGHTS:-checkpoints/dvi_scalar_imu_confidence_3000_final_003000.pth}"
MAX_FRAMES="${MAX_FRAMES:-0}"
STRIDE="${STRIDE:-2}"
BUFFER="${BUFFER:-4096}"
IMU_BA_WEIGHT="${IMU_BA_WEIGHT:-0.002}"
# Candidate hand-tuned control.  It is intentionally exposed as an environment
# variable: record any schedule selected on a held-out sequence before using
# this script on the reporting sequence.
TUNED_SCHEDULE="${TUNED_SCHEDULE:-0.25 0.50 0.75 1.00}"
OUT_DIR="${OUT_DIR:-outputs/euroc_dvi_imu_4way_$(basename "${DATA_ROOT}")_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "$GT_CSV" && -f "$DATA_ROOT/source.txt" ]]; then
  GT_CSV="$(sed -n 's/^groundtruth=//p' "$DATA_ROOT/source.txt" | head -n 1)"
fi
if [[ -z "$GT_CSV" && -f "$DATA_ROOT/source.txt" ]]; then
  SOURCE_MAV0="$(sed -n 's/^source_mav0=//p' "$DATA_ROOT/source.txt" | head -n 1)"
  if [[ -n "$SOURCE_MAV0" ]]; then
    GT_CSV="$SOURCE_MAV0/state_groundtruth_estimate0/data.csv"
  fi
fi

for path in "$IMAGEDIR" "$CALIB" "$IMU_PRIOR" "$FRAMES_CSV" \
  "$CAMERA_SENSOR_YAML" "$IMU_SENSOR_YAML" "$GT_CSV" "$WEIGHTS"; do
  if [[ ! -e "$path" ]]; then
    echo "required input does not exist: $path" >&2
    exit 2
  fi
done

cd "$ROOT"
mkdir -p "$OUT_DIR"

{
  echo "weights=$WEIGHTS"
  echo "data_root=$DATA_ROOT"
  echo "max_frames=$MAX_FRAMES"
  echo "stride=$STRIDE"
  echo "imu_ba_weight=$IMU_BA_WEIGHT"
  echo "tuned_schedule=$TUNED_SCHEDULE"
  git rev-parse HEAD
  git status --short
} > "$OUT_DIR/run_manifest.txt"

MAX_ARGS=()
if [[ "$MAX_FRAMES" != "0" ]]; then
  MAX_ARGS=(--max_frames "$MAX_FRAMES")
fi

RUN_ARGS=()
run_case() {
  local label="$1"
  shift
  local recon="$OUT_DIR/${label}_recon.pth"
  local debug="$OUT_DIR/${label}_debug.csv"

  echo "[RUN] $label"
  "$PYTHON" demo.py \
    --imagedir "$IMAGEDIR" \
    --calib "$CALIB" \
    --weights "$WEIGHTS" \
    --stride "$STRIDE" \
    "${MAX_ARGS[@]}" \
    --buffer "$BUFFER" \
    --disable_vis \
    --reconstruction_path "$recon" \
    "$@"

  if [[ -f "$debug" ]]; then
    RUN_ARGS+=(--run "$label=$recon,$debug")
  else
    RUN_ARGS+=(--run "$label=$recon")
  fi
}

IMU_COMMON_ARGS=(
  --imu_prior "$IMU_PRIOR"
  --imu_frame_manifest "$FRAMES_CSV"
  --cam_sensor_yaml "$CAMERA_SENSOR_YAML"
  --imu_sensor_yaml "$IMU_SENSOR_YAML"
  --use_two_stage_imu_ba
  --use_imu_noise_model
  --imu_ba_prior_weight "$IMU_BA_WEIGHT"
)

# A keeps the same visual checkpoint and the same startup-frame policy while
# disabling every IMU factor.  Do not add IMU flags to this invocation.
run_case visual --visual_bootstrap_control

# B + IMU fixed: w^pu from the existing IMU-noise path, w^u exactly one.
run_case imu_fixed \
  "${IMU_COMMON_ARGS[@]}" \
  --imu_ba_debug \
  --imu_ba_debug_path "$OUT_DIR/imu_fixed_debug.csv"

# B + IMU tuned: manual per-outer-DBA schedule, no learned confidence branch.
# shellcheck disable=SC2206
TUNED_SCHEDULE_ARGS=($TUNED_SCHEDULE)
run_case imu_tuned \
  "${IMU_COMMON_ARGS[@]}" \
  --imu_tuned_confidence_schedule "${TUNED_SCHEDULE_ARGS[@]}" \
  --imu_ba_debug \
  --imu_ba_debug_path "$OUT_DIR/imu_tuned_debug.csv"

# B + IMU learned: the only learned term is scalar w^u from the DVI head.
run_case imu_learned \
  "${IMU_COMMON_ARGS[@]}" \
  --use_learned_imu_confidence \
  --use_dvi_imu_confidence \
  --imu_confidence_floor 0 \
  --imu_ba_debug \
  --imu_ba_debug_path "$OUT_DIR/imu_learned_debug.csv"

EVAL_ARGS=(
  --frames-csv "$FRAMES_CSV"
  --gt-csv "$GT_CSV"
  --gt-frame body
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML"
  "${RUN_ARGS[@]}"
)

# The common-frame report is the comparison to cite; the full report exposes
# tracking/coverage differences rather than silently hiding them.
"$PYTHON" tools/evaluate_euroc_recon.py \
  --common-frame-ids \
  --output-csv "$OUT_DIR/summary_common.csv" \
  --output-md "$OUT_DIR/summary_common.md" \
  "${EVAL_ARGS[@]}"
"$PYTHON" tools/evaluate_euroc_recon.py \
  --output-csv "$OUT_DIR/summary_full.csv" \
  --output-md "$OUT_DIR/summary_full.md" \
  "${EVAL_ARGS[@]}"

echo "[DONE] $OUT_DIR"
echo "matched report: $OUT_DIR/summary_common.md"
echo "full report:    $OUT_DIR/summary_full.md"

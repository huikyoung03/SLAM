#!/usr/bin/env bash
set -euo pipefail

# One held-out, apples-to-apples comparison:
#   stock visual DROID  vs.  one 15DoF confidence-only checkpoint.
#
# The reporting sequence is never used for checkpoint selection.  This runner
# intentionally has no fixed-IMU or tuned-IMU arm: its report answers only
# whether the learned model beats the requested visual-DROID baseline.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to prepared V1_03/V2_03 *_droid_cam0}"
WEIGHTS="${WEIGHTS:?Set WEIGHTS to one learned 15DoF checkpoint}"
ORIGINAL_WEIGHTS="${ORIGINAL_WEIGHTS:-droid.pth}"
# An existing stock-DROID reconstruction from the identical data/stride can
# be reused when only the learned arm's runtime wiring changed. This avoids a
# needless second visual-only run while keeping the evaluator's common-frame
# comparison exactly the same.
ORIGINAL_RECON="${ORIGINAL_RECON:-}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR to a new output directory}"
IMAGEDIR="${IMAGEDIR:-${DATA_ROOT}/data}"
CALIB="${CALIB:-${DATA_ROOT}/calib.txt}"
IMU_PRIOR="${IMU_PRIOR:-${DATA_ROOT}/imu_prior.csv}"
FRAMES_CSV="${FRAMES_CSV:-${DATA_ROOT}/frames.csv}"
CAMERA_SENSOR_YAML="${CAMERA_SENSOR_YAML:-${DATA_ROOT}/cam0_sensor.yaml}"
IMU_SENSOR_YAML="${IMU_SENSOR_YAML:-${DATA_ROOT}/imu0_sensor.yaml}"
GT_CSV="${GT_CSV:-}"
STRIDE="${STRIDE:-2}"
BUFFER="${BUFFER:-4096}"
MAX_FRAMES="${MAX_FRAMES:-0}"
# ``learned`` is the normal submission-facing comparison. ``fixed_one`` is
# a diagnostic control: it uses the identical 15DoF objective but supplies
# c_ij=1 for every valid IMU factor, without loading or invoking a confidence
# head. It isolates whether a result is limited by confidence prediction or
# by the underlying inertial BA/calibration.
CONFIDENCE_MODE="${CONFIDENCE_MODE:-learned}"
# A residual-calibrated checkpoint must opt in at runtime.  Leave this off by
# default so prior comparisons retain their exact fixed p/v settings.
LEARNED_RESIDUAL_WEIGHTS="${LEARNED_RESIDUAL_WEIGHTS:-0}"
# ``dense`` reproduces the DVI/DROID benchmark convention: evaluate the
# PoseTrajectoryFiller output at every input frame, not only retained
# MotionFilter keyframes. Existing reconstructions without dense export must
# be regenerated before this mode can be used.
TRAJECTORY_SOURCE="${TRAJECTORY_SOURCE:-keyframes}"

export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

if [[ -z "$GT_CSV" && -f "$DATA_ROOT/source.txt" ]]; then
  GT_CSV="$(sed -n 's/^groundtruth=//p' "$DATA_ROOT/source.txt" | head -n 1)"
fi
if [[ -z "$GT_CSV" && -f "$DATA_ROOT/source.txt" ]]; then
  source_mav0="$(sed -n 's/^source_mav0=//p' "$DATA_ROOT/source.txt" | head -n 1)"
  if [[ -n "$source_mav0" ]]; then
    GT_CSV="$source_mav0/state_groundtruth_estimate0/data.csv"
  fi
fi

for path in "$IMAGEDIR" "$CALIB" "$IMU_PRIOR" "$FRAMES_CSV" \
  "$CAMERA_SENSOR_YAML" "$IMU_SENSOR_YAML" "$GT_CSV" "$WEIGHTS"; do
  if [[ ! -e "$path" ]]; then
    echo "required input does not exist: $path" >&2
    exit 2
  fi
done
if [[ -n "$ORIGINAL_RECON" ]]; then
  if [[ ! -f "$ORIGINAL_RECON" ]]; then
    echo "ORIGINAL_RECON does not exist: $ORIGINAL_RECON" >&2
    exit 2
  fi
elif [[ ! -f "$ORIGINAL_WEIGHTS" ]]; then
  echo "required input does not exist: $ORIGINAL_WEIGHTS" >&2
  exit 2
fi
if [[ -e "$OUT_DIR" ]]; then
  echo "refusing to overwrite existing output directory: $OUT_DIR" >&2
  exit 2
fi
if ! [[ "$STRIDE" =~ ^[1-9][0-9]*$ ]]; then
  echo "STRIDE must be a positive integer, got: $STRIDE" >&2
  exit 2
fi
if [[ "$CONFIDENCE_MODE" != "learned" && "$CONFIDENCE_MODE" != "fixed_one" ]]; then
  echo "CONFIDENCE_MODE must be learned or fixed_one, got: $CONFIDENCE_MODE" >&2
  exit 2
fi
if [[ "$LEARNED_RESIDUAL_WEIGHTS" != "0" && "$LEARNED_RESIDUAL_WEIGHTS" != "1" ]]; then
  echo "LEARNED_RESIDUAL_WEIGHTS must be 0 or 1, got: $LEARNED_RESIDUAL_WEIGHTS" >&2
  exit 2
fi
if [[ "$CONFIDENCE_MODE" == "fixed_one" && "$LEARNED_RESIDUAL_WEIGHTS" == "1" ]]; then
  echo "LEARNED_RESIDUAL_WEIGHTS=1 is only valid with CONFIDENCE_MODE=learned" >&2
  exit 2
fi
if [[ "$TRAJECTORY_SOURCE" != "keyframes" && "$TRAJECTORY_SOURCE" != "dense" ]]; then
  echo "TRAJECTORY_SOURCE must be keyframes or dense, got: $TRAJECTORY_SOURCE" >&2
  exit 2
fi

case "$CONFIDENCE_MODE" in
  learned)
    RUN_LABEL="learned_15dof_confidence"
    CONFIDENCE_ARGS=(
      --imu_confidence_only_weighting
      --use_learned_imu_confidence
      --use_dvi_imu_confidence
      --use_dvi_imu_quality_fusion
      --imu_confidence_floor 0
    )
    ;;
  fixed_one)
    RUN_LABEL="fixed_c1_15dof"
    CONFIDENCE_ARGS=(
      --imu_ba_prior_weight 1.0
      --imu_tuned_confidence_schedule 1.0
    )
    ;;
esac

if [[ "$LEARNED_RESIDUAL_WEIGHTS" == "1" ]]; then
  CONFIDENCE_ARGS+=(--use_learned_imu_residual_weights)
fi

mkdir -p "$OUT_DIR"
{
  echo "purpose=held-out stock-visual-DROID versus 15DoF-IMU-control-or-confidence"
  echo "data_root=$(realpath "$DATA_ROOT")"
  echo "imu_prior=$(realpath "$IMU_PRIOR")"
  echo "weights=$(realpath "$WEIGHTS")"
  echo "original_weights=$(realpath "$ORIGINAL_WEIGHTS")"
  echo "original_recon=${ORIGINAL_RECON:-generated_in_this_run}"
  echo "stride=$STRIDE"
  echo "buffer=$BUFFER"
  echo "max_frames=$MAX_FRAMES"
  echo "confidence_mode=$CONFIDENCE_MODE"
  echo "learned_residual_weights=$LEARNED_RESIDUAL_WEIGHTS"
  echo "trajectory_source=$TRAJECTORY_SOURCE"
  git rev-parse HEAD
  git status --short
} > "$OUT_DIR/run_manifest.txt"

max_args=()
if [[ "$MAX_FRAMES" != "0" ]]; then
  max_args=(--max_frames "$MAX_FRAMES")
fi

if [[ -n "$ORIGINAL_RECON" ]]; then
  original_recon="$ORIGINAL_RECON"
  echo "[REUSE] original_droid reconstruction: $original_recon"
else
  original_recon="$OUT_DIR/original_droid_recon.pth"
  echo "[RUN] original_droid"
  "$PYTHON" demo.py \
    --imagedir "$IMAGEDIR" \
    --calib "$CALIB" \
    --weights "$ORIGINAL_WEIGHTS" \
    --stride "$STRIDE" \
    "${max_args[@]}" \
    --buffer "$BUFFER" \
    --disable_vis \
    --reconstruction_path "$original_recon"
fi

echo "[RUN] $RUN_LABEL"
# Training uses frozen keyframe manifests selected by the ordinary DROID
# MotionFilter at this same stride. Keep the runtime selection policy
# identical: two-stage 15DoF still performs VI alignment and full IMU BA, but
# does not force additional startup keyframes that the visual baseline and
# training clips never received.
"$PYTHON" demo.py \
  --imagedir "$IMAGEDIR" \
  --calib "$CALIB" \
  --weights "$WEIGHTS" \
  --stride "$STRIDE" \
  "${max_args[@]}" \
  --buffer "$BUFFER" \
  --disable_vis \
  --reconstruction_path "$OUT_DIR/${RUN_LABEL}_recon.pth" \
  --imu_prior "$IMU_PRIOR" \
  --imu_frame_manifest "$FRAMES_CSV" \
  --cam_sensor_yaml "$CAMERA_SENSOR_YAML" \
  --imu_sensor_yaml "$IMU_SENSOR_YAML" \
  --use_two_stage_imu_ba \
  --no_vi_init_bootstrap \
  --use_imu_noise_model \
  "${CONFIDENCE_ARGS[@]}" \
  --imu_ba_debug \
  --imu_ba_debug_path "$OUT_DIR/${RUN_LABEL}_debug.csv"

eval_args=(
  --frames-csv "$FRAMES_CSV"
  --gt-csv "$GT_CSV"
  --gt-frame body
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML"
  --run "original_droid=$original_recon"
  --run "$RUN_LABEL=$OUT_DIR/${RUN_LABEL}_recon.pth,$OUT_DIR/${RUN_LABEL}_debug.csv"
)

"$PYTHON" tools/evaluate_euroc_recon.py \
  --trajectory-source "$TRAJECTORY_SOURCE" \
  --common-frame-ids \
  --output-csv "$OUT_DIR/summary_common.csv" \
  --output-md "$OUT_DIR/summary_common.md" \
  "${eval_args[@]}"
"$PYTHON" tools/evaluate_euroc_recon.py \
  --trajectory-source "$TRAJECTORY_SOURCE" \
  --output-csv "$OUT_DIR/summary_full.csv" \
  --output-md "$OUT_DIR/summary_full.md" \
  "${eval_args[@]}"

echo "[DONE] $OUT_DIR"
echo "[DONE] comparison=$OUT_DIR/summary_common.md"

#!/usr/bin/env bash
set -euo pipefail

# EuRoC 4-way ablation:
#
# 1. original_droid
# 2. visual_bootstrap_control
# 3. two_stage_15dof
# 4. two_stage_15dof_reintegrated
#
# The fourth case uses an IMU prior that was re-preintegrated around the
# bias estimate obtained from the first full 15DoF VI run.
#
# Purpose:
#   - separate startup/keyframe-retention effect
#   - measure original full 15DoF VI effect
#   - measure bias relinearization + re-preintegration effect


ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"

export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

DATA_ROOT="${DATA_ROOT:-/home/ubuntu/SLAM/uploads/euroc_v1_03_droid_cam0}"

IMAGEDIR="${IMAGEDIR:-${DATA_ROOT}/data}"
CALIB="${CALIB:-${DATA_ROOT}/calib.txt}"

FRAMES_CSV="${FRAMES_CSV:-${DATA_ROOT}/frames.csv}"

CAMERA_SENSOR_YAML="${CAMERA_SENSOR_YAML:-${DATA_ROOT}/cam0_sensor.yaml}"
IMU_SENSOR_YAML="${IMU_SENSOR_YAML:-${DATA_ROOT}/imu0_sensor.yaml}"

GT_CSV="${GT_CSV:-}"


# ----------------------------------------------------------------------
# IMU priors
# ----------------------------------------------------------------------

# Pass 1:
# Prior generated around the original bias linearization point.
IMU_PRIOR="${IMU_PRIOR:-${DATA_ROOT}/imu_prior_fd_jacobian.csv}"

# Pass 2:
# Prior regenerated after VI initialization / 15DoF bias estimation.
#
# You can override this when running:
#
# REINTEGRATED_IMU_PRIOR=/path/to/new_prior.csv \
# bash tools/run_euroc_reintegrated_15dof_validation.sh
#
REINTEGRATED_IMU_PRIOR="${REINTEGRATED_IMU_PRIOR:-${DATA_ROOT}/imu_prior_fd_jacobian_reintegrated.csv}"


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

ORIGINAL_WEIGHTS="${ORIGINAL_WEIGHTS:-droid.pth}"

# Physics / optimizer validation.
# No learned IMU confidence weights are used here.
TWO_STAGE_WEIGHTS="${TWO_STAGE_WEIGHTS:-${ORIGINAL_WEIGHTS}}"


# ----------------------------------------------------------------------
# Evaluation settings
# ----------------------------------------------------------------------

MAX_FRAMES="${MAX_FRAMES:-450}"
STRIDE="${STRIDE:-2}"
BUFFER="${BUFFER:-1024}"

RUN_PLY="${RUN_PLY:-0}"

OUT_DIR="${OUT_DIR:-outputs/euroc_v1_03_reintegrated_15dof_$(date +%Y%m%d_%H%M%S)}"


# ----------------------------------------------------------------------
# Ground truth discovery
# ----------------------------------------------------------------------

if [[ -z "$GT_CSV" && -f "$DATA_ROOT/source.txt" ]]; then
  GT_CSV="$(
    sed -n 's/^groundtruth=//p' "$DATA_ROOT/source.txt" |
    head -n 1
  )"
fi

if [[ -z "$GT_CSV" && -f "$DATA_ROOT/source.txt" ]]; then
  SOURCE_MAV0="$(
    sed -n 's/^source_mav0=//p' "$DATA_ROOT/source.txt" |
    head -n 1
  )"

  if [[ -n "$SOURCE_MAV0" ]]; then
    GT_CSV="$SOURCE_MAV0/state_groundtruth_estimate0/data.csv"
  fi
fi


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------

for path in \
  "$IMAGEDIR" \
  "$CALIB" \
  "$IMU_PRIOR" \
  "$REINTEGRATED_IMU_PRIOR" \
  "$FRAMES_CSV" \
  "$CAMERA_SENSOR_YAML" \
  "$IMU_SENSOR_YAML" \
  "$GT_CSV"
do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] required input does not exist:"
    echo "        $path"
    exit 2
  fi
done


cd "$ROOT"
mkdir -p "$OUT_DIR"


# ----------------------------------------------------------------------
# Save experiment configuration
# ----------------------------------------------------------------------

{
  echo "date=$(date -Iseconds)"
  echo "root=$ROOT"
  echo "data_root=$DATA_ROOT"
  echo "image_dir=$IMAGEDIR"
  echo "calib=$CALIB"

  echo "imu_prior_original=$IMU_PRIOR"
  echo "imu_prior_reintegrated=$REINTEGRATED_IMU_PRIOR"

  echo "frames_csv=$FRAMES_CSV"
  echo "camera_sensor_yaml=$CAMERA_SENSOR_YAML"
  echo "imu_sensor_yaml=$IMU_SENSOR_YAML"

  echo "gt_csv=$GT_CSV"

  echo "max_frames=$MAX_FRAMES"
  echo "stride=$STRIDE"
  echo "buffer=$BUFFER"

  echo "weights=$TWO_STAGE_WEIGHTS"
} > "$OUT_DIR/run_config.txt"


echo
echo "============================================================"
echo " EuRoC Reintegrated 15DoF Validation"
echo "============================================================"
echo
echo "DATA_ROOT:"
echo "  $DATA_ROOT"
echo
echo "Original IMU prior:"
echo "  $IMU_PRIOR"
echo
echo "Reintegrated IMU prior:"
echo "  $REINTEGRATED_IMU_PRIOR"
echo
echo "Output:"
echo "  $OUT_DIR"
echo
echo "MAX_FRAMES = $MAX_FRAMES"
echo "STRIDE     = $STRIDE"
echo "BUFFER     = $BUFFER"
echo
echo "============================================================"
echo


# ----------------------------------------------------------------------
# max_frames option
# ----------------------------------------------------------------------

MAX_ARGS=()

if [[ "$MAX_FRAMES" != "0" ]]; then
  MAX_ARGS=(--max_frames "$MAX_FRAMES")
fi


# evaluate_euroc_recon arguments populated after each run
RUN_ARGS=()


# ----------------------------------------------------------------------
# Common runner
# ----------------------------------------------------------------------

run_case() {

  local label="$1"
  local weights="$2"

  shift 2

  local recon="$OUT_DIR/${label}_recon.pth"
  local debug="$OUT_DIR/${label}_debug.csv"
  local log="$OUT_DIR/${label}.log"

  echo
  echo "============================================================"
  echo "[RUN] $label"
  echo "============================================================"
  echo

  "$PYTHON" demo.py \
    --imagedir "$IMAGEDIR" \
    --calib "$CALIB" \
    --weights "$weights" \
    --stride "$STRIDE" \
    "${MAX_ARGS[@]}" \
    --buffer "$BUFFER" \
    --disable_vis \
    --reconstruction_path "$recon" \
    "$@" \
    2>&1 | tee "$log"

  echo
  echo "[DONE] $label"
  echo "reconstruction: $recon"
  echo "log:            $log"

  if [[ "$RUN_PLY" == "1" ]]; then

    "$PYTHON" export_reconstruction_ply.py \
      "$recon" \
      --output "$OUT_DIR/${label}.ply"

  fi

  if [[ -f "$debug" ]]; then

    RUN_ARGS+=(
      --run "$label=$recon,$debug"
    )

  else

    RUN_ARGS+=(
      --run "$label=$recon"
    )

  fi
}


# ======================================================================
# CASE 1
#
# Original DROID
# ======================================================================

run_case \
  original_droid \
  "$ORIGINAL_WEIGHTS"


# ======================================================================
# CASE 2
#
# Visual bootstrap control
#
# Same startup/keyframe-retention behavior as the VI path,
# but without IMU factors.
# ======================================================================

run_case \
  visual_bootstrap_control \
  "$ORIGINAL_WEIGHTS" \
  --visual_bootstrap_control


# ======================================================================
# CASE 3
#
# Original two-stage full 15DoF
#
# Uses the first IMU prior.
# ======================================================================

run_case \
  two_stage_15dof \
  "$TWO_STAGE_WEIGHTS" \
  --imu_prior "$IMU_PRIOR" \
  --imu_frame_manifest "$FRAMES_CSV" \
  --cam_sensor_yaml "$CAMERA_SENSOR_YAML" \
  --imu_sensor_yaml "$IMU_SENSOR_YAML" \
  --use_two_stage_imu_ba \
  --use_imu_info_weighting \
  --imu_ba_debug \
  --imu_ba_debug_path "$OUT_DIR/two_stage_15dof_debug.csv"


# ======================================================================
# CASE 4
#
# Re-preintegrated two-stage full 15DoF
#
# IMPORTANT:
#
# This run is identical to CASE 3 except for the IMU prior.
#
# The new prior was generated around the updated ba/bg estimate, so:
#
# old bias linearization
#
#       ↓
#
# VI initialization
#
#       ↓
#
# estimated ba/bg
#
#       ↓
#
# RAW IMU re-preintegration
#
#       ↓
#
# new ΔR, Δv, Δp + bias Jacobians
#
#       ↓
#
# local/global 15DoF
#
# ======================================================================

run_case \
  two_stage_15dof_reintegrated \
  "$TWO_STAGE_WEIGHTS" \
  --imu_prior "$REINTEGRATED_IMU_PRIOR" \
  --imu_frame_manifest "$FRAMES_CSV" \
  --cam_sensor_yaml "$CAMERA_SENSOR_YAML" \
  --imu_sensor_yaml "$IMU_SENSOR_YAML" \
  --use_two_stage_imu_ba \
  --use_imu_info_weighting \
  --imu_ba_debug \
  --imu_ba_debug_path "$OUT_DIR/two_stage_15dof_reintegrated_debug.csv"


# ======================================================================
# Evaluation
# ======================================================================

EVAL_ARGS=(
  --frames-csv "$FRAMES_CSV"
  --gt-csv "$GT_CSV"
  --gt-frame body
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML"
  "${RUN_ARGS[@]}"
)


echo
echo "============================================================"
echo "[EVAL] Common retained keyframes"
echo "============================================================"
echo


# Paired evaluation:
# all methods evaluated only at frame IDs shared by every run.
"$PYTHON" tools/evaluate_euroc_recon.py \
  --common-frame-ids \
  --output-csv "$OUT_DIR/summary.csv" \
  --output-md "$OUT_DIR/summary.md" \
  "${EVAL_ARGS[@]}"


echo
echo "============================================================"
echo "[EVAL] Full retained trajectories"
echo "============================================================"
echo


# Full trajectory:
# preserves different tracking/keyframe coverage.
"$PYTHON" tools/evaluate_euroc_recon.py \
  --output-csv "$OUT_DIR/summary_full.csv" \
  --output-md "$OUT_DIR/summary_full.md" \
  "${EVAL_ARGS[@]}"


# ======================================================================
# Finish
# ======================================================================

echo
echo "============================================================"
echo "[DONE]"
echo "============================================================"
echo

echo "Output directory:"
echo "  $OUT_DIR"
echo

echo "Paired report:"
echo "  $OUT_DIR/summary.md"
echo

echo "Full report:"
echo "  $OUT_DIR/summary_full.md"
echo

echo "Original 15DoF debug:"
echo "  $OUT_DIR/two_stage_15dof_debug.csv"
echo

echo "Reintegrated 15DoF debug:"
echo "  $OUT_DIR/two_stage_15dof_reintegrated_debug.csv"
echo

echo "Run configuration:"
echo "  $OUT_DIR/run_config.txt"

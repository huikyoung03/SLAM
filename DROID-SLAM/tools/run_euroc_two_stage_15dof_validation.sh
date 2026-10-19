#!/usr/bin/env bash
set -euo pipefail

# Clean evaluation path for the two-stage visual-inertial design:
# visual pose-only frontend -> VI alignment -> local 15DoF DBA -> one global
# 15DoF DBA. An optional offline shared-bias pass performs one robust global
# 6D update and a short confirming 15DoF DBA. A visual startup-retention
# control separates the effect of keeping early keyframes from the effect of
# visual-inertial optimization.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"

DATA_ROOT="${DATA_ROOT:-/home/ubuntu/SLAM/uploads/euroc_v1_03_droid_cam0}"
IMAGEDIR="${IMAGEDIR:-${DATA_ROOT}/data}"
CALIB="${CALIB:-${DATA_ROOT}/calib.txt}"
# Full-state DBA must use a newly regenerated prior with all five bias
# Jacobians. demo.py also enforces this contract at runtime.
IMU_PRIOR="${IMU_PRIOR:-${DATA_ROOT}/imu_prior_fd_jacobian.csv}"
FRAMES_CSV="${FRAMES_CSV:-${DATA_ROOT}/frames.csv}"
CAMERA_SENSOR_YAML="${CAMERA_SENSOR_YAML:-${DATA_ROOT}/cam0_sensor.yaml}"
IMU_SENSOR_YAML="${IMU_SENSOR_YAML:-${DATA_ROOT}/imu0_sensor.yaml}"
GT_CSV="${GT_CSV:-}"

ORIGINAL_WEIGHTS="${ORIGINAL_WEIGHTS:-droid.pth}"
# By default this remains a physics/optimizer check. Set the USE_LEARNED_*
# variables below for an end-to-end checkpoint validation.
TWO_STAGE_WEIGHTS="${TWO_STAGE_WEIGHTS:-${ORIGINAL_WEIGHTS}}"

MAX_FRAMES="${MAX_FRAMES:-450}"
STRIDE="${STRIDE:-2}"
BUFFER="${BUFFER:-1024}"
RUN_PLY="${RUN_PLY:-0}"
RUN_GLOBAL_BIAS_15DOF="${RUN_GLOBAL_BIAS_15DOF:-0}"
RUN_GLOBAL_BIAS_MATCHED_CONTROLS="${RUN_GLOBAL_BIAS_MATCHED_CONTROLS:-0}"
GLOBAL_BIAS_POST_STEPS="${GLOBAL_BIAS_POST_STEPS:-2}"
GLOBAL_BIAS_LIKELIHOOD_WEIGHT="${GLOBAL_BIAS_LIKELIHOOD_WEIGHT:-1.0}"
# Keep all learned-runtime switches explicit. This prevents a trained
# confidence/weight checkpoint from being accidentally evaluated as the old
# fixed-weight physics-only configuration.
USE_IMU_NOISE_MODEL="${USE_IMU_NOISE_MODEL:-0}"
USE_LEARNED_IMU_CONFIDENCE="${USE_LEARNED_IMU_CONFIDENCE:-0}"
USE_COMPONENT_IMU_CONFIDENCE="${USE_COMPONENT_IMU_CONFIDENCE:-0}"
USE_LEARNED_IMU_BA_WEIGHT="${USE_LEARNED_IMU_BA_WEIGHT:-0}"
USE_LEARNED_IMU_RESIDUAL_WEIGHTS="${USE_LEARNED_IMU_RESIDUAL_WEIGHTS:-0}"
LEARNED_IMU_BA_WEIGHT_SCALE="${LEARNED_IMU_BA_WEIGHT_SCALE:-1.0}"
OUT_DIR="${OUT_DIR:-outputs/euroc_two_stage_15dof_$(basename "${DATA_ROOT}")_$(date +%Y%m%d_%H%M%S)}"

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
  "$CAMERA_SENSOR_YAML" "$IMU_SENSOR_YAML" "$GT_CSV"; do
  if [[ ! -e "$path" ]]; then
    echo "required input does not exist: $path" >&2
    exit 2
  fi
done

cd "$ROOT"
mkdir -p "$OUT_DIR"

MAX_ARGS=()
if [[ "$MAX_FRAMES" != "0" ]]; then
  MAX_ARGS=(--max_frames "$MAX_FRAMES")
fi

RUN_ARGS=()
TWO_STAGE_RUNTIME_ARGS=()

for option in \
  USE_IMU_NOISE_MODEL \
  USE_LEARNED_IMU_CONFIDENCE \
  USE_COMPONENT_IMU_CONFIDENCE \
  USE_LEARNED_IMU_BA_WEIGHT \
  USE_LEARNED_IMU_RESIDUAL_WEIGHTS; do
  value="${!option}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "$option must be 0 or 1, got: $value" >&2
    exit 2
  fi
done

if [[ "$USE_IMU_NOISE_MODEL" == "1" ]]; then
  TWO_STAGE_RUNTIME_ARGS+=(--use_imu_noise_model)
else
  TWO_STAGE_RUNTIME_ARGS+=(--use_imu_info_weighting)
fi
if [[ "$USE_LEARNED_IMU_CONFIDENCE" == "1" ]]; then
  TWO_STAGE_RUNTIME_ARGS+=(--use_learned_imu_confidence)
fi
if [[ "$USE_COMPONENT_IMU_CONFIDENCE" == "1" ]]; then
  if [[ "$USE_LEARNED_IMU_CONFIDENCE" != "1" ]]; then
    echo "USE_COMPONENT_IMU_CONFIDENCE=1 requires USE_LEARNED_IMU_CONFIDENCE=1" >&2
    exit 2
  fi
  TWO_STAGE_RUNTIME_ARGS+=(--use_component_imu_confidence)
fi
if [[ "$USE_LEARNED_IMU_BA_WEIGHT" == "1" ]]; then
  TWO_STAGE_RUNTIME_ARGS+=(
    --use_learned_imu_ba_weight
    --learned_imu_ba_weight_scale "$LEARNED_IMU_BA_WEIGHT_SCALE"
  )
fi
if [[ "$USE_LEARNED_IMU_RESIDUAL_WEIGHTS" == "1" ]]; then
  TWO_STAGE_RUNTIME_ARGS+=(--use_learned_imu_residual_weights)
fi

run_case() {
  local label="$1"
  local weights="$2"
  shift 2

  local recon="$OUT_DIR/${label}_recon.pth"
  local debug="$OUT_DIR/${label}_debug.csv"

  echo "[RUN] $label"
  "$PYTHON" demo.py \
    --imagedir "$IMAGEDIR" \
    --calib "$CALIB" \
    --weights "$weights" \
    --stride "$STRIDE" \
    "${MAX_ARGS[@]}" \
    --buffer "$BUFFER" \
    --disable_vis \
    --reconstruction_path "$recon" \
    "$@"

  if [[ "$RUN_PLY" == "1" ]]; then
    "$PYTHON" export_reconstruction_ply.py "$recon" \
      --output "$OUT_DIR/${label}.ply"
  fi

  if [[ -f "$debug" ]]; then
    RUN_ARGS+=(--run "$label=$recon,$debug")
  else
    RUN_ARGS+=(--run "$label=$recon")
  fi
}

# Visual-only control with exactly the same input frames and stride.
run_case original_droid "$ORIGINAL_WEIGHTS"

# The two-stage frontend intentionally preserves early keyframes until metric
# initialization becomes observable. This control keeps the same bounded
# startup window without IMU input, so a pose-count/coverage difference cannot
# be misreported as an IMU-factor improvement.
run_case visual_bootstrap_control "$ORIGINAL_WEIGHTS" \
  --visual_bootstrap_control

run_two_stage_case() {
  local label="$1"
  shift

  run_case "$label" "$TWO_STAGE_WEIGHTS" \
    --imu_prior "$IMU_PRIOR" \
    --imu_frame_manifest "$FRAMES_CSV" \
    --cam_sensor_yaml "$CAMERA_SENSOR_YAML" \
    --imu_sensor_yaml "$IMU_SENSOR_YAML" \
    --use_two_stage_imu_ba \
    "${TWO_STAGE_RUNTIME_ARGS[@]}" \
    --imu_ba_debug \
    --imu_ba_debug_path "$OUT_DIR/${label}_debug.csv" \
    "$@"
}

GLOBAL_BIAS_ARGS=(
  --use_global_bias_15dof_refinement
  --imu_global_bias_refine_post_steps "$GLOBAL_BIAS_POST_STEPS"
  --imu_global_bias_refine_weight "$GLOBAL_BIAS_LIKELIHOOD_WEIGHT"
)

if [[ "$RUN_GLOBAL_BIAS_15DOF" == "1" && "$RUN_GLOBAL_BIAS_MATCHED_CONTROLS" == "1" ]]; then
  # Isolate the shared-bias update from both keyframe coverage and the two
  # confirming backend iterations it needs after an accepted bias step.
  run_two_stage_case two_stage_15dof_pass0
  run_two_stage_case two_stage_15dof_post_control \
    --imu_two_stage_matched_post_steps "$GLOBAL_BIAS_POST_STEPS"
  run_two_stage_case two_stage_15dof_global_bias "${GLOBAL_BIAS_ARGS[@]}"
elif [[ "$RUN_GLOBAL_BIAS_15DOF" == "1" ]]; then
  run_two_stage_case two_stage_15dof_global_bias "${GLOBAL_BIAS_ARGS[@]}"
else
  run_two_stage_case two_stage_15dof
fi

EVAL_ARGS=(
  --frames-csv "$FRAMES_CSV"
  --gt-csv "$GT_CSV"
  --gt-frame body
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML"
  "${RUN_ARGS[@]}"
)

# Common-frame metrics give a paired accuracy comparison. The full report
# retains tracking coverage differences and should be read alongside it.
"$PYTHON" tools/evaluate_euroc_recon.py \
  --common-frame-ids \
  --output-csv "$OUT_DIR/summary.csv" \
  --output-md "$OUT_DIR/summary.md" \
  "${EVAL_ARGS[@]}"
"$PYTHON" tools/evaluate_euroc_recon.py \
  --output-csv "$OUT_DIR/summary_full.csv" \
  --output-md "$OUT_DIR/summary_full.md" \
  "${EVAL_ARGS[@]}"

echo "[DONE] $OUT_DIR"
echo "paired report: $OUT_DIR/summary.md"
echo "full report:   $OUT_DIR/summary_full.md"

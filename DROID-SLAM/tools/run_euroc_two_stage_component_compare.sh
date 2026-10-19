#!/usr/bin/env bash
set -euo pipefail

# One matched EuRoC comparison for the scalar 6k two-stage checkpoint and the
# quality-aware component-confidence checkpoint.  The visual controls run once
# so a full-sequence experiment does not duplicate their expensive inference.

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

ORIGINAL_WEIGHTS="${ORIGINAL_WEIGHTS:-droid.pth}"
# Reuse is deliberately opt-in because older reconstruction bundles may not
# carry input provenance. Supply this only after confirming sequence, stride,
# and frame coverage from its run log.
ORIGINAL_RECON="${ORIGINAL_RECON:-}"
VISUAL_BOOTSTRAP_RECON="${VISUAL_BOOTSTRAP_RECON:-}"
SCALAR_RECON="${SCALAR_RECON:-}"
COMPONENT_RECON="${COMPONENT_RECON:-}"
RUN_VISUAL_BOOTSTRAP_CONTROL="${RUN_VISUAL_BOOTSTRAP_CONTROL:-1}"
SCALAR_WEIGHTS="${SCALAR_WEIGHTS:-checkpoints/two_stage_15dof_conf_noise_6000_final_006000.pth}"
COMPONENT_WEIGHTS="${COMPONENT_WEIGHTS:-checkpoints/two_stage_15dof_component_confidence_quality_3000_final_003000.pth}"

# MAX_FRAMES=0 consumes the complete prepared sequence. Set 450 only for a
# short runtime smoke test; that prefix must not be used as a final result.
MAX_FRAMES="${MAX_FRAMES:-0}"
STRIDE="${STRIDE:-2}"
BUFFER="${BUFFER:-4096}"
RUN_PLY="${RUN_PLY:-0}"
OUT_DIR="${OUT_DIR:-outputs/euroc_two_stage_component_compare_$(basename "${DATA_ROOT}")_$(date +%Y%m%d_%H%M%S)}"

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
  "$CAMERA_SENSOR_YAML" "$IMU_SENSOR_YAML" "$GT_CSV" \
  "$ORIGINAL_WEIGHTS" "$SCALAR_WEIGHTS" "$COMPONENT_WEIGHTS"; do
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

reuse_case() {
  local label="$1"
  local recon="$2"

  if [[ ! -f "$recon" ]]; then
    echo "${label} reconstruction does not exist: $recon" >&2
    exit 2
  fi
  echo "[REUSE] ${label} reconstruction=$recon"
  RUN_ARGS+=(--run "${label}=${recon}")
}

IMU_COMMON_ARGS=(
  --imu_prior "$IMU_PRIOR"
  --imu_frame_manifest "$FRAMES_CSV"
  --cam_sensor_yaml "$CAMERA_SENSOR_YAML"
  --imu_sensor_yaml "$IMU_SENSOR_YAML"
  --use_two_stage_imu_ba
  --use_imu_noise_model
  --use_learned_imu_confidence
  --use_learned_imu_ba_weight
  --use_learned_imu_residual_weights
  --imu_ba_debug
)

if [[ -n "$ORIGINAL_RECON" ]]; then
  reuse_case original_droid "$ORIGINAL_RECON"
else
  run_case original_droid "$ORIGINAL_WEIGHTS"
fi

if [[ "$RUN_VISUAL_BOOTSTRAP_CONTROL" == "1" ]]; then
  if [[ -n "$VISUAL_BOOTSTRAP_RECON" ]]; then
    reuse_case visual_bootstrap_control "$VISUAL_BOOTSTRAP_RECON"
  else
    run_case visual_bootstrap_control "$ORIGINAL_WEIGHTS" --visual_bootstrap_control
  fi
elif [[ "$RUN_VISUAL_BOOTSTRAP_CONTROL" != "0" ]]; then
  echo "RUN_VISUAL_BOOTSTRAP_CONTROL must be 0 or 1, got: $RUN_VISUAL_BOOTSTRAP_CONTROL" >&2
  exit 2
fi

if [[ -n "$SCALAR_RECON" ]]; then
  reuse_case two_stage_scalar_6k "$SCALAR_RECON"
else
  run_case two_stage_scalar_6k "$SCALAR_WEIGHTS" \
    "${IMU_COMMON_ARGS[@]}" \
    --imu_ba_debug_path "$OUT_DIR/two_stage_scalar_6k_debug.csv"
fi

if [[ -n "$COMPONENT_RECON" ]]; then
  reuse_case two_stage_component_quality_3k "$COMPONENT_RECON"
else
  run_case two_stage_component_quality_3k "$COMPONENT_WEIGHTS" \
    "${IMU_COMMON_ARGS[@]}" \
    --use_component_imu_confidence \
    --imu_ba_debug_path "$OUT_DIR/two_stage_component_quality_3k_debug.csv"
fi

EVAL_ARGS=(
  --frames-csv "$FRAMES_CSV"
  --gt-csv "$GT_CSV"
  --gt-frame body
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML"
  "${RUN_ARGS[@]}"
)

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

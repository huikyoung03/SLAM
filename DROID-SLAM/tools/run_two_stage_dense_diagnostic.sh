#!/usr/bin/env bash
set -euo pipefail

# Compare the current two-stage model at three controlled points:
# 1) visual-only startup-retention control, 2) local 15DoF state before the
# final global graph, and 3) normal final global 15DoF state.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to prepared EuRoC *_droid_cam0 directory}"
WEIGHTS="${WEIGHTS:-checkpoints/two_stage_15dof_component_confidence_quality_3000_final_003000.pth}"
STRIDE="${STRIDE:-2}"
BUFFER="${BUFFER:-4096}"
MAX_FRAMES="${MAX_FRAMES:-0}"
RUN_PLY="${RUN_PLY:-0}"
OUT_DIR="${OUT_DIR:-outputs/two_stage_dense_diagnostic_$(basename "$DATA_ROOT")_$(date +%Y%m%d_%H%M%S)}"

IMAGEDIR="${DATA_ROOT}/data"
CALIB="${DATA_ROOT}/calib.txt"
IMU_PRIOR="${DATA_ROOT}/imu_prior_fd_jacobian.csv"
FRAMES_CSV="${DATA_ROOT}/frames.csv"
CAMERA_SENSOR_YAML="${DATA_ROOT}/cam0_sensor.yaml"
IMU_SENSOR_YAML="${DATA_ROOT}/imu0_sensor.yaml"
GT_CSV="${GT_CSV:-}"

if [[ -z "$GT_CSV" && -f "$DATA_ROOT/source.txt" ]]; then
  GT_CSV="$(sed -n 's/^groundtruth=//p' "$DATA_ROOT/source.txt" | head -n 1)"
fi

for path in "$IMAGEDIR" "$CALIB" "$IMU_PRIOR" "$FRAMES_CSV" \
  "$CAMERA_SENSOR_YAML" "$IMU_SENSOR_YAML" "$WEIGHTS" "$GT_CSV"; do
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

run_case() {
  local label="$1"
  shift
  echo "[RUN] $label"
  "$PYTHON" demo.py \
    --imagedir "$IMAGEDIR" \
    --calib "$CALIB" \
    --weights "$WEIGHTS" \
    --stride "$STRIDE" \
    "${MAX_ARGS[@]}" \
    --buffer "$BUFFER" \
    --disable_vis \
    --reconstruction_path "$OUT_DIR/${label}_recon.pth" \
    "$@"
}

# The same checkpoint and startup keyframe retention, but no IMU residual.
run_case visual_pose_only --visual_bootstrap_control

IMU_ARGS=(
  --imu_prior "$IMU_PRIOR"
  --imu_frame_manifest "$FRAMES_CSV"
  --cam_sensor_yaml "$CAMERA_SENSOR_YAML"
  --imu_sensor_yaml "$IMU_SENSOR_YAML"
  --use_two_stage_imu_ba
  --use_imu_noise_model
  --use_learned_imu_confidence
  --use_component_imu_confidence
  --use_learned_imu_ba_weight
  --use_learned_imu_residual_weights
  --imu_ba_debug
  --imu_ba_debug_max_rows 100000
)

run_case local_15dof \
  "${IMU_ARGS[@]}" \
  --imu_two_stage_skip_global_backend \
  --imu_ba_debug_path "$OUT_DIR/local_15dof_debug.csv"

run_case global_15dof \
  "${IMU_ARGS[@]}" \
  --imu_ba_debug_path "$OUT_DIR/global_15dof_debug.csv"

"$PYTHON" tools/evaluate_euroc_recon.py \
  --common-frame-ids \
  --frames-csv "$FRAMES_CSV" \
  --gt-csv "$GT_CSV" \
  --gt-frame body \
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML" \
  --run "visual_pose_only=$OUT_DIR/visual_pose_only_recon.pth" \
  --run "local_15dof=$OUT_DIR/local_15dof_recon.pth,$OUT_DIR/local_15dof_debug.csv" \
  --run "global_15dof=$OUT_DIR/global_15dof_recon.pth,$OUT_DIR/global_15dof_debug.csv" \
  --output-csv "$OUT_DIR/summary_common.csv" \
  --output-md "$OUT_DIR/summary_common.md"

"$PYTHON" tools/evaluate_euroc_recon.py \
  --frames-csv "$FRAMES_CSV" \
  --gt-csv "$GT_CSV" \
  --gt-frame body \
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML" \
  --run "visual_pose_only=$OUT_DIR/visual_pose_only_recon.pth" \
  --run "local_15dof=$OUT_DIR/local_15dof_recon.pth,$OUT_DIR/local_15dof_debug.csv" \
  --run "global_15dof=$OUT_DIR/global_15dof_recon.pth,$OUT_DIR/global_15dof_debug.csv" \
  --output-csv "$OUT_DIR/summary_full.csv" \
  --output-md "$OUT_DIR/summary_full.md"

"$PYTHON" tools/analyze_reconstruction_health.py \
  --recon "visual_pose_only=$OUT_DIR/visual_pose_only_recon.pth" \
  --recon "local_15dof=$OUT_DIR/local_15dof_recon.pth" \
  --recon "global_15dof=$OUT_DIR/global_15dof_recon.pth" \
  --imu-debug "local_15dof=$OUT_DIR/local_15dof_debug.csv" \
  --imu-debug "global_15dof=$OUT_DIR/global_15dof_debug.csv" \
  --output-dir "$OUT_DIR/health"

if [[ "$RUN_PLY" == "1" ]]; then
  for label in visual_pose_only local_15dof global_15dof; do
    "$PYTHON" export_reconstruction_ply.py \
      "$OUT_DIR/${label}_recon.pth" \
      --output "$OUT_DIR/${label}.ply"
  done
fi

echo "[DONE] $OUT_DIR"
echo "common trajectory metrics: $OUT_DIR/summary_common.md"
echo "full trajectory metrics:   $OUT_DIR/summary_full.md"
echo "dense/IMU health:          $OUT_DIR/health/health_summary.md"

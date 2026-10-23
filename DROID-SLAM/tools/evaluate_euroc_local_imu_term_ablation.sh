#!/usr/bin/env bash
set -euo pipefail

# Local-only diagnostic for the two translational terms of the full 15DoF
# inertial factor.  The visual-DROID and full fixed-c=1 reconstructions from
# the preceding matched local three-way experiment are reused verbatim.
#
#   no_position: r_v + r_R + r_bias (position-row information is exactly 0)
#   no_velocity: r_p + r_R + r_bias (velocity-row information is exactly 0)
#
# This is a factor ablation, not a tuned weight sweep.  Global BA and dense
# trajectory filling remain disabled in every newly executed arm.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to prepared V1_03 *_droid_cam0}"
WEIGHTS="${WEIGHTS:?Set WEIGHTS to the strict 15DoF checkpoint}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR to a new output directory}"
REFERENCE_DIR="${REFERENCE_DIR:-outputs/v1_03_strict_local_three_way_2k}"
IMAGEDIR="${IMAGEDIR:-${DATA_ROOT}/data}"
CALIB="${CALIB:-${DATA_ROOT}/calib.txt}"
IMU_PRIOR="${IMU_PRIOR:?Set IMU_PRIOR to the V1_03 central-difference prior}"
FRAMES_CSV="${FRAMES_CSV:-${DATA_ROOT}/frames.csv}"
CAMERA_SENSOR_YAML="${CAMERA_SENSOR_YAML:?Set CAMERA_SENSOR_YAML}"
IMU_SENSOR_YAML="${IMU_SENSOR_YAML:-${DATA_ROOT}/imu0_sensor.yaml}"
GT_CSV="${GT_CSV:-}"
STRIDE="${STRIDE:-2}"
TRAJECTORY_FILL_STRIDE="${TRAJECTORY_FILL_STRIDE:-1}"
BUFFER="${BUFFER:-512}"
EUROC_ORIGINAL_PROTOCOL="${EUROC_ORIGINAL_PROTOCOL:-1}"

export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

if [[ -z "$GT_CSV" && -f "$DATA_ROOT/source.txt" ]]; then
  GT_CSV="$(sed -n 's/^groundtruth=//p' "$DATA_ROOT/source.txt" | head -n 1)"
fi

VISUAL_RECON="$REFERENCE_DIR/visual_droid_local_recon.pth"
FIXED_RECON="$REFERENCE_DIR/fixed_c1_15dof_local_recon.pth"
FIXED_DEBUG="$REFERENCE_DIR/fixed_c1_15dof_local_debug.csv"

for path in "$IMAGEDIR" "$CALIB" "$WEIGHTS" "$IMU_PRIOR" "$FRAMES_CSV" \
  "$CAMERA_SENSOR_YAML" "$IMU_SENSOR_YAML" "$GT_CSV" \
  "$VISUAL_RECON" "$FIXED_RECON" "$FIXED_DEBUG"; do
  if [[ ! -e "$path" ]]; then
    echo "required input does not exist: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUT_DIR" ]]; then
  echo "refusing to overwrite existing output directory: $OUT_DIR" >&2
  exit 2
fi
if [[ "$EUROC_ORIGINAL_PROTOCOL" != "1" || "$STRIDE" != "2" || \
  "$TRAJECTORY_FILL_STRIDE" != "1" || "$BUFFER" != "512" ]]; then
  echo "this diagnostic requires the strict original DROID protocol: EUROC_ORIGINAL_PROTOCOL=1, STRIDE=2, TRAJECTORY_FILL_STRIDE=1, BUFFER=512" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
{
  echo "purpose=local_only_15dof_position_velocity_factor_ablation"
  echo "global_backend=disabled"
  echo "trajectory_filler=disabled"
  echo "reference_visual=$(realpath "$VISUAL_RECON")"
  echo "reference_fixed_c1=$(realpath "$FIXED_RECON")"
  echo "weights=$(realpath "$WEIGHTS")"
  echo "imu_prior=$(realpath "$IMU_PRIOR")"
  echo "position_weight_default=0.05"
  echo "velocity_weight_default=0.05"
  echo "no_position_weight=0.0"
  echo "no_velocity_weight=0.0"
  git rev-parse HEAD
} > "$OUT_DIR/run_manifest.txt"

common_args=(
  --imagedir "$IMAGEDIR"
  --calib "$CALIB"
  --weights "$WEIGHTS"
  --stride "$STRIDE"
  --trajectory_fill_stride "$TRAJECTORY_FILL_STRIDE"
  --buffer "$BUFFER"
  --euroc_original_protocol
  --disable_vis
  --local_tracking_only
  --imu_prior "$IMU_PRIOR"
  --imu_frame_manifest "$FRAMES_CSV"
  --cam_sensor_yaml "$CAMERA_SENSOR_YAML"
  --imu_sensor_yaml "$IMU_SENSOR_YAML"
  --use_two_stage_imu_ba
  --no_vi_init_bootstrap
  --use_imu_noise_model
  --imu_ba_debug
  --imu_ba_prior_weight 1.0
  --imu_tuned_confidence_schedule 1.0
)

run_case() {
  local label="$1"
  local disabled_weight_arg="$2"

  echo "[RUN] $label"
  "$PYTHON" demo.py \
    "${common_args[@]}" \
    "$disabled_weight_arg" 0 \
    --reconstruction_path "$OUT_DIR/${label}_recon.pth" \
    --imu_ba_debug_path "$OUT_DIR/${label}_debug.csv"
}

run_case no_position_15dof_local --imu_full_pos_weight
run_case no_velocity_15dof_local --imu_full_vel_weight

eval_args=(
  --trajectory-source keyframes
  --common-frame-ids
  --frames-csv "$FRAMES_CSV"
  --gt-csv "$GT_CSV"
  --gt-frame body
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML"
  --run "visual_droid_local=$VISUAL_RECON"
  --run "fixed_c1_15dof_local=$FIXED_RECON,$FIXED_DEBUG"
  --run "no_position_15dof_local=$OUT_DIR/no_position_15dof_local_recon.pth,$OUT_DIR/no_position_15dof_local_debug.csv"
  --run "no_velocity_15dof_local=$OUT_DIR/no_velocity_15dof_local_recon.pth,$OUT_DIR/no_velocity_15dof_local_debug.csv"
)

"$PYTHON" tools/evaluate_euroc_recon.py \
  "${eval_args[@]}" \
  --output-csv "$OUT_DIR/summary_common.csv" \
  --output-md "$OUT_DIR/summary_common.md"

"$PYTHON" tools/evaluate_euroc_recon.py \
  --trajectory-source keyframes \
  --frames-csv "$FRAMES_CSV" \
  --gt-csv "$GT_CSV" \
  --gt-frame body \
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML" \
  --run "visual_droid_local=$VISUAL_RECON" \
  --run "fixed_c1_15dof_local=$FIXED_RECON,$FIXED_DEBUG" \
  --run "no_position_15dof_local=$OUT_DIR/no_position_15dof_local_recon.pth,$OUT_DIR/no_position_15dof_local_debug.csv" \
  --run "no_velocity_15dof_local=$OUT_DIR/no_velocity_15dof_local_recon.pth,$OUT_DIR/no_velocity_15dof_local_debug.csv" \
  --output-csv "$OUT_DIR/summary_full.csv" \
  --output-md "$OUT_DIR/summary_full.md"

echo "[DONE] $OUT_DIR"
echo "[DONE] comparison=$OUT_DIR/summary_common.md"

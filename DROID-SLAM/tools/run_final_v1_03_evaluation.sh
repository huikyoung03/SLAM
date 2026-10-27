#!/usr/bin/env bash
set -euo pipefail

# Reproduce the final paper-facing comparison only:
#   original visual DROID vs. final direct R/V/P learned IMU attenuation.
# No fixed-confidence, lambda, discrepancy, innovation, or edge-fusion arm is
# available from this entry point.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to prepared V1_03_difficult_droid_cam0}"
IMU_PRIOR="${IMU_PRIOR:?Set IMU_PRIOR to rectified imu_prior_*_jacobian.csv}"
GT_CSV="${GT_CSV:?Set GT_CSV to the V1_03 ground-truth CSV}"
ORIGINAL_WEIGHTS="${ORIGINAL_WEIGHTS:?Set ORIGINAL_WEIGHTS to droid.pth}"
LEARNED_WEIGHTS="${LEARNED_WEIGHTS:?Set LEARNED_WEIGHTS to the final learned checkpoint}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR to a new output directory}"

IMAGEDIR="${IMAGEDIR:-${DATA_ROOT}/data}"
FRAMES_CSV="${FRAMES_CSV:-${DATA_ROOT}/frames.csv}"
IMU_SENSOR_YAML="${IMU_SENSOR_YAML:-${DATA_ROOT}/imu0_sensor.yaml}"
STRICT_ASSET_DIR="${STRICT_ASSET_DIR:-$(dirname "${IMU_PRIOR}")}" 
CALIB="${CALIB:-${STRICT_ASSET_DIR}/calib_rectified.txt}"
CAMERA_SENSOR_YAML="${CAMERA_SENSOR_YAML:-${STRICT_ASSET_DIR}/cam0_sensor_rectified.yaml}"

cd "$ROOT"
export LD_LIBRARY_PATH="${CONDA_PREFIX:+${CONDA_PREFIX}/lib:}${LD_LIBRARY_PATH:-}"

for path in "$IMAGEDIR" "$FRAMES_CSV" "$IMU_PRIOR" "$IMU_SENSOR_YAML" \
            "$CALIB" "$CAMERA_SENSOR_YAML" "$GT_CSV" "$ORIGINAL_WEIGHTS" \
            "$LEARNED_WEIGHTS"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done
[[ ! -e "$OUT_DIR" ]] || { echo "Refusing to overwrite OUT_DIR: $OUT_DIR" >&2; exit 2; }
mkdir -p "$OUT_DIR"

cat > "$OUT_DIR/run_manifest.txt" <<EOF
protocol=final_direct_rvp_v1_03
tracking_stride=2
trajectory_fill_stride=1
buffer=512
confidence_mode=direct_rvp_precision
global_imu_lambda=none
direct_extensions=none
local_schedule=visual:3,full15dof:2
global_schedule=visual:7,full15dof:12
EOF

echo '[1/3] Original visual DROID'
"$PYTHON" demo.py \
  --imagedir "$IMAGEDIR" \
  --calib "$CALIB" \
  --weights "$ORIGINAL_WEIGHTS" \
  --stride 2 \
  --trajectory_fill_stride 1 \
  --euroc_original_protocol \
  --buffer 512 \
  --disable_vis \
  --reconstruction_path "$OUT_DIR/original_droid_recon.pth"

echo '[2/3] Final direct R/V/P learned 15DoF model'
"$PYTHON" demo.py \
  --imagedir "$IMAGEDIR" \
  --calib "$CALIB" \
  --weights "$LEARNED_WEIGHTS" \
  --stride 2 \
  --trajectory_fill_stride 1 \
  --euroc_original_protocol \
  --buffer 512 \
  --disable_vis \
  --reconstruction_path "$OUT_DIR/learned_direct_rvp_recon.pth" \
  --imu_prior "$IMU_PRIOR" \
  --imu_frame_manifest "$FRAMES_CSV" \
  --cam_sensor_yaml "$CAMERA_SENSOR_YAML" \
  --imu_sensor_yaml "$IMU_SENSOR_YAML" \
  --use_two_stage_imu_ba \
  --imu_two_stage_visual_steps 3 \
  --imu_two_stage_full_steps 2 \
  --imu_two_stage_global_visual_pre_steps 7 \
  --imu_two_stage_global_steps 12 \
  --no_vi_init_bootstrap \
  --use_imu_noise_model \
  --use_imu_bias_random_walk \
  --use_imu_full_covariance_weighting \
  --use_learned_imu_precision \
  --imu_ba_debug \
  --imu_ba_debug_path "$OUT_DIR/learned_direct_rvp_debug.csv"

echo '[3/3] Dense whole-trajectory Sim(3) evaluation'
"$PYTHON" tools/evaluate_euroc_recon.py \
  --trajectory-source dense \
  --common-frame-ids \
  --frames-csv "$FRAMES_CSV" \
  --gt-csv "$GT_CSV" \
  --gt-frame body \
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML" \
  --run "original_droid=$OUT_DIR/original_droid_recon.pth" \
  --run "learned_direct_rvp=$OUT_DIR/learned_direct_rvp_recon.pth,$OUT_DIR/learned_direct_rvp_debug.csv" \
  --output-csv "$OUT_DIR/summary_common.csv" \
  --output-md "$OUT_DIR/summary_common.md"

"$PYTHON" tools/evaluate_euroc_recon.py \
  --trajectory-source dense \
  --frames-csv "$FRAMES_CSV" \
  --gt-csv "$GT_CSV" \
  --gt-frame body \
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML" \
  --run "original_droid=$OUT_DIR/original_droid_recon.pth" \
  --run "learned_direct_rvp=$OUT_DIR/learned_direct_rvp_recon.pth,$OUT_DIR/learned_direct_rvp_debug.csv" \
  --output-csv "$OUT_DIR/summary_full.csv" \
  --output-md "$OUT_DIR/summary_full.md"

echo "[DONE] $OUT_DIR/summary_full.md"

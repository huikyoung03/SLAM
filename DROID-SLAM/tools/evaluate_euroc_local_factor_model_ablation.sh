#!/usr/bin/env bash
set -euo pipefail

# Strict, dense local-VIO diagnostic for the *physical IMU factor*, before a
# new confidence training run.  All newly executed arms use c=0.25, the best
# fixed-control value observed under the legacy factor.  Thus a difference is
# attributable to p/v/R covariance normalization and/or bias dynamics, never
# to learned confidence or terminal global BA.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to prepared V1_03 *_droid_cam0}"
WEIGHTS="${WEIGHTS:?Set WEIGHTS to the strict 15DoF checkpoint}"
IMU_PRIOR="${IMU_PRIOR:?Set IMU_PRIOR to the strict rectified IMU prior}"
CAMERA_SENSOR_YAML="${CAMERA_SENSOR_YAML:?Set CAMERA_SENSOR_YAML to cam0_sensor_rectified.yaml}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR to a new output directory}"
REFERENCE_DIR="${REFERENCE_DIR:-outputs/v1_03_strict_local_dense_fixed_confidence_curve}"
VISUAL_DIR="${VISUAL_DIR:-outputs/v1_03_dvi_local_dense_threeway}"
IMAGEDIR="${IMAGEDIR:-${DATA_ROOT}/data}"
CALIB="${CALIB:-${DATA_ROOT}/calib.txt}"
FRAMES_CSV="${FRAMES_CSV:-${DATA_ROOT}/frames.csv}"
IMU_SENSOR_YAML="${IMU_SENSOR_YAML:-${DATA_ROOT}/imu0_sensor.yaml}"
GT_CSV="${GT_CSV:-}"

export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

if [[ -z "$GT_CSV" && -f "$DATA_ROOT/source.txt" ]]; then
  GT_CSV="$(sed -n 's/^groundtruth=//p' "$DATA_ROOT/source.txt" | head -n 1)"
fi

VISUAL_RECON="$VISUAL_DIR/visual_droid_local_recon.pth"
LEGACY_RECON="$REFERENCE_DIR/fixed_c025_recon.pth"
LEGACY_DEBUG="$REFERENCE_DIR/fixed_c025_debug.csv"
for path in "$IMAGEDIR" "$CALIB" "$WEIGHTS" "$IMU_PRIOR" "$FRAMES_CSV" \
  "$CAMERA_SENSOR_YAML" "$IMU_SENSOR_YAML" "$GT_CSV" \
  "$VISUAL_RECON" "$LEGACY_RECON" "$LEGACY_DEBUG"; do
  if [[ ! -e "$path" ]]; then
    echo "required input does not exist: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUT_DIR" ]]; then
  echo "refusing to overwrite existing output directory: $OUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
{
  echo "purpose=strict_dense_local_15dof_factor_model_ablation"
  echo "confidence=fixed_0.25"
  echo "global_backend=disabled"
  echo "trajectory_filler=enabled"
  echo "legacy_reference=$LEGACY_RECON"
  echo "camera_sensor_yaml=$(realpath "$CAMERA_SENSOR_YAML")"
  echo "imu_prior=$(realpath "$IMU_PRIOR")"
  git rev-parse HEAD
} > "$OUT_DIR/run_manifest.txt"

common_args=(
  --imagedir "$IMAGEDIR"
  --calib "$CALIB"
  --weights "$WEIGHTS"
  --stride 2
  --trajectory_fill_stride 1
  --euroc_original_protocol
  --buffer 512
  --disable_vis
  --local_vio_only
  --imu_prior "$IMU_PRIOR"
  --imu_frame_manifest "$FRAMES_CSV"
  --cam_sensor_yaml "$CAMERA_SENSOR_YAML"
  --imu_sensor_yaml "$IMU_SENSOR_YAML"
  --use_two_stage_imu_ba
  --no_vi_init_bootstrap
  --use_imu_noise_model
  --imu_ba_debug
  --imu_ba_prior_weight 1.0
  --imu_tuned_confidence_schedule 0.25
)

run_case() {
  local label="$1"
  shift
  echo "[RUN] $label"
  "$PYTHON" demo.py \
    "${common_args[@]}" \
    "$@" \
    --reconstruction_path "$OUT_DIR/${label}_recon.pth" \
    --imu_ba_debug_path "$OUT_DIR/${label}_debug.csv"
}

# Isolate each physical correction, then evaluate their combination.  No
# learned confidence flag is present in any arm.
run_case legacy_bias_rw --use_imu_bias_random_walk
run_case covariance_pvr --use_imu_full_covariance_weighting
run_case covariance_pvr_bias_rw \
  --use_imu_full_covariance_weighting \
  --use_imu_bias_random_walk

eval_args=(
  --trajectory-source dense
  --common-frame-ids
  --frames-csv "$FRAMES_CSV"
  --gt-csv "$GT_CSV"
  --gt-frame body
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML"
  --run "visual=$VISUAL_RECON"
  --run "legacy_c025=$LEGACY_RECON,$LEGACY_DEBUG"
  --run "legacy_bias_rw=$OUT_DIR/legacy_bias_rw_recon.pth,$OUT_DIR/legacy_bias_rw_debug.csv"
  --run "covariance_pvr=$OUT_DIR/covariance_pvr_recon.pth,$OUT_DIR/covariance_pvr_debug.csv"
  --run "covariance_pvr_bias_rw=$OUT_DIR/covariance_pvr_bias_rw_recon.pth,$OUT_DIR/covariance_pvr_bias_rw_debug.csv"
)

"$PYTHON" tools/evaluate_euroc_recon.py \
  "${eval_args[@]}" \
  --output-csv "$OUT_DIR/summary_common.csv" \
  --output-md "$OUT_DIR/summary_common.md"

echo "[DONE] $OUT_DIR"
echo "[DONE] comparison=$OUT_DIR/summary_common.md"

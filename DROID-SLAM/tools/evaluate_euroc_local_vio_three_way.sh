#!/usr/bin/env bash
set -euo pipefail

# DVI-style local VO/VIO ablation. This intentionally does *not* call
# Droid.terminate(): every arm retains the online local-BA state and skips all
# termination-time global BA. It does run PoseTrajectoryFiller on that frozen
# local state, so ATE-scale is computed on the raw camera time axis rather
# than on a potentially different set of selected keyframes.
#
# It answers two causal questions before another training run:
#   visual -> fixed c=1 : does this calibrated full IMU factor help at all?
#   fixed c=1 -> learned : does the learned confidence help beyond full IMU?
#
# This is the DVI-style local VIO metric, not the final global-SLAM metric.
# Use evaluate_original_vs_15dof_confidence.sh separately for global BA.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to prepared V1_03 *_droid_cam0}"
WEIGHTS="${WEIGHTS:?Set WEIGHTS to the learned 15DoF checkpoint}"
ORIGINAL_WEIGHTS="${ORIGINAL_WEIGHTS:-droid.pth}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR to a new directory}"
IMAGEDIR="${IMAGEDIR:-${DATA_ROOT}/data}"
CALIB="${CALIB:-${DATA_ROOT}/calib.txt}"
IMU_PRIOR="${IMU_PRIOR:-${DATA_ROOT}/imu_prior_fd_jacobian.csv}"
FRAMES_CSV="${FRAMES_CSV:-${DATA_ROOT}/frames.csv}"
# A rectified IMU prior is expressed in the rectified camera coordinate
# system.  Its camera-to-body transform must also be used when converting
# EuRoC body GT for evaluation.  Prefer the sidecar generated with that
# prior; callers can still explicitly override CAMERA_SENSOR_YAML.
STRICT_PRIOR_CAMERA_YAML="$(dirname "$IMU_PRIOR")/cam0_sensor_rectified.yaml"
if [[ -n "${CAMERA_SENSOR_YAML:-}" ]]; then
  CAMERA_SENSOR_YAML="$CAMERA_SENSOR_YAML"
elif [[ "${EUROC_ORIGINAL_PROTOCOL:-1}" == "1" && -f "$STRICT_PRIOR_CAMERA_YAML" ]]; then
  CAMERA_SENSOR_YAML="$STRICT_PRIOR_CAMERA_YAML"
else
  CAMERA_SENSOR_YAML="${DATA_ROOT}/cam0_sensor.yaml"
fi
IMU_SENSOR_YAML="${IMU_SENSOR_YAML:-${DATA_ROOT}/imu0_sensor.yaml}"
GT_CSV="${GT_CSV:-}"
STRIDE="${STRIDE:-2}"
TRAJECTORY_FILL_STRIDE="${TRAJECTORY_FILL_STRIDE:-1}"
BUFFER="${BUFFER:-512}"
MAX_FRAMES="${MAX_FRAMES:-0}"
EUROC_ORIGINAL_PROTOCOL="${EUROC_ORIGINAL_PROTOCOL:-1}"
# When a checkpoint has learned the p/v-vs-rotation relative residual
# calibration, runtime must explicitly opt in to load it.  Both fixed-c=1 and
# learned-confidence arms then share that calibration; their only difference
# remains interval confidence.
USE_LEARNED_IMU_RESIDUAL_WEIGHTS="${USE_LEARNED_IMU_RESIDUAL_WEIGHTS:-0}"
# Component mode produces independent rotation/velocity/position confidence.
# It is mutually exclusive with the older scalar DVI head.
USE_COMPONENT_IMU_CONFIDENCE="${USE_COMPONENT_IMU_CONFIDENCE:-0}"
# Match the calibrated 15DoF training objective: local b_a/b_g transitions
# use sensor random-walk covariance, rather than the legacy fixed 0.001
# coefficient. Both fixed-c=1 and learned arms receive the same factor.
USE_IMU_BIAS_RANDOM_WALK="${USE_IMU_BIAS_RANDOM_WALK:-1}"

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
  "$CAMERA_SENSOR_YAML" "$IMU_SENSOR_YAML" "$GT_CSV" "$WEIGHTS" "$ORIGINAL_WEIGHTS"; do
  if [[ ! -e "$path" ]]; then
    echo "required input does not exist: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUT_DIR" ]]; then
  echo "refusing to overwrite existing output directory: $OUT_DIR" >&2
  exit 2
fi
if [[ "$EUROC_ORIGINAL_PROTOCOL" != "0" && "$EUROC_ORIGINAL_PROTOCOL" != "1" ]]; then
  echo "EUROC_ORIGINAL_PROTOCOL must be 0 or 1" >&2
  exit 2
fi
if [[ "$EUROC_ORIGINAL_PROTOCOL" == "1" ]]; then
  if [[ "$STRIDE" != "2" || "$TRAJECTORY_FILL_STRIDE" != "1" || "$BUFFER" != "512" ]]; then
    echo "strict original protocol requires STRIDE=2, TRAJECTORY_FILL_STRIDE=1, BUFFER=512" >&2
    exit 2
  fi
fi
if [[ "$USE_LEARNED_IMU_RESIDUAL_WEIGHTS" != "0" && "$USE_LEARNED_IMU_RESIDUAL_WEIGHTS" != "1" ]]; then
  echo "USE_LEARNED_IMU_RESIDUAL_WEIGHTS must be 0 or 1" >&2
  exit 2
fi
if [[ "$USE_COMPONENT_IMU_CONFIDENCE" != "0" && "$USE_COMPONENT_IMU_CONFIDENCE" != "1" ]]; then
  echo "USE_COMPONENT_IMU_CONFIDENCE must be 0 or 1" >&2
  exit 2
fi
if [[ "$USE_IMU_BIAS_RANDOM_WALK" != "0" && "$USE_IMU_BIAS_RANDOM_WALK" != "1" ]]; then
  echo "USE_IMU_BIAS_RANDOM_WALK must be 0 or 1" >&2
  exit 2
fi

profile_args=()
if [[ "$EUROC_ORIGINAL_PROTOCOL" == "1" ]]; then
  profile_args=(--euroc_original_protocol)
fi
max_args=()
if [[ "$MAX_FRAMES" != "0" ]]; then
  max_args=(--max_frames "$MAX_FRAMES")
fi

mkdir -p "$OUT_DIR"
{
  echo "purpose=dvi_style_local_vio_three_way_visual_fixed_c1_learned_confidence"
  echo "global_backend=disabled"
  echo "trajectory_filler=enabled_from_online_local_state"
  echo "evaluation_trajectory_source=dense"
  echo "data_root=$(realpath "$DATA_ROOT")"
  echo "weights=$(realpath "$WEIGHTS")"
  echo "original_weights=$(realpath "$ORIGINAL_WEIGHTS")"
  echo "imu_prior=$(realpath "$IMU_PRIOR")"
  echo "tracking_stride=$STRIDE"
  echo "buffer=$BUFFER"
  echo "euroc_original_protocol=$EUROC_ORIGINAL_PROTOCOL"
  echo "use_learned_imu_residual_weights=$USE_LEARNED_IMU_RESIDUAL_WEIGHTS"
  echo "use_component_imu_confidence=$USE_COMPONENT_IMU_CONFIDENCE"
  echo "use_imu_bias_random_walk=$USE_IMU_BIAS_RANDOM_WALK"
  git rev-parse HEAD
  git status --short
} > "$OUT_DIR/run_manifest.txt"

local_args=(
  --imagedir "$IMAGEDIR"
  --calib "$CALIB"
  --stride "$STRIDE"
  --trajectory_fill_stride "$TRAJECTORY_FILL_STRIDE"
  "${profile_args[@]}"
  "${max_args[@]}"
  --buffer "$BUFFER"
  --disable_vis
  --local_vio_only
)

echo "[RUN] visual_droid_local"
"$PYTHON" demo.py \
  "${local_args[@]}" \
  --weights "$ORIGINAL_WEIGHTS" \
  --reconstruction_path "$OUT_DIR/visual_droid_local_recon.pth"

imu_args=(
  --imu_prior "$IMU_PRIOR"
  --imu_frame_manifest "$FRAMES_CSV"
  --cam_sensor_yaml "$CAMERA_SENSOR_YAML"
  --imu_sensor_yaml "$IMU_SENSOR_YAML"
  --use_two_stage_imu_ba
  --no_vi_init_bootstrap
  --use_imu_noise_model
  --imu_ba_debug
)

if [[ "$USE_LEARNED_IMU_RESIDUAL_WEIGHTS" == "1" ]]; then
  imu_args+=(--use_learned_imu_residual_weights)
fi
if [[ "$USE_IMU_BIAS_RANDOM_WALK" == "1" ]]; then
  imu_args+=(--use_imu_bias_random_walk)
fi

echo "[RUN] fixed_c1_15dof_local"
"$PYTHON" demo.py \
  "${local_args[@]}" \
  --weights "$WEIGHTS" \
  --reconstruction_path "$OUT_DIR/fixed_c1_15dof_local_recon.pth" \
  "${imu_args[@]}" \
  --imu_ba_debug_path "$OUT_DIR/fixed_c1_15dof_local_debug.csv" \
  --imu_ba_prior_weight 1.0 \
  --imu_tuned_confidence_schedule 1.0

echo "[RUN] learned_15dof_confidence_local"
learned_confidence_args=(--use_dvi_imu_confidence --use_dvi_imu_quality_fusion)
if [[ "$USE_COMPONENT_IMU_CONFIDENCE" == "1" ]]; then
  learned_confidence_args=(--use_component_imu_confidence)
fi
"$PYTHON" demo.py \
  "${local_args[@]}" \
  --weights "$WEIGHTS" \
  --reconstruction_path "$OUT_DIR/learned_15dof_confidence_local_recon.pth" \
  "${imu_args[@]}" \
  --imu_ba_debug_path "$OUT_DIR/learned_15dof_confidence_local_debug.csv" \
  --imu_confidence_only_weighting \
  --use_learned_imu_confidence \
  "${learned_confidence_args[@]}" \
  --imu_confidence_floor 0

eval_args=(
  --trajectory-source dense
  --common-frame-ids
  --frames-csv "$FRAMES_CSV"
  --gt-csv "$GT_CSV"
  --gt-frame body
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML"
  --run "visual_droid_local=$OUT_DIR/visual_droid_local_recon.pth"
  --run "fixed_c1_15dof_local=$OUT_DIR/fixed_c1_15dof_local_recon.pth,$OUT_DIR/fixed_c1_15dof_local_debug.csv"
  --run "learned_15dof_confidence_local=$OUT_DIR/learned_15dof_confidence_local_recon.pth,$OUT_DIR/learned_15dof_confidence_local_debug.csv"
)

"$PYTHON" tools/evaluate_euroc_recon.py \
  "${eval_args[@]}" \
  --output-csv "$OUT_DIR/summary_common.csv" \
  --output-md "$OUT_DIR/summary_common.md"
"$PYTHON" tools/evaluate_euroc_recon.py \
  "${eval_args[@]}" \
  --output-csv "$OUT_DIR/summary_full.csv" \
  --output-md "$OUT_DIR/summary_full.md"

echo "[DONE] $OUT_DIR"
echo "[DONE] comparison=$OUT_DIR/summary_common.md"

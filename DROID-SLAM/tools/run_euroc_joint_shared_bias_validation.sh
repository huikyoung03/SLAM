#!/usr/bin/env bash
set -euo pipefail

# Compare visual-only DROID, ordinary fixed-bias 9DoF VI-DROID, a matched
# fixed-bias outer-relinearization control, and one Schur-eliminated shared
# bias block per sliding window.  The control uses the same update interval,
# local-state window, and number of CUDA linearizations as the joint solve.
# The dataset must be held out from checkpoint selection and fine-tuning.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to a prepared EuRoC *_droid_cam0 directory}"
ORIGINAL_WEIGHTS="${ORIGINAL_WEIGHTS:-droid.pth}"
# Optional visual-only reconstruction from an identical dataset/stride/buffer
# run.  Supplying it avoids spending another full sequence on a baseline that
# does not depend on the IMU solver changes under test.
ORIGINAL_RECON="${ORIGINAL_RECON:-}"
# Optional ordinary fixed-bias 9DoF reconstruction from the same frozen
# checkpoint and runtime condition.  It is separate from the new matched
# outer-relinearization control, which must always be run after solver edits.
FIXED_RECON="${FIXED_RECON:-}"
IMU_WEIGHTS="${IMU_WEIGHTS:?Set IMU_WEIGHTS to a frozen 9DoF checkpoint}"
GT_CSV="${GT_CSV:-}"
MAX_FRAMES="${MAX_FRAMES:-0}"
STRIDE="${STRIDE:-2}"
BUFFER="${BUFFER:-4096}"
RUN_PLY="${RUN_PLY:-0}"
RUN_IMU_DEBUG="${RUN_IMU_DEBUG:-0}"
OUT_DIR="${OUT_DIR:-outputs/euroc_joint_shared_bias_$(basename "${DATA_ROOT}")_$(date +%Y%m%d_%H%M%S)}"

IMU_JOINT_BIAS_WINDOW="${IMU_JOINT_BIAS_WINDOW:-20}"
IMU_JOINT_BIAS_INTERVAL="${IMU_JOINT_BIAS_INTERVAL:-10}"
IMU_JOINT_BIAS_DAMPING="${IMU_JOINT_BIAS_DAMPING:-1e-4}"
IMU_JOINT_BIAS_MIN_EDGES="${IMU_JOINT_BIAS_MIN_EDGES:-8}"
IMU_JOINT_BIAS_MIN_DURATION="${IMU_JOINT_BIAS_MIN_DURATION:-0.5}"
IMU_JOINT_BIAS_MIN_ROTATION_DEG="${IMU_JOINT_BIAS_MIN_ROTATION_DEG:-2.0}"
IMU_BA_OUTER_RELINEARIZE_INTERVAL="${IMU_BA_OUTER_RELINEARIZE_INTERVAL:-$IMU_JOINT_BIAS_INTERVAL}"
IMU_BA_OUTER_WINDOW="${IMU_BA_OUTER_WINDOW:-$IMU_JOINT_BIAS_WINDOW}"

IMAGEDIR="${IMAGEDIR:-${DATA_ROOT}/data}"
CALIB="${CALIB:-${DATA_ROOT}/calib.txt}"
IMU_PRIOR="${IMU_PRIOR:-${DATA_ROOT}/imu_prior.csv}"
FRAMES_CSV="${FRAMES_CSV:-${DATA_ROOT}/frames.csv}"
CAMERA_SENSOR_YAML="${CAMERA_SENSOR_YAML:-${DATA_ROOT}/cam0_sensor.yaml}"
IMU_SENSOR_YAML="${IMU_SENSOR_YAML:-${DATA_ROOT}/imu0_sensor.yaml}"

if [[ -z "$GT_CSV" && -f "${DATA_ROOT}/source.txt" ]]; then
  GT_CSV="$(sed -n 's/^groundtruth=//p' "${DATA_ROOT}/source.txt" | head -n 1)"
fi
if [[ -z "$GT_CSV" && -f "${DATA_ROOT}/source.txt" ]]; then
  SOURCE_MAV0="$(sed -n 's/^source_mav0=//p' "${DATA_ROOT}/source.txt" | head -n 1)"
  if [[ -n "$SOURCE_MAV0" ]]; then
    GT_CSV="${SOURCE_MAV0}/state_groundtruth_estimate0/data.csv"
  fi
fi

for path in "$IMAGEDIR" "$CALIB" "$IMU_PRIOR" "$FRAMES_CSV" "$GT_CSV" "$CAMERA_SENSOR_YAML" "$IMU_SENSOR_YAML"; do
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
  local debug_args=()
  if [[ "$RUN_IMU_DEBUG" == "1" ]]; then
    debug_args=(--imu_ba_debug --imu_ba_debug_path "$debug")
  fi

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
    "${debug_args[@]}" \
    "$@"

  if [[ "$RUN_PLY" == "1" ]]; then
    "$PYTHON" export_reconstruction_ply.py "$recon" --output "$OUT_DIR/${label}.ply"
  fi
  if [[ -f "$debug" ]]; then
    RUN_ARGS+=(--run "${label}=${recon},${debug}")
  else
    RUN_ARGS+=(--run "${label}=${recon}")
  fi
}

if [[ -n "$ORIGINAL_RECON" ]]; then
  if [[ ! -f "$ORIGINAL_RECON" ]]; then
    echo "ORIGINAL_RECON does not exist: $ORIGINAL_RECON" >&2
    exit 2
  fi
  echo "[REUSE] original_droid reconstruction=$ORIGINAL_RECON"
  RUN_ARGS+=(--run "original_droid=${ORIGINAL_RECON}")
else
  run_case original_droid "$ORIGINAL_WEIGHTS"
fi

IMU_ARGS=(
  --imu_prior "$IMU_PRIOR"
  --cam_sensor_yaml "$CAMERA_SENSOR_YAML"
  --imu_sensor_yaml "$IMU_SENSOR_YAML"
  --use_pose_velocity_imu_ba
  --use_vi_initialization
  --use_learned_imu_confidence
  --use_learned_imu_ba_weight
  --use_learned_imu_residual_weights
  --use_imu_noise_model
)

if [[ -n "$FIXED_RECON" ]]; then
  if [[ ! -f "$FIXED_RECON" ]]; then
    echo "FIXED_RECON does not exist: $FIXED_RECON" >&2
    exit 2
  fi
  echo "[REUSE] imu_9dof_fixed reconstruction=$FIXED_RECON"
  RUN_ARGS+=(--run "imu_9dof_fixed=${FIXED_RECON}")
else
  run_case imu_9dof_fixed "$IMU_WEIGHTS" "${IMU_ARGS[@]}"
fi

FIXED_OUTER_CONTROL_ARGS=(
  "${IMU_ARGS[@]}"
  --imu_ba_outer_relinearize
  --imu_ba_outer_relinearize_interval "$IMU_BA_OUTER_RELINEARIZE_INTERVAL"
  --imu_ba_outer_window "$IMU_BA_OUTER_WINDOW"
)
run_case imu_9dof_fixed_outer_control "$IMU_WEIGHTS" "${FIXED_OUTER_CONTROL_ARGS[@]}"

JOINT_ARGS=(
  "${IMU_ARGS[@]}"
  --use_joint_shared_bias_ba
  --imu_ba_outer_relinearize
  --imu_joint_bias_interval "$IMU_JOINT_BIAS_INTERVAL"
  --imu_joint_bias_window "$IMU_JOINT_BIAS_WINDOW"
  --imu_joint_bias_damping "$IMU_JOINT_BIAS_DAMPING"
  --imu_joint_bias_min_edges "$IMU_JOINT_BIAS_MIN_EDGES"
  --imu_joint_bias_min_duration "$IMU_JOINT_BIAS_MIN_DURATION"
  --imu_joint_bias_min_rotation_deg "$IMU_JOINT_BIAS_MIN_ROTATION_DEG"
)
run_case imu_9dof_joint_shared_bias "$IMU_WEIGHTS" "${JOINT_ARGS[@]}"

"$PYTHON" tools/evaluate_euroc_recon.py \
  --frames-csv "$FRAMES_CSV" \
  --gt-csv "$GT_CSV" \
  --gt-frame body \
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML" \
  --common-frame-ids \
  --output-csv "$OUT_DIR/summary.csv" \
  --output-md "$OUT_DIR/summary.md" \
  "${RUN_ARGS[@]}"

# The all-run summary uses only frame IDs common to every trajectory.  This
# two-run report is the primary diagnostic for the Schur term itself and keeps
# every frame shared by the matched fixed control and joint-bias run.
"$PYTHON" tools/evaluate_euroc_recon.py \
  --frames-csv "$FRAMES_CSV" \
  --gt-csv "$GT_CSV" \
  --gt-frame body \
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML" \
  --common-frame-ids \
  --output-csv "$OUT_DIR/fixed_outer_control_vs_joint.csv" \
  --output-md "$OUT_DIR/fixed_outer_control_vs_joint.md" \
  --run "imu_9dof_fixed_outer_control=$OUT_DIR/imu_9dof_fixed_outer_control_recon.pth" \
  --run "imu_9dof_joint_shared_bias=$OUT_DIR/imu_9dof_joint_shared_bias_recon.pth"

echo "[DONE] $OUT_DIR"

#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"

DATA_ROOT="${DATA_ROOT:-/home/ubuntu/SLAM/uploads/euroc_v1_03_droid_cam0}"
IMAGEDIR="${IMAGEDIR:-${DATA_ROOT}/data}"
CALIB="${CALIB:-${DATA_ROOT}/calib.txt}"
IMU_PRIOR="${IMU_PRIOR:-${DATA_ROOT}/imu_prior.csv}"
FRAMES_CSV="${FRAMES_CSV:-${DATA_ROOT}/frames.csv}"
CAMERA_SENSOR_YAML="${CAMERA_SENSOR_YAML:-${DATA_ROOT}/cam0_sensor.yaml}"
GT_CSV="${GT_CSV:-/home/ubuntu/SLAM/uploads/euroc_v1_03_mav0/state_groundtruth_estimate0/data.csv}"

ORIGINAL_WEIGHTS="${ORIGINAL_WEIGHTS:-droid.pth}"
IMU_WEIGHTS="${IMU_WEIGHTS:-checkpoints/full_imu_loss_v1_final_002000.pth}"

MAX_FRAMES="${MAX_FRAMES:-450}"
BUFFER="${BUFFER:-1024}"
STRIDE="${STRIDE:-1}"
TURN_FRAME="${TURN_FRAME:-223}"
OUT_DIR="${OUT_DIR:-outputs/euroc_ablation_$(date +%Y%m%d_%H%M%S)}"
RUN_PLY="${RUN_PLY:-1}"
RUN_SCALE_SWEEP="${RUN_SCALE_SWEEP:-0}"
SCALES="${SCALES:-0.02 0.04 0.08 0.10}"

cd "$ROOT"
mkdir -p "$OUT_DIR"

RUN_ARGS=()

run_case() {
  local label="$1"
  local weights="$2"
  shift 2

  local recon="${OUT_DIR}/${label}_recon.pth"
  local debug="${OUT_DIR}/${label}_debug.csv"
  local ply="${OUT_DIR}/${label}.ply"

  echo
  echo "============================================================"
  echo "[RUN] ${label}"
  echo "============================================================"

  "$PYTHON" demo.py \
    --imagedir "$IMAGEDIR" \
    --calib "$CALIB" \
    --weights "$weights" \
    --stride "$STRIDE" \
    --max_frames "$MAX_FRAMES" \
    --buffer "$BUFFER" \
    --disable_vis \
    --reconstruction_path "$recon" \
    "$@"

  if [[ "$RUN_PLY" == "1" ]]; then
    "$PYTHON" export_reconstruction_ply.py "$recon" --output "$ply"
  fi

  if [[ -f "$debug" ]]; then
    RUN_ARGS+=(--run "${label}=${recon},${debug}")
  else
    RUN_ARGS+=(--run "${label}=${recon}")
  fi
}

run_case "original_droid" "$ORIGINAL_WEIGHTS"

run_case "imu_data_only" "$IMU_WEIGHTS" \
  --imu_prior "$IMU_PRIOR"

run_case "rot_ba_fixed_w0002" "$IMU_WEIGHTS" \
  --imu_prior "$IMU_PRIOR" \
  --use_imu_ba_prior \
  --imu_ba_prior_weight 0.002 \
  --imu_ba_prior_max_deg 10

run_case "full_ba_fixed_w0002" "$IMU_WEIGHTS" \
  --imu_prior "$IMU_PRIOR" \
  --use_full_imu_ba \
  --imu_ba_prior_weight 0.002 \
  --imu_ba_prior_max_deg 10 \
  --imu_full_pos_weight 0.05 \
  --imu_full_vel_weight 0.05 \
  --imu_full_bias_weight 0.001 \
  --estimate_imu_gravity \
  --imu_ba_debug \
  --imu_ba_debug_path "$OUT_DIR/full_ba_fixed_w0002_debug.csv"

run_case "full_ba_conf_fixed_w0002" "$IMU_WEIGHTS" \
  --imu_prior "$IMU_PRIOR" \
  --use_full_imu_ba \
  --use_learned_imu_confidence \
  --imu_ba_prior_weight 0.002 \
  --imu_ba_prior_max_deg 10 \
  --imu_full_pos_weight 0.05 \
  --imu_full_vel_weight 0.05 \
  --imu_full_bias_weight 0.001 \
  --estimate_imu_gravity \
  --imu_ba_debug \
  --imu_ba_debug_path "$OUT_DIR/full_ba_conf_fixed_w0002_debug.csv"

run_case "full_ba_conf_learned_scale004_info" "$IMU_WEIGHTS" \
  --imu_prior "$IMU_PRIOR" \
  --use_full_imu_ba \
  --use_learned_imu_confidence \
  --use_learned_imu_ba_weight \
  --learned_imu_ba_weight_scale 0.04 \
  --imu_ba_prior_weight 0.002 \
  --imu_ba_prior_max_deg 10 \
  --imu_full_pos_weight 0.05 \
  --imu_full_vel_weight 0.05 \
  --imu_full_bias_weight 0.001 \
  --use_imu_info_weighting \
  --estimate_imu_gravity \
  --imu_ba_debug \
  --imu_ba_debug_path "$OUT_DIR/full_ba_conf_learned_scale004_info_debug.csv"

if [[ "$RUN_SCALE_SWEEP" == "1" ]]; then
  for scale in $SCALES; do
    safe_scale="${scale//./p}"
    run_case "full_ba_conf_learned_scale${safe_scale}_info" "$IMU_WEIGHTS" \
      --imu_prior "$IMU_PRIOR" \
      --use_full_imu_ba \
      --use_learned_imu_confidence \
      --use_learned_imu_ba_weight \
      --learned_imu_ba_weight_scale "$scale" \
      --imu_ba_prior_weight 0.002 \
      --imu_ba_prior_max_deg 10 \
      --imu_full_pos_weight 0.05 \
      --imu_full_vel_weight 0.05 \
      --imu_full_bias_weight 0.001 \
      --use_imu_info_weighting \
      --estimate_imu_gravity \
      --imu_ba_debug \
      --imu_ba_debug_path "$OUT_DIR/full_ba_conf_learned_scale${safe_scale}_info_debug.csv"
  done
fi

"$PYTHON" tools/evaluate_euroc_recon.py \
  --frames-csv "$FRAMES_CSV" \
  --gt-csv "$GT_CSV" \
  --gt-frame body \
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML" \
  --turn-frame "$TURN_FRAME" \
  --output-csv "$OUT_DIR/summary.csv" \
  --output-md "$OUT_DIR/summary.md" \
  "${RUN_ARGS[@]}"

if [[ "${RUN_PLY}" == "1" ]]; then
  "$PYTHON" tools/plot_trajectory_topview.py \
    "$OUT_DIR"/*_recon.pth \
    --output "$OUT_DIR/topview.png" \
    --title "EuRoC IMU Ablation"
fi

echo
echo "[DONE] ${OUT_DIR}"
echo "summary: ${OUT_DIR}/summary.csv"
echo "report:  ${OUT_DIR}/summary.md"

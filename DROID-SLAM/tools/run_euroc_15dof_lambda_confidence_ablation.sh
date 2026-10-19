#!/usr/bin/env bash
set -euo pipefail

# Legacy controlled 15DoF scale/confidence ablation.
#
# This script intentionally learns a global IMU scale and is retained only as
# an historical control.  The current main formulation fixes the global scale
# to one and trains confidence as the sole IMU-reliability attenuation; use
# tools/run_euroc_15dof_confidence_only.sh for that experiment.
#
# State optimized by full IMU DBA:
#   [pose(6), velocity(3), accelerometer-bias(3), gyroscope-bias(3)]
#
# Training: MH_03_medium only.
# Validation: V1_02_medium and V2_02_medium.
# Final held-out test: V1_03_difficult and V2_03_difficult.
#
# Modes:
#   lambda_only       learn global IMU strength with neutral frozen c=0.5
#   joint_anchored    learn lambda + confidence; mean c is anchored to 0.5
#   joint_unanchored  learn both without the mean anchor (legacy control)

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/SLAM/datasets/euroc_stage3/raw}"
CKPT="${CKPT:-checkpoints/two_stage_15dof_conf_noise_6000_final_006000.pth}"
MODE="${MODE:-joint_anchored}"
RUN_NAME="${RUN_NAME:-full15dof_lambda_conf_${MODE}}"
STEPS="${STEPS:-3000}"
ITERS="${ITERS:-7}"
POSE_ONLY_STEPS="${POSE_ONLY_STEPS:-3}"
LR="${LR:-0.0001}"
GPUS="${GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
LOG_FREQ="${LOG_FREQ:-50}"
SAVE_FINAL="${SAVE_FINAL:-1}"
# This is only the positive initialization for a trainable Softplus parameter.
# It is not a frozen deployment weight.
INITIAL_IMU_BA_WEIGHT="${INITIAL_IMU_BA_WEIGHT:-0.002}"
CONFIDENCE_TARGET="${CONFIDENCE_TARGET:-0.50}"
CONFIDENCE_ANCHOR_WEIGHT="${CONFIDENCE_ANCHOR_WEIGHT:-0.05}"

export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

if [[ ! -f "$DATA_ROOT/MH_03_medium/mav0/cam0/data.csv" ]]; then
  echo "MH_03_medium is not a valid raw EuRoC sequence under: $DATA_ROOT" >&2
  exit 2
fi
if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint does not exist: $CKPT" >&2
  exit 2
fi
if (( ITERS <= POSE_ONLY_STEPS )); then
  echo "ITERS must be greater than POSE_ONLY_STEPS" >&2
  exit 2
fi

ARGS=(
  --name "$RUN_NAME"
  --ckpt "$CKPT"
  --datasets euroc
  --datapath "$DATA_ROOT"
  --euroc_sequence_include MH_03_medium
  --euroc_camera_frame_stride 2
  --imu_train_frame_strides 3 4 5
  --gpus "$GPUS"
  --steps "$STEPS"
  --iters "$ITERS"
  --imu_train_pose_only_steps "$POSE_ONLY_STEPS"
  --lr "$LR"
  --num_workers "$NUM_WORKERS"
  --log_freq "$LOG_FREQ"
  --save_freq "$SAVE_FREQ"
  --w_imu 0.05
  --freeze_non_imu
  --freeze_global_imu_bias
  --imu_ba_weight "$INITIAL_IMU_BA_WEIGHT"
  --use_imu_ba
  --use_full_imu_ba
  --use_imu_loss
  --use_full_imu_loss
  --use_dvi_imu_confidence
  --reset_imu_confidence_head
  --imu_confidence_reset_value "$CONFIDENCE_TARGET"
  --use_imu_noise_model
  --imu_require
  --imu_gravity 0 0 -9.81
  --imu_full_bias_weight 0.001
  --imu_loss_bias_weight 0.001
  --w_imu_bias 0
  --w_imu_confidence_saturation 0.01
  --imu_confidence_saturation_ceiling 0.90
  --w_imu_confidence_nll 0.02
)

case "$MODE" in
  lambda_only)
    ARGS+=(--freeze_imu_confidence_head)
    ;;
  joint_anchored)
    TARGETS=()
    for ((i = 0; i < ITERS - POSE_ONLY_STEPS; ++i)); do
      TARGETS+=("$CONFIDENCE_TARGET")
    done
    ARGS+=(
      --w_imu_confidence_schedule "$CONFIDENCE_ANCHOR_WEIGHT"
      --imu_confidence_schedule_targets "${TARGETS[@]}"
    )
    ;;
  joint_unanchored)
    ;;
  *)
    echo "MODE must be lambda_only, joint_anchored, or joint_unanchored; got: $MODE" >&2
    exit 2
    ;;
esac

if [[ "$SAVE_FINAL" == "1" ]]; then
  ARGS+=(--save_final)
elif [[ "$SAVE_FINAL" != "0" ]]; then
  echo "SAVE_FINAL must be 0 or 1, got: $SAVE_FINAL" >&2
  exit 2
fi

echo "[EXPERIMENT] state=15DoF, mode=$MODE, global_lambda=trainable, confidence_target=$CONFIDENCE_TARGET"
exec "$PYTHON" train.py "${ARGS[@]}"

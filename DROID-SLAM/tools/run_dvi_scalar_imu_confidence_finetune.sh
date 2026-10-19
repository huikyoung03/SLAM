#!/usr/bin/env bash
set -euo pipefail

# DVI stage-3 analogue: freeze every visual parameter and every global IMU
# scale/bias parameter; train only the scalar 1x1+average-pooling confidence
# head.  No feature-metric factor is enabled by this repository's recipe.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/SLAM/datasets/tartanair_v2}"
CKPT="${CKPT:-checkpoints/two_stage_15dof_conf_noise_6000_final_006000.pth}"
RUN_NAME="${RUN_NAME:-dvi_scalar_imu_confidence_3000}"
STEPS="${STEPS:-3000}"
ITERS="${ITERS:-5}"
POSE_ONLY_STEPS="${POSE_ONLY_STEPS:-3}"
LR="${LR:-0.0001}"
GPUS="${GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
LOG_FREQ="${LOG_FREQ:-50}"
SAVE_FINAL="${SAVE_FINAL:-1}"

export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "TartanAir V2 dataset does not exist: $DATA_ROOT" >&2
  exit 2
fi
if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint does not exist: $CKPT" >&2
  exit 2
fi
if (( POSE_ONLY_STEPS >= ITERS )); then
  echo "POSE_ONLY_STEPS must be smaller than ITERS" >&2
  exit 2
fi

ARGS=(
  --name "$RUN_NAME"
  --ckpt "$CKPT"
  --datasets tartan_v2
  --datapath "$DATA_ROOT"
  --gpus "$GPUS"
  --steps "$STEPS"
  --iters "$ITERS"
  --imu_train_pose_only_steps "$POSE_ONLY_STEPS"
  --lr "$LR"
  --num_workers "$NUM_WORKERS"
  --log_freq "$LOG_FREQ"
  --save_freq "$SAVE_FREQ"
  --freeze_non_imu
  --freeze_global_imu_bias
  --freeze_imu_ba_weight
  --use_imu_ba
  --use_full_imu_ba
  --use_imu_loss
  --use_full_imu_loss
  --use_dvi_imu_confidence
  --use_imu_noise_model
  --imu_require
  --imu_train_frame_strides 1 2 4 8
  --imu_train_gyro_noise_density 1.6968e-4
  --imu_train_accel_noise_density 2.0e-3
  --imu_gravity 0 0 0
  --imu_full_bias_weight 0.001
  --imu_loss_bias_weight 0.001
  --w_imu_bias 0
  --w_imu_confidence_saturation 0.01
  --imu_confidence_saturation_ceiling 0.90
  --w_imu_confidence_nll 0.02
)

if [[ "$SAVE_FINAL" == "1" ]]; then
  ARGS+=(--save_final)
elif [[ "$SAVE_FINAL" != "0" ]]; then
  echo "SAVE_FINAL must be 0 or 1, got: $SAVE_FINAL" >&2
  exit 2
fi

exec "$PYTHON" train.py "${ARGS[@]}"

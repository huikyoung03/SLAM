#!/usr/bin/env bash

 
set -euo pipefail

# Clean fixed-bias 9DoF fine-tuning recipe.
#
# This starts from the official visual DROID checkpoint, after the TartanAir
# IMU axis conversion and unrolled velocity-state preservation fixes.  Do not
# resume the earlier Hospital 8k checkpoints here: they predate those fixes
# and are retained only as diagnostic baselines.
#
# Trainable components are the IMU-confidence GRU head, a global IMU factor
# weight, and learned relative position/velocity weights.  Local DBA optimizes
# pose, velocity, and dense inverse depth; gravity and sensor biases stay
# fixed for this 9DoF experiment.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/SLAM/datasets/tartanair_v2}"
CKPT="${CKPT:-droid.pth}"
W_IMU="${W_IMU:-0.05}"
RUN_NAME="${RUN_NAME:-pose_velocity_9dof_axisfix_hospital_wimu${W_IMU}}"
STEPS="${STEPS:-8000}"
ITERS="${ITERS:-6}"
LR="${LR:-0.0001}"
GPUS="${GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-50}"
SAVE_FINAL="${SAVE_FINAL:-1}"
CONF_SAT_WEIGHT="${CONF_SAT_WEIGHT:-0.01}"
CONF_SAT_CEILING="${CONF_SAT_CEILING:-0.90}"
RESIDUAL_WEIGHT_PRIOR="${RESIDUAL_WEIGHT_PRIOR:-0.001}"

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

ARGS=(
  --name "$RUN_NAME"
  --ckpt "$CKPT"
  --datasets tartan_v2
  --datapath "$DATA_ROOT"
  --gpus "$GPUS"
  --steps "$STEPS"
  --iters "$ITERS"
  --lr "$LR"
  --num_workers "$NUM_WORKERS"
  --log_freq "$LOG_FREQ"
  --save_freq "$SAVE_FREQ"
  --w_imu "$W_IMU"
  --use_imu_ba
  --use_full_imu_ba
  --use_pose_velocity_imu_ba
  --use_imu_loss
  --use_full_imu_loss
  --freeze_non_imu
  --use_imu_noise_model
  --use_learned_imu_residual_weights
  --imu_require
  --reset_imu_confidence_head
  --imu_confidence_reset_value 0.5
  --w_imu_confidence_saturation "$CONF_SAT_WEIGHT"
  --imu_confidence_saturation_ceiling "$CONF_SAT_CEILING"
  --w_imu_residual_weight_prior "$RESIDUAL_WEIGHT_PRIOR"
  --imu_train_frame_strides 1 2 4 8
  --imu_train_gyro_noise_density 1.6968e-4
  --imu_train_accel_noise_density 2.0e-3
  # TartanAir acc_nograv_body.txt is already gravity-free.
  --imu_gravity 0 0 0
  --imu_full_bias_weight 0
  --imu_loss_bias_weight 0
  --w_imu_bias 0
)

if [[ "$SAVE_FINAL" == "1" ]]; then
  ARGS+=(--save_final)
fi

exec "$PYTHON" train.py "${ARGS[@]}"

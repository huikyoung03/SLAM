#!/usr/bin/env bash
set -euo pipefail

# Fixed-bias 9DoF fine-tuning recipe.
#
# The trainable state is limited to the pretrained GRU IMU-confidence head,
# the learned global IMU factor weight, and learned relative position/velocity
# weights. Per-keyframe bias states and global bias parameters remain fixed.
# Preintegration covariance provides interval-dependent weighting; no legacy
# residual threshold, information clip, confidence floor, or heuristic gate is
# enabled by this launcher.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/SLAM/datasets/tartanair_v2}"
# Start confidence calibration from the latest axis/frame-corrected 9DoF
# checkpoint. It retains the learned global and relative IMU weights while
# the final confidence projection is deliberately recalibrated below.
CKPT="${CKPT:-checkpoints/pose_velocity_9dof_axisfix_hospital_wimu0.05_final_008000.pth}"
# Do not use the generic NAME environment variable: on this machine it is set
# to the host name and previously leaked into checkpoint file names.
RUN_NAME="${RUN_NAME:-pose_velocity_9dof_confcal_nll_v2}"
STEPS="${STEPS:-2000}"
ITERS="${ITERS:-6}"
LR="${LR:-0.0001}"
GPUS="${GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_FREQ="${SAVE_FREQ:-500}"
LOG_FREQ="${LOG_FREQ:-50}"
SAVE_FINAL="${SAVE_FINAL:-1}"
CONF_SAT_WEIGHT="${CONF_SAT_WEIGHT:-0.01}"
CONF_SAT_CEILING="${CONF_SAT_CEILING:-0.90}"
CONF_NLL_WEIGHT="${CONF_NLL_WEIGHT:-0.02}"
RESIDUAL_WEIGHT_PRIOR="${RESIDUAL_WEIGHT_PRIOR:-0.001}"
RESET_CONFIDENCE_HEAD="${RESET_CONFIDENCE_HEAD:-1}"

export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "TartanAir V2 dataset does not exist: $DATA_ROOT" >&2
  exit 2
fi
if [[ ! -f "$CKPT" ]]; then
  echo "IMU checkpoint does not exist: $CKPT" >&2
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
  --use_imu_ba
  --use_full_imu_ba
  --use_pose_velocity_imu_ba
  --use_imu_loss
  --use_full_imu_loss
  --freeze_non_imu
  --use_imu_noise_model
  --use_learned_imu_residual_weights
  --imu_require
  --w_imu_confidence_saturation "$CONF_SAT_WEIGHT"
  --imu_confidence_saturation_ceiling "$CONF_SAT_CEILING"
  --w_imu_confidence_nll "$CONF_NLL_WEIGHT"
  --w_imu_residual_weight_prior "$RESIDUAL_WEIGHT_PRIOR"
  --imu_train_frame_strides 1 2 4 8
  --imu_train_gyro_noise_density 1.6968e-4
  --imu_train_accel_noise_density 2.0e-3
  --imu_gravity 0 0 0
  --imu_full_bias_weight 0
  --imu_loss_bias_weight 0
  --w_imu_bias 0
)

if [[ "$RESET_CONFIDENCE_HEAD" == "1" ]]; then
  ARGS+=(--reset_imu_confidence_head --imu_confidence_reset_value 0.5)
elif [[ "$RESET_CONFIDENCE_HEAD" != "0" ]]; then
  echo "RESET_CONFIDENCE_HEAD must be 0 or 1, got: $RESET_CONFIDENCE_HEAD" >&2
  exit 2
fi

if [[ "$SAVE_FINAL" == "1" ]]; then
  ARGS+=(--save_final)
fi

exec "$PYTHON" train.py "${ARGS[@]}"

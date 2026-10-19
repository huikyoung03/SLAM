#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the component-wise IMU confidence model introduced after the
# scalar 6k two-stage baseline. The initial component outputs reproduce the
# scalar checkpoint exactly; training then learns independent confidence for
# rotation, velocity, and position while seeing preintegration quality.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/SLAM/datasets/tartanair_v2}"
CKPT="${CKPT:-checkpoints/two_stage_15dof_conf_noise_6000_final_006000.pth}"
RUN_NAME="${RUN_NAME:-two_stage_15dof_component_confidence_quality_3000}"
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
FIRST_IMAGE="$(find "$DATA_ROOT" -type f -path '*/Data_*/P*/image_lcam_front/*.png' -print -quit 2>/dev/null)"
if [[ -z "$FIRST_IMAGE" ]]; then
  echo "DATA_ROOT is not a TartanAir V2 root with lcam_front trajectories: $DATA_ROOT" >&2
  echo "For this fine-tuning run, use:" >&2
  echo "  export DATA_ROOT=/home/ubuntu/SLAM/datasets/tartanair_v2" >&2cd /home/ubuntu/DROID-SLAM

DATA_ROOT=/home/ubuntu/SLAM/datasets/tartanair_v2 \
bash tools/run_component_imu_confidence_finetune.sh
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
  --use_imu_ba
  --use_full_imu_ba
  --use_imu_loss
  --use_full_imu_loss
  --use_component_imu_confidence
  --use_imu_noise_model
  --use_learned_imu_residual_weights
  --imu_require
  --imu_train_frame_strides 1 2 4 8
  --imu_train_gyro_noise_density 1.6968e-4
  --imu_train_accel_noise_density 2.0e-3
  # TartanAir acc_nograv_body is gravity-free by construction.
  --imu_gravity 0 0 0
  --imu_full_bias_weight 0.001
  --imu_loss_bias_weight 0.001
  --w_imu_bias 0.0001
  --w_imu_confidence_saturation 0.01
  --imu_confidence_saturation_ceiling 0.90
  --w_imu_confidence_nll 0.02
  --w_imu_residual_weight_prior 0.001
)

if [[ "$SAVE_FINAL" == "1" ]]; then
  ARGS+=(--save_final)
elif [[ "$SAVE_FINAL" != "0" ]]; then
  echo "SAVE_FINAL must be 0 or 1, got: $SAVE_FINAL" >&2
  exit 2
fi

exec "$PYTHON" train.py "${ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

# DVI-style stage-3 training for a real EuRoC IMU stream.  The pretrained
# visual network, global IMU biases, and global IMU BA scale remain frozen;
# only the scalar 1x1 + spatial-average confidence head is trainable.  The
# global BA scale is reset explicitly to the same value used at inference so
# the confidence head never learns under a different visual:IMU tradeoff.
#
# Camera input is decimated from EuRoC's native 20 Hz to 10 Hz.  The sampled
# clip gaps 3, 4, and 5 therefore correspond to 0.3, 0.4, and 0.5 seconds.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/SLAM/datasets/euroc_stage3/raw}"
CKPT="${CKPT:-checkpoints/two_stage_15dof_conf_noise_6000_final_006000.pth}"
RUN_NAME="${RUN_NAME:-dvi_mh03_10fps_parity_ramp_8000}"
STEPS="${STEPS:-8000}"
# Three visual-only warm-up updates followed by four full 15DoF updates. This
# makes one training clip expose all four weak manual curriculum targets.
ITERS="${ITERS:-7}"
POSE_ONLY_STEPS="${POSE_ONLY_STEPS:-3}"
LR="${LR:-0.0001}"
GPUS="${GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-50}"
# The step-8000 periodic checkpoint is already the final model, so avoid a
# redundant *_final_008000.pth by default.
SAVE_FINAL="${SAVE_FINAL:-0}"
IMU_BA_WEIGHT="${IMU_BA_WEIGHT:-0.002}"
# DVI-inspired manual ramp used only as a weak mean-confidence teacher. It is
# not claimed to be the unpublished exact DVI-SLAM tuned vector.
SCHEDULE_WEIGHT="${SCHEDULE_WEIGHT:-0.05}"
SCHEDULE_TARGETS="${SCHEDULE_TARGETS:-0.25 0.50 0.75 1.00}"

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
if (( POSE_ONLY_STEPS >= ITERS )); then
  echo "POSE_ONLY_STEPS must be smaller than ITERS" >&2
  exit 2
fi
read -r -a SCHEDULE_TARGET_ARGS <<< "$SCHEDULE_TARGETS"
if (( ${#SCHEDULE_TARGET_ARGS[@]} == 0 )); then
  echo "SCHEDULE_TARGETS must contain at least one confidence target" >&2
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
  --freeze_non_imu
  --freeze_global_imu_bias
  --freeze_imu_ba_weight
  --imu_ba_weight "$IMU_BA_WEIGHT"
  --use_imu_ba
  --use_full_imu_ba
  --use_imu_loss
  --use_full_imu_loss
  --use_dvi_imu_confidence
  --reset_imu_confidence_head
  --imu_confidence_reset_value 0.5
  --use_imu_noise_model
  --imu_require
  --imu_gravity 0 0 -9.81
  --imu_full_bias_weight 0.001
  --imu_loss_bias_weight 0.001
  --w_imu_bias 0
  --w_imu_confidence_saturation 0.01
  --imu_confidence_saturation_ceiling 0.90
  --w_imu_confidence_nll 0.02
  --w_imu_confidence_schedule "$SCHEDULE_WEIGHT"
  --imu_confidence_schedule_targets "${SCHEDULE_TARGET_ARGS[@]}"
)

if [[ "$SAVE_FINAL" == "1" ]]; then
  ARGS+=(--save_final)
elif [[ "$SAVE_FINAL" != "0" ]]; then
  echo "SAVE_FINAL must be 0 or 1, got: $SAVE_FINAL" >&2
  exit 2
fi

exec "$PYTHON" train.py "${ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

# Direct learned IMU reliability experiment (no global IMU lambda).
#
# The only trainable network blocks are
#   ConvGRU visual state -> 32D projection
#   seven preintegration-quality features -> 16D MLP
#   fused 48D feature -> [log c_R, log c_v, log c_p]
# and BA applies c in (0, 1] to covariance-normalized R/v/p rows. The
# physical sensor covariance defines full IMU strength; the network only
# attenuates unreliable intervals. Thus the historical fixed 0.05 p/v
# coefficients and imu_ba_log_weight are not active. Bias random-walk rows
# remain sensor-model based.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/SLAM/datasets/euroc_stage3/raw}"
CKPT="${CKPT:-droid.pth}"
RUN_NAME="${RUN_NAME:-learned_attenuation_rvp_calibrated_mh03_mh04_2000}"
STEPS="${STEPS:-2000}"
ITERS="${ITERS:-7}"
POSE_ONLY_STEPS="${POSE_ONLY_STEPS:-3}"
LR="${LR:-0.0001}"
GPUS="${GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
LOG_FREQ="${LOG_FREQ:-50}"
POST_BA_WEIGHT="${POST_BA_WEIGHT:-1.0}"
AUGMENT_CALIBRATION_WEIGHT="${AUGMENT_CALIBRATION_WEIGHT:-1.0}"
TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-MH_03_medium MH_04_difficult}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/home/ubuntu/SLAM/datasets/euroc_stage3/motion_filter_manifests_original_protocol}"

export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

read -r -a SEQUENCES <<< "$TRAIN_SEQUENCES"
if (( ${#SEQUENCES[@]} == 0 )); then
  echo "TRAIN_SEQUENCES must not be empty" >&2
  exit 2
fi
if [[ ! -f "$CKPT" ]]; then
  echo "checkpoint not found: $CKPT" >&2
  exit 2
fi
if (( ITERS <= POSE_ONLY_STEPS )); then
  echo "ITERS must be greater than POSE_ONLY_STEPS" >&2
  exit 2
fi
for sequence in "${SEQUENCES[@]}"; do
  if [[ ! -f "$DATA_ROOT/$sequence/mav0/cam0/data.csv" ]]; then
    echo "missing EuRoC sequence: $DATA_ROOT/$sequence" >&2
    exit 2
  fi
  if [[ ! -f "$MANIFEST_ROOT/$sequence.csv" ]]; then
    echo "missing strict MotionFilter manifest: $MANIFEST_ROOT/$sequence.csv" >&2
    exit 2
  fi
done

POST_BA_ARGS=()
# The calibration-only pretraining phase has no pose loss, so multi-horizon
# supervision must be absent. Joint fine-tuning retains the exact previous
# multi-horizon objective.
case "$POST_BA_WEIGHT" in
  0|0.0|0.00|0.000|0.0000) ;;
  *) POST_BA_ARGS+=(--imu_confidence_post_ba_multi_horizon) ;;
esac

exec "$PYTHON" train.py \
  --name "$RUN_NAME" \
  --ckpt "$CKPT" \
  --datasets euroc \
  --datapath "$DATA_ROOT" \
  --euroc_sequence_include "${SEQUENCES[@]}" \
  --euroc_original_protocol \
  --euroc_camera_frame_stride 2 \
  --euroc_keyframe_manifest_root "$MANIFEST_ROOT" \
  --imu_train_frame_strides 1 \
  --balance_sequence_sampling \
  --gpus "$GPUS" \
  --steps "$STEPS" \
  --iters "$ITERS" \
  --imu_train_pose_only_steps "$POSE_ONLY_STEPS" \
  --lr "$LR" \
  --num_workers "$NUM_WORKERS" \
  --log_freq "$LOG_FREQ" \
  --save_freq "$SAVE_FREQ" \
  --save_final \
  --freeze_non_imu \
  --freeze_global_imu_bias \
  --use_imu_ba \
  --use_full_imu_ba \
  --use_learned_imu_precision \
  --use_imu_noise_model \
  --use_imu_full_covariance_weighting \
  --use_imu_bias_random_walk \
  --imu_train_accel_bias_random_walk 0.003 \
  --imu_train_gyro_bias_random_walk 1.9393e-5 \
  --imu_require \
  --imu_gravity 0 0 -9.81 \
  --w1 0 --w2 0 --w3 0 --w_imu 0 --w_imu_bias 0 \
  --w_imu_confidence_post_ba_pose "$POST_BA_WEIGHT" \
  --w_imu_precision_augmentation_calibration "$AUGMENT_CALIBRATION_WEIGHT" \
  "${POST_BA_ARGS[@]}" \
  --use_imu_confidence_augmentation \
  --disable_flow_loss

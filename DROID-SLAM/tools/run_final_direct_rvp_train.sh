#!/usr/bin/env bash
set -euo pipefail

# Train the final direct R/V/P precision architecture.  This is intentionally
# the only public training route: frozen visual DROID, no global IMU lambda,
# no learned residual-weight scalar, and no discrepancy/innovation adapter.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${TRAIN_DATA_ROOT:?Set TRAIN_DATA_ROOT to the EuRoC training root}"
MANIFEST_ROOT="${MANIFEST_ROOT:?Set MANIFEST_ROOT to original-protocol manifests}"
BASE_WEIGHTS="${BASE_WEIGHTS:-${ROOT}/droid.pth}"
PRETRAIN_NAME="${PRETRAIN_NAME:-final_direct_rvp_pretrain500}"
JOINT_NAME="${JOINT_NAME:-final_direct_rvp_joint2000}"
GPUS="${GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"

cd "$ROOT"
export LD_LIBRARY_PATH="${CONDA_PREFIX:+${CONDA_PREFIX}/lib:}${LD_LIBRARY_PATH:-}"
[[ -f "$BASE_WEIGHTS" ]] || { echo "Missing BASE_WEIGHTS: $BASE_WEIGHTS" >&2; exit 2; }

common_args=(
  --datasets euroc
  --datapath "$DATA_ROOT"
  --euroc_sequence_include MH_03_medium MH_04_difficult
  --euroc_original_protocol
  --euroc_camera_frame_stride 2
  --euroc_keyframe_manifest_root "$MANIFEST_ROOT"
  --imu_train_frame_strides 1
  --balance_sequence_sampling
  --gpus "$GPUS"
  --iters 7
  --imu_train_pose_only_steps 3
  --num_workers "$NUM_WORKERS"
  --freeze_non_imu
  --freeze_global_imu_bias
  --use_imu_ba
  --use_full_imu_ba
  --use_learned_imu_precision
  --use_imu_noise_model
  --use_imu_full_covariance_weighting
  --use_imu_bias_random_walk
  --imu_train_accel_bias_random_walk 0.003
  --imu_train_gyro_bias_random_walk 1.9393e-5
  --imu_require
  --imu_gravity 0 0 -9.81
  --w1 0 --w2 0 --w3 0 --w_imu 0 --w_imu_bias 0
  --w_imu_precision_augmentation_calibration 1.0
  --use_imu_confidence_augmentation
  --disable_flow_loss
  --save_final
)

echo '[phase A] 500-step IMU-quality calibration pretraining'
"$PYTHON" train.py \
  --name "$PRETRAIN_NAME" \
  --ckpt "$BASE_WEIGHTS" \
  --steps 500 \
  --lr 1e-4 \
  --log_freq 50 \
  --save_freq 500 \
  --w_imu_confidence_post_ba_pose 0.0 \
  "${common_args[@]}"

PRETRAIN_CKPT="${PRETRAIN_CKPT:-checkpoints/${PRETRAIN_NAME}_final_000500.pth}"
[[ -f "$PRETRAIN_CKPT" ]] || { echo "Missing phase-A checkpoint: $PRETRAIN_CKPT" >&2; exit 2; }

echo '[phase B] 2,000-step joint post-BA pose + calibration fine-tuning'
"$PYTHON" train.py \
  --name "$JOINT_NAME" \
  --ckpt "$PRETRAIN_CKPT" \
  --steps 2000 \
  --lr 1e-4 \
  --log_freq 50 \
  --save_freq 1000 \
  --w_imu_confidence_post_ba_pose 1.0 \
  --imu_confidence_post_ba_multi_horizon \
  "${common_args[@]}"

echo "[DONE] Evaluate checkpoints/${JOINT_NAME}_001000.pth and _final_002000.pth on a held-out sequence."

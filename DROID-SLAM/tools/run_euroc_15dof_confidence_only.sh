#!/usr/bin/env bash
set -euo pipefail

# 15DoF IMU-reliability experiment.
#
# State: [pose(6), velocity(3), accelerometer bias(3), gyroscope bias(3)]
#
# The global IMU factor multiplier is fixed to 1.0.  The DVI scalar
# confidence is the only learned attenuation of each temporal IMU factor:
#
#   information_ij = covariance_relative_information_ij * confidence_ij
#
# A confidence of one therefore means that the IMU preintegration enters DBA
# at full covariance-relative strength. The single head is calibrated from
# the absolute and relative pre-BA visual-pose/IMU mismatch, while train-only
# noise, bias-drift, and partial-dropout augmentation supplies low-quality
# examples. Direct visual temporal edges use GRU+quality confidence; missing
# temporal edges use the same trained quality branch rather than a fixed 1.0.
# This intentionally does *not* use the legacy learned global
# imu_ba_log_weight, a manual confidence schedule, or a post-BA residual.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/SLAM/datasets/euroc_stage3/raw}"
CKPT="${CKPT:-checkpoints/two_stage_15dof_conf_noise_6000_final_006000.pth}"
RUN_NAME="${RUN_NAME:-full15dof_confidence_only_dvi_quality_mh03_mh04_8000}"
# The full OneCycle schedule is 8k steps. The early-stop controller can set
# STOP_AFTER_STEPS to 2k/4k/6k, validate after the process exits, then resume
# the same optimizer/scheduler state for the next segment.
STEPS="${STEPS:-8000}"
STOP_AFTER_STEPS="${STOP_AFTER_STEPS:-$STEPS}"
ITERS="${ITERS:-7}"
POSE_ONLY_STEPS="${POSE_ONLY_STEPS:-3}"
LR="${LR:-0.0001}"
GPUS="${GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-50}"
SAVE_FINAL="${SAVE_FINAL:-1}"
SAVE_TRAINING_STATE="${SAVE_TRAINING_STATE:-0}"
RESUME_TRAINING_STATE="${RESUME_TRAINING_STATE:-}"
# Resetting the confidence head is a *new-run* operation only.  A resumed
# segment must preserve both the DVI projection and the IMU-quality fusion
# learned in the preceding segment.
RESET_IMU_CONFIDENCE_HEAD="${RESET_IMU_CONFIDENCE_HEAD:-1}"
# This is an initial sigmoid bias, not a fixed IMU multiplier.  The visual
# DBA loss learns the global calibration from it; 0.5 is neutral and does not
# import the old hand-set lambda=0.002 into the new architecture.
INITIAL_CONFIDENCE="${INITIAL_CONFIDENCE:-0.5}"
# The old NLL consumes residuals after full IMU DBA and is deliberately off.
# The new ranking loss uses only the last visual-only pose estimate.
CONFIDENCE_NLL_WEIGHT="${CONFIDENCE_NLL_WEIGHT:-0}"
CONFIDENCE_POSE_PRIOR_WEIGHT="${CONFIDENCE_POSE_PRIOR_WEIGHT:-0.10}"
# The following are supervision-shape hyperparameters, not a global IMU
# multiplier or a fixed confidence schedule. The EMA is saved in each
# checkpoint/trainstate, so segmented training retains one calibration scale.
CONFIDENCE_POSE_PRIOR_ABSOLUTE_WEIGHT="${CONFIDENCE_POSE_PRIOR_ABSOLUTE_WEIGHT:-1.0}"
CONFIDENCE_POSE_PRIOR_EMA_MOMENTUM="${CONFIDENCE_POSE_PRIOR_EMA_MOMENTUM:-0.98}"
CONFIDENCE_POSE_PRIOR_TEMPERATURE="${CONFIDENCE_POSE_PRIOR_TEMPERATURE:-0.75}"
# The raw-quality branch treats supplied covariance as nominal full IMU
# information. The fused visual-GRU+quality confidence preserves normal clean
# intervals at full strength and attenuates only the learned high-residual
# visual/IMU-disagreement tail. This remains one runtime scalar.
CONFIDENCE_POSE_PRIOR_ABSOLUTE_TARGET_MODE="${CONFIDENCE_POSE_PRIOR_ABSOLUTE_TARGET_MODE:-one_sided_nominal_quality}"
CONFIDENCE_AUG_PROBABILITY="${CONFIDENCE_AUG_PROBABILITY:-0.35}"
CONFIDENCE_AUG_GYRO_NOISE_STD="${CONFIDENCE_AUG_GYRO_NOISE_STD:-0.03}"
CONFIDENCE_AUG_ACCEL_NOISE_STD="${CONFIDENCE_AUG_ACCEL_NOISE_STD:-0.30}"
CONFIDENCE_AUG_GYRO_BIAS_DRIFT_STD="${CONFIDENCE_AUG_GYRO_BIAS_DRIFT_STD:-0.01}"
CONFIDENCE_AUG_ACCEL_BIAS_DRIFT_STD="${CONFIDENCE_AUG_ACCEL_BIAS_DRIFT_STD:-0.10}"
CONFIDENCE_AUG_DROPOUT_PROBABILITY="${CONFIDENCE_AUG_DROPOUT_PROBABILITY:-0.35}"
CONFIDENCE_AUG_DROPOUT_MAX_LENGTH="${CONFIDENCE_AUG_DROPOUT_MAX_LENGTH:-2}"
CONFIDENCE_AUG_DROPOUT_INFO_SCALE="${CONFIDENCE_AUG_DROPOUT_INFO_SCALE:-0.05}"
# The scalar DVI confidence consumes both its visual-GRU logit and direct
# preintegrated-IMU quality features. Set to 0 only for the visual-only DVI
# ablation; confidence-only factor weighting remains active either way.
USE_DVI_IMU_QUALITY_FUSION="${USE_DVI_IMU_QUALITY_FUSION:-1}"
# Rotation remains the fixed reference scale.  Optionally learn only the
# position/velocity category-relative IMU residual weights; this is distinct
# from per-interval confidence, which still controls interval reliability.
# The log-space prior prevents trivial factor collapse.
LEARN_IMU_RESIDUAL_WEIGHTS="${LEARN_IMU_RESIDUAL_WEIGHTS:-0}"
IMU_RESIDUAL_WEIGHT_PRIOR="${IMU_RESIDUAL_WEIGHT_PRIOR:-0.01}"
# A fresh calibration should not inherit the small p/v correction contained
# in the TartanAir pretraining checkpoint.  This is a new-run operation;
# segmented resumes preserve the checkpointed values.
RESET_IMU_RESIDUAL_WEIGHTS="${RESET_IMU_RESIDUAL_WEIGHTS:-0}"
# Match the original DROID EuRoC protocol and all existing VIO validation
# runners: feed every second 20 Hz EuRoC image (10 Hz candidate stream) to
# MotionFilter.  A generic demo.py parser default is not the EuRoC benchmark
# protocol; the original EuRoC evaluator itself uses images[::2].
CAMERA_FRAME_STRIDE="${CAMERA_FRAME_STRIDE:-2}"
# Whitespace-separated sequence names, e.g. "MH_03_medium MH_04_difficult".
# The balanced sampler below gives each selected sequence equal expected
# number of clips despite a different recording length.
TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-MH_03_medium MH_04_difficult}"
# One <sequence>.csv per training sequence, exported by the real DROID
# MotionFilter from the same 10 Hz candidate stream. Consecutive rows already
# have variable physical gaps, so stride 1 is the correct ordered-training
# stride; the reader composes every raw IMU sample over each such gap.
EUROC_KEYFRAME_MANIFEST_ROOT="${EUROC_KEYFRAME_MANIFEST_ROOT:-/home/ubuntu/SLAM/datasets/euroc_stage3/motion_filter_manifests_stride2}"
IMU_TRAIN_FRAME_STRIDES="${IMU_TRAIN_FRAME_STRIDES:-1}"

export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

read -r -a TRAIN_SEQUENCE_ARGS <<< "$TRAIN_SEQUENCES"
read -r -a IMU_TRAIN_STRIDE_ARGS <<< "$IMU_TRAIN_FRAME_STRIDES"
if (( ${#TRAIN_SEQUENCE_ARGS[@]} == 0 )); then
  echo "TRAIN_SEQUENCES must contain at least one EuRoC sequence name" >&2
  exit 2
fi
if (( ${#IMU_TRAIN_STRIDE_ARGS[@]} == 0 )); then
  echo "IMU_TRAIN_FRAME_STRIDES must contain at least one positive integer" >&2
  exit 2
fi
for sequence in "${TRAIN_SEQUENCE_ARGS[@]}"; do
  if [[ ! -f "$DATA_ROOT/$sequence/mav0/cam0/data.csv" ]]; then
    echo "$sequence is not a valid raw EuRoC sequence under: $DATA_ROOT" >&2
    exit 2
  fi
  if [[ ! -f "$EUROC_KEYFRAME_MANIFEST_ROOT/$sequence.csv" ]]; then
    echo "Missing frozen MotionFilter manifest: $EUROC_KEYFRAME_MANIFEST_ROOT/$sequence.csv" >&2
    exit 2
  fi
done
if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint does not exist: $CKPT" >&2
  exit 2
fi
if (( ITERS <= POSE_ONLY_STEPS )); then
  echo "ITERS must be greater than POSE_ONLY_STEPS" >&2
  exit 2
fi
if (( STEPS <= 0 || STOP_AFTER_STEPS <= 0 || STOP_AFTER_STEPS > STEPS )); then
  echo "Require 0 < STOP_AFTER_STEPS <= STEPS; got STOP_AFTER_STEPS=$STOP_AFTER_STEPS, STEPS=$STEPS" >&2
  exit 2
fi
if (( CAMERA_FRAME_STRIDE <= 0 )); then
  echo "CAMERA_FRAME_STRIDE must be positive" >&2
  exit 2
fi
if [[ "$USE_DVI_IMU_QUALITY_FUSION" != "0" && "$USE_DVI_IMU_QUALITY_FUSION" != "1" ]]; then
  echo "USE_DVI_IMU_QUALITY_FUSION must be 0 or 1" >&2
  exit 2
fi
if [[ "$LEARN_IMU_RESIDUAL_WEIGHTS" != "0" && "$LEARN_IMU_RESIDUAL_WEIGHTS" != "1" ]]; then
  echo "LEARN_IMU_RESIDUAL_WEIGHTS must be 0 or 1" >&2
  exit 2
fi
if [[ "$RESET_IMU_RESIDUAL_WEIGHTS" != "0" && "$RESET_IMU_RESIDUAL_WEIGHTS" != "1" ]]; then
  echo "RESET_IMU_RESIDUAL_WEIGHTS must be 0 or 1" >&2
  exit 2
fi
if [[ "$LEARN_IMU_RESIDUAL_WEIGHTS" == "0" && "$RESET_IMU_RESIDUAL_WEIGHTS" == "1" ]]; then
  echo "RESET_IMU_RESIDUAL_WEIGHTS=1 requires LEARN_IMU_RESIDUAL_WEIGHTS=1" >&2
  exit 2
fi
if [[ -n "$RESUME_TRAINING_STATE" && "$RESET_IMU_RESIDUAL_WEIGHTS" == "1" ]]; then
  echo "RESET_IMU_RESIDUAL_WEIGHTS=1 is incompatible with RESUME_TRAINING_STATE" >&2
  exit 2
fi
if [[ "$SAVE_TRAINING_STATE" != "0" && "$SAVE_TRAINING_STATE" != "1" ]]; then
  echo "SAVE_TRAINING_STATE must be 0 or 1, got: $SAVE_TRAINING_STATE" >&2
  exit 2
fi
if [[ "$RESET_IMU_CONFIDENCE_HEAD" != "0" && "$RESET_IMU_CONFIDENCE_HEAD" != "1" ]]; then
  echo "RESET_IMU_CONFIDENCE_HEAD must be 0 or 1, got: $RESET_IMU_CONFIDENCE_HEAD" >&2
  exit 2
fi
if [[ -n "$RESUME_TRAINING_STATE" && "$RESET_IMU_CONFIDENCE_HEAD" == "1" ]]; then
  echo "RESET_IMU_CONFIDENCE_HEAD=1 is incompatible with RESUME_TRAINING_STATE; a resumed run must retain learned confidence parameters" >&2
  exit 2
fi
if [[ -n "$RESUME_TRAINING_STATE" && ! -f "$RESUME_TRAINING_STATE" ]]; then
  echo "RESUME_TRAINING_STATE does not exist: $RESUME_TRAINING_STATE" >&2
  exit 2
fi
for stride in "${IMU_TRAIN_STRIDE_ARGS[@]}"; do
  if ! [[ "$stride" =~ ^[1-9][0-9]*$ ]]; then
    echo "IMU_TRAIN_FRAME_STRIDES must contain positive integers, got: $stride" >&2
    exit 2
  fi
done

ARGS=(
  --name "$RUN_NAME"
  --ckpt "$CKPT"
  --datasets euroc
  --datapath "$DATA_ROOT"
  --euroc_sequence_include "${TRAIN_SEQUENCE_ARGS[@]}"
  --euroc_camera_frame_stride "$CAMERA_FRAME_STRIDE"
  --euroc_keyframe_manifest_root "$EUROC_KEYFRAME_MANIFEST_ROOT"
  --imu_train_frame_strides "${IMU_TRAIN_STRIDE_ARGS[@]}"
  --balance_sequence_sampling
  --gpus "$GPUS"
  --steps "$STEPS"
  --stop_after_steps "$STOP_AFTER_STEPS"
  --iters "$ITERS"
  --imu_train_pose_only_steps "$POSE_ONLY_STEPS"
  --lr "$LR"
  --num_workers "$NUM_WORKERS"
  --log_freq "$LOG_FREQ"
  --save_freq "$SAVE_FREQ"
  --w_imu 0.05
  --freeze_non_imu
  --freeze_global_imu_bias
  --use_imu_ba
  --use_full_imu_ba
  --use_imu_loss
  --use_full_imu_loss
  --use_dvi_imu_confidence
  --imu_confidence_only_weighting
  --imu_confidence_reset_value "$INITIAL_CONFIDENCE"
  --w_imu_confidence_nll "$CONFIDENCE_NLL_WEIGHT"
  --w_imu_confidence_pose_prior "$CONFIDENCE_POSE_PRIOR_WEIGHT"
  --imu_confidence_pose_prior_absolute_weight "$CONFIDENCE_POSE_PRIOR_ABSOLUTE_WEIGHT"
  --imu_confidence_pose_prior_ema_momentum "$CONFIDENCE_POSE_PRIOR_EMA_MOMENTUM"
  --imu_confidence_pose_prior_temperature "$CONFIDENCE_POSE_PRIOR_TEMPERATURE"
  --imu_confidence_pose_prior_absolute_target_mode "$CONFIDENCE_POSE_PRIOR_ABSOLUTE_TARGET_MODE"
  --use_imu_confidence_augmentation
  --imu_confidence_aug_probability "$CONFIDENCE_AUG_PROBABILITY"
  --imu_confidence_aug_gyro_noise_std "$CONFIDENCE_AUG_GYRO_NOISE_STD"
  --imu_confidence_aug_accel_noise_std "$CONFIDENCE_AUG_ACCEL_NOISE_STD"
  --imu_confidence_aug_gyro_bias_drift_std "$CONFIDENCE_AUG_GYRO_BIAS_DRIFT_STD"
  --imu_confidence_aug_accel_bias_drift_std "$CONFIDENCE_AUG_ACCEL_BIAS_DRIFT_STD"
  --imu_confidence_aug_dropout_probability "$CONFIDENCE_AUG_DROPOUT_PROBABILITY"
  --imu_confidence_aug_dropout_max_length "$CONFIDENCE_AUG_DROPOUT_MAX_LENGTH"
  --imu_confidence_aug_dropout_info_scale "$CONFIDENCE_AUG_DROPOUT_INFO_SCALE"
  --use_imu_noise_model
  --imu_require
  --imu_gravity 0 0 -9.81
  --imu_full_bias_weight 0.001
  --imu_loss_bias_weight 0.001
  --w_imu_bias 0
)

if [[ "$RESET_IMU_CONFIDENCE_HEAD" == "1" ]]; then
  ARGS+=(--reset_imu_confidence_head)
fi

if [[ "$SAVE_TRAINING_STATE" == "1" ]]; then
  ARGS+=(--save_training_state)
fi
if [[ -n "$RESUME_TRAINING_STATE" ]]; then
  ARGS+=(--resume_training_state "$RESUME_TRAINING_STATE")
fi

if [[ "$USE_DVI_IMU_QUALITY_FUSION" == "1" ]]; then
  ARGS+=(--use_dvi_imu_quality_fusion)
fi

if [[ "$LEARN_IMU_RESIDUAL_WEIGHTS" == "1" ]]; then
  ARGS+=(
    --use_learned_imu_residual_weights
    --w_imu_residual_weight_prior "$IMU_RESIDUAL_WEIGHT_PRIOR"
  )
  if [[ "$RESET_IMU_RESIDUAL_WEIGHTS" == "1" ]]; then
    ARGS+=(--reset_imu_residual_weights)
  fi
fi

if [[ "$SAVE_FINAL" == "1" ]]; then
  ARGS+=(--save_final)
elif [[ "$SAVE_FINAL" != "0" ]]; then
  echo "SAVE_FINAL must be 0 or 1, got: $SAVE_FINAL" >&2
  exit 2
fi

echo "[EXPERIMENT] state=15DoF, schedule_steps=$STEPS, stop_after_steps=$STOP_AFTER_STEPS, camera_fps=$((20 / CAMERA_FRAME_STRIDE)), clip_selected_keyframe_strides=${IMU_TRAIN_STRIDE_ARGS[*]}, motion_filter_manifest_root=$EUROC_KEYFRAME_MANIFEST_ROOT, sequences=${TRAIN_SEQUENCE_ARGS[*]}, balanced_sampling=true, global_imu_weight=1, confidence_only=true, learned_relative_pv_weights=$LEARN_IMU_RESIDUAL_WEIGHTS, reset_relative_pv_weights=$RESET_IMU_RESIDUAL_WEIGHTS, relative_pv_log_prior=$IMU_RESIDUAL_WEIGHT_PRIOR, supervisor=pre_ba_pose_prior_ranking_plus_absolute, absolute_target_mode=$CONFIDENCE_POSE_PRIOR_ABSOLUTE_TARGET_MODE, post_ba_nll=$CONFIDENCE_NLL_WEIGHT, pose_prior_weight=$CONFIDENCE_POSE_PRIOR_WEIGHT, pose_prior_absolute_weight=$CONFIDENCE_POSE_PRIOR_ABSOLUTE_WEIGHT, pose_prior_ema_momentum=$CONFIDENCE_POSE_PRIOR_EMA_MOMENTUM, pose_prior_temperature=$CONFIDENCE_POSE_PRIOR_TEMPERATURE, augmentation=noise+bias_drift+partial_dropout, dvi_imu_quality_fusion=$USE_DVI_IMU_QUALITY_FUSION, temporal_quality_fallback=enabled, initial_confidence=$INITIAL_CONFIDENCE, reset_imu_confidence_head=$RESET_IMU_CONFIDENCE_HEAD, resume_training_state=${RESUME_TRAINING_STATE:-none}"
exec "$PYTHON" train.py "${ARGS[@]}"

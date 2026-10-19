#!/usr/bin/env bash
set -euo pipefail

# Memory-safe early-stop protocol for the 15DoF confidence-only experiment:
#
#   train to 2k -> exit/release GPU -> V1_02 evaluation
#   resume to 4k -> exit/release GPU -> V1_02 evaluation
#   ... 6k, 8k
#
# ``train.py`` restores the model, Adam state, and OneCycleLR state from its
# companion trainstate file. Consequently every segment belongs to one 8k
# schedule; it is not four independent 2k fine-tunes.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
BASE_CKPT="${BASE_CKPT:-checkpoints/two_stage_15dof_conf_noise_6000_final_006000.pth}"
RUN_NAME="${RUN_NAME:-full15dof_confidence_only_dvi_quality_mh03_mh04_8000_earlystop}"
TOTAL_STEPS="${TOTAL_STEPS:-8000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2000}"
TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-MH_03_medium MH_04_difficult}"
MANIFEST_ROOT="${EUROC_KEYFRAME_MANIFEST_ROOT:-/home/ubuntu/SLAM/datasets/euroc_stage3/motion_filter_manifests_stride2}"
VALIDATION_ROOT="${EARLY_STOP_VALIDATION_ROOT:-/home/ubuntu/SLAM/datasets/euroc_validation/prepared/V1_02_medium_droid_cam0}"
OUT_ROOT="${OUT_ROOT:-outputs/$RUN_NAME}"
MIN_COVERAGE_RATIO="${EARLY_STOP_MIN_COVERAGE_RATIO:-0.95}"
# Treat a >2 mm Sim(3)-ATE increase as a real decline; smaller differences are
# typically numerical evaluation jitter rather than evidence to stop an 8k run.
DETERIORATION_TOLERANCE_M="${EARLY_STOP_DETERIORATION_TOLERANCE_M:-0.002}"
# Sim(3) ATE removes monocular-DROID's arbitrary global scale and is the
# cross-model ATE used for checkpoint selection in this experiment.
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-sim3_ate_rmse_m}"

export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

if (( TOTAL_STEPS <= 0 || CHECKPOINT_INTERVAL <= 0 || TOTAL_STEPS % CHECKPOINT_INTERVAL != 0 )); then
  echo "TOTAL_STEPS must be a positive multiple of CHECKPOINT_INTERVAL; got $TOTAL_STEPS / $CHECKPOINT_INTERVAL" >&2
  exit 2
fi
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "base checkpoint does not exist: $BASE_CKPT" >&2
  exit 2
fi
for sequence in $TRAIN_SEQUENCES; do
  if [[ ! -f "$MANIFEST_ROOT/$sequence.csv" ]]; then
    echo "missing MotionFilter manifest: $MANIFEST_ROOT/$sequence.csv" >&2
    exit 2
  fi
done
for required in data calib.txt frames.csv imu_prior_fd_jacobian.csv cam0_sensor.yaml imu0_sensor.yaml; do
  if [[ ! -e "$VALIDATION_ROOT/$required" ]]; then
    echo "validation input is missing: $VALIDATION_ROOT/$required" >&2
    exit 2
  fi
done

mkdir -p "$OUT_ROOT/train_logs"
state_path="$OUT_ROOT/early_stop_v1_02/state.json"
resume_state=""

for (( target_step=CHECKPOINT_INTERVAL; target_step<=TOTAL_STEPS; target_step+=CHECKPOINT_INTERVAL )); do
  checkpoint="checkpoints/${RUN_NAME}_$(printf '%06d' "$target_step").pth"
  trainstate="checkpoints/${RUN_NAME}_trainstate_$(printf '%06d' "$target_step").pth"
  train_log="$OUT_ROOT/train_logs/train_to_$(printf '%06d' "$target_step").log"
  reset_confidence_head=1
  if [[ -n "$resume_state" ]]; then
    reset_confidence_head=0
  fi

  if [[ -e "$checkpoint" || -e "$trainstate" ]]; then
    echo "refusing to reuse an existing stage output: $checkpoint or $trainstate" >&2
    exit 2
  fi

  echo "[STAGE] training through step $target_step / $TOTAL_STEPS"
  set +e
  BASE_CKPT="$BASE_CKPT" \
  CKPT="$BASE_CKPT" \
  RUN_NAME="$RUN_NAME" \
  STEPS="$TOTAL_STEPS" \
  STOP_AFTER_STEPS="$target_step" \
  SAVE_FREQ="$CHECKPOINT_INTERVAL" \
  SAVE_FINAL=0 \
  SAVE_TRAINING_STATE=1 \
  RESUME_TRAINING_STATE="$resume_state" \
  RESET_IMU_CONFIDENCE_HEAD="$reset_confidence_head" \
  TRAIN_SEQUENCES="$TRAIN_SEQUENCES" \
  EUROC_KEYFRAME_MANIFEST_ROOT="$MANIFEST_ROOT" \
  bash tools/run_euroc_15dof_confidence_only.sh |& tee "$train_log"
  train_status=${PIPESTATUS[0]}
  set -e
  if (( train_status != 0 )); then
    echo "training stage failed at step $target_step (exit=$train_status)" >&2
    exit "$train_status"
  fi
  if [[ ! -f "$checkpoint" || ! -f "$trainstate" ]]; then
    echo "training stage did not create expected checkpoint/state at step $target_step" >&2
    exit 2
  fi

  echo "[STAGE] V1_02 validation for $checkpoint"
  set +e
  EARLY_STOP_RUN_NAME="$RUN_NAME" \
  EARLY_STOP_VALIDATION_ROOT="$VALIDATION_ROOT" \
  EARLY_STOP_OUT_ROOT="$OUT_ROOT/early_stop_v1_02" \
  EARLY_STOP_STATE_PATH="$state_path" \
  EARLY_STOP_MIN_COVERAGE_RATIO="$MIN_COVERAGE_RATIO" \
  EARLY_STOP_DETERIORATION_TOLERANCE_M="$DETERIORATION_TOLERANCE_M" \
  EARLY_STOP_METRIC="$EARLY_STOP_METRIC" \
  bash tools/validate_15dof_confidence_checkpoint.sh "$checkpoint" |& tee "$OUT_ROOT/train_logs/validation_$(printf '%06d' "$target_step").log"
  validation_status=${PIPESTATUS[0]}
  set -e

  if (( validation_status == 42 )); then
    echo "[EARLY-STOP] validation declined at step $target_step; no later stage will start"
    echo "[EARLY-STOP] state: $state_path"
    exit 0
  fi
  if (( validation_status != 0 )); then
    echo "validation stage failed at step $target_step (exit=$validation_status)" >&2
    exit "$validation_status"
  fi

  resume_state="$trainstate"
done

echo "[DONE] all checkpoints validated through $TOTAL_STEPS"
echo "[DONE] early-stop history: $state_path"

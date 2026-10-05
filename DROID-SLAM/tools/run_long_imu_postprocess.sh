#!/usr/bin/env bash
set -euo pipefail

TRAIN_PID="${1:-573372}"
ROOT="/home/ubuntu/DROID-SLAM"
PYTHON="/home/ubuntu/miniconda3/envs/droid_env/bin/python"
DATA_ROOT="/home/ubuntu/SLAM/datasets/tartanair_v2"
TRAIN_NAME="full_imu_tartan_v2_long_20k"
EVAL_NAME="${TRAIN_NAME}_eval"
POST_LOG="${ROOT}/logs/${TRAIN_NAME}_postprocess.log"

cd "${ROOT}"
mkdir -p logs

{
  echo "[$(date '+%F %T')] waiting for training pid ${TRAIN_PID}"
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    sleep 60
  done
  echo "[$(date '+%F %T')] training pid ${TRAIN_PID} finished"

  FINAL_CKPT="checkpoints/${TRAIN_NAME}_final_020000.pth"
  if [[ ! -f "${FINAL_CKPT}" ]]; then
    FINAL_CKPT="$(find checkpoints -maxdepth 1 -type f -name "${TRAIN_NAME}_*.pth" | sort | tail -1)"
  fi

  if [[ -z "${FINAL_CKPT}" || ! -f "${FINAL_CKPT}" ]]; then
    echo "[$(date '+%F %T')] ERROR: no checkpoint found for ${TRAIN_NAME}"
    exit 1
  fi

  echo "[$(date '+%F %T')] evaluating ${FINAL_CKPT}"
  "${PYTHON}" train.py \
    --name "${EVAL_NAME}" \
    --ckpt "${FINAL_CKPT}" \
    --datasets tartan_v2 \
    --datapath "${DATA_ROOT}" \
    --gpus 1 \
    --steps 200 \
    --iters 5 \
    --lr 0.0001 \
    --eval_only \
    --use_imu_ba \
    --use_full_imu_ba \
    --use_imu_loss \
    --log_freq 10 \
    --save_freq 0 \
    --num_workers 0 \
    > "logs/${EVAL_NAME}.log" 2>&1

  echo "[$(date '+%F %T')] building comparison report"
  "${PYTHON}" tools/analyze_imu_experiments.py \
    --run baseline=baseline_droid_tartan_v2_eval_200 \
    --run rotation_only=rot_imu_tartan_v2_200_eval \
    --run full_200=full_imu_tartan_v2_200_eval \
    --run full_long="${EVAL_NAME}" \
    --output-md logs/imu_comparison_report.md \
    --output-csv logs/imu_comparison_summary.csv

  echo "[$(date '+%F %T')] done"
  echo "report: logs/imu_comparison_report.md"
  echo "csv: logs/imu_comparison_summary.csv"
} >> "${POST_LOG}" 2>&1

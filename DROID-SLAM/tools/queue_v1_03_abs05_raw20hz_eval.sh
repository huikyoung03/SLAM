#!/usr/bin/env bash
set -euo pipefail

# Run the final V1_03 comparison only after the active 2k training session has
# released the GPU. This intentionally creates a fresh stock-visual DROID
# reconstruction; legacy 10 Hz dense reconstructions are not reused.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
TRAIN_SESSION="${TRAIN_SESSION:-full15_abs05_2k}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/full15dof_onesidedquality_residualcal_abs05_mh03_mh04_2k_002000.pth}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/SLAM/datasets/euroc_heldout/prepared/V1_03_difficult_droid_cam0}"
OUT_DIR="${OUT_DIR:-outputs/v1_03_abs05_2k_raw20hz_dense}"
LOG="${LOG:-logs/v1_03_abs05_2k_raw20hz_dense.log}"

cd "$ROOT"
mkdir -p "$(dirname "$LOG")"

if [[ -e "$OUT_DIR" ]]; then
    echo "[QUEUE ERROR] refusing to overwrite existing output: $OUT_DIR" | tee -a "$LOG"
    exit 2
fi

echo "[QUEUE] waiting for training session $TRAIN_SESSION to finish: $(date -Is)" | tee -a "$LOG"
while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
    sleep 30
done

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[QUEUE ERROR] expected checkpoint was not created: $CHECKPOINT" | tee -a "$LOG"
    exit 1
fi

echo "[QUEUE] training finished; starting paired V1_03 raw-20Hz evaluation: $(date -Is)" | tee -a "$LOG"
DATA_ROOT="$DATA_ROOT" \
WEIGHTS="$CHECKPOINT" \
OUT_DIR="$OUT_DIR" \
STRIDE=2 \
TRAJECTORY_FILL_STRIDE=1 \
TRAJECTORY_SOURCE=dense \
CONFIDENCE_MODE=learned \
LEARNED_RESIDUAL_WEIGHTS=1 \
PYTHON="$PYTHON" \
bash tools/evaluate_original_vs_15dof_confidence.sh 2>&1 | tee -a "$LOG"

echo "[QUEUE] V1_03 evaluation completed: $(date -Is)" | tee -a "$LOG"

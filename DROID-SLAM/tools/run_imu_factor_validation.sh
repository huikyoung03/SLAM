#!/usr/bin/env bash
# Validate the physical 15DoF IMU factor before any new confidence training.
# This uses only MH03/MH04 development data; it never reads V1_03 or V2_03.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/SLAM/datasets/euroc_stage3}"
RUN_NAME="${RUN_NAME:-imu_factor_validation_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/${RUN_NAME}}"

MH03_SESSION="${MH03_SESSION:-${DATA_ROOT}/prepared_motion_filter/MH_03_medium_droid_cam0}"
MH04_SESSION="${MH04_SESSION:-${DATA_ROOT}/prepared_motion_filter/MH_04_difficult_droid_cam0}"
MH03_GT="${MH03_GT:-${DATA_ROOT}/raw/MH_03_medium/mav0/state_groundtruth_estimate0/data.csv}"
MH04_GT="${MH04_GT:-${DATA_ROOT}/raw/MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv}"

for path in "$PYTHON" "$MH03_SESSION/imu_prior_fd_jacobian.csv" "$MH04_SESSION/imu_prior_fd_jacobian.csv" "$MH03_GT" "$MH04_GT"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] required path does not exist: $path" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"
cd "$ROOT"

echo "[RUN] output=$OUT_DIR"
"$PYTHON" tools/test_full_imu_factor_jacobians.py \
  2>&1 | tee "$OUT_DIR/full_imu_factor_jacobian.log"

"$PYTHON" tools/audit_euroc_preintegration.py \
  --session "$MH03_SESSION" \
  --groundtruth "$MH03_GT" \
  --output-dir "$OUT_DIR/mh03" \
  2>&1 | tee "$OUT_DIR/mh03_preintegration_audit.log"

"$PYTHON" tools/audit_euroc_preintegration.py \
  --session "$MH04_SESSION" \
  --groundtruth "$MH04_GT" \
  --output-dir "$OUT_DIR/mh04" \
  2>&1 | tee "$OUT_DIR/mh04_preintegration_audit.log"

echo "[DONE] IMU factor validation passed; inspect $OUT_DIR"

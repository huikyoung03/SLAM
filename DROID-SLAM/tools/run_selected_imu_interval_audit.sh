#!/usr/bin/env bash
# Audit the exact IMU intervals that strict-protocol DROID MotionFilter chose.
# This is a CPU-only diagnostic on MH03/MH04 training data, not an ATE run.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/SLAM/datasets/euroc_stage3}"
RUN_NAME="${RUN_NAME:-selected_imu_interval_audit_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/${RUN_NAME}}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${DATA_ROOT}/motion_filter_manifests_original_protocol}"

MH03_SESSION="${MH03_SESSION:-${DATA_ROOT}/prepared_motion_filter/MH_03_medium_droid_cam0}"
MH04_SESSION="${MH04_SESSION:-${DATA_ROOT}/prepared_motion_filter/MH_04_difficult_droid_cam0}"
MH03_GT="${MH03_GT:-${DATA_ROOT}/raw/MH_03_medium/mav0/state_groundtruth_estimate0/data.csv}"
MH04_GT="${MH04_GT:-${DATA_ROOT}/raw/MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv}"
MH03_MANIFEST="${MH03_MANIFEST:-${MANIFEST_ROOT}/MH_03_medium.csv}"
MH04_MANIFEST="${MH04_MANIFEST:-${MANIFEST_ROOT}/MH_04_difficult.csv}"

for path in \
  "$PYTHON" \
  "$ROOT/tools/test_imu_preintegration_jacobians.py" \
  "$ROOT/tools/test_full_imu_factor_jacobians.py" \
  "$MH03_SESSION/imu_prior_fd_jacobian.csv" \
  "$MH04_SESSION/imu_prior_fd_jacobian.csv" \
  "$MH03_GT" "$MH04_GT" "$MH03_MANIFEST" "$MH04_MANIFEST"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] required path does not exist: $path" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"
cd "$ROOT"
echo "[RUN] output=$OUT_DIR"
echo "[RUN] manifests=$MH03_MANIFEST ; $MH04_MANIFEST"

"$PYTHON" tools/test_imu_preintegration_jacobians.py \
  2>&1 | tee "$OUT_DIR/imu_composition_jacobian.log"

"$PYTHON" tools/test_full_imu_factor_jacobians.py \
  2>&1 | tee "$OUT_DIR/full_imu_factor_jacobian.log"

"$PYTHON" tools/audit_euroc_preintegration.py \
  --session "$MH03_SESSION" \
  --groundtruth "$MH03_GT" \
  --selected-manifest "$MH03_MANIFEST" \
  --output-dir "$OUT_DIR/mh03" \
  2>&1 | tee "$OUT_DIR/mh03_selected_interval_audit.log"

"$PYTHON" tools/audit_euroc_preintegration.py \
  --session "$MH04_SESSION" \
  --groundtruth "$MH04_GT" \
  --selected-manifest "$MH04_MANIFEST" \
  --output-dir "$OUT_DIR/mh04" \
  2>&1 | tee "$OUT_DIR/mh04_selected_interval_audit.log"

echo "[DONE] selected-keyframe IMU interval audit passed; inspect $OUT_DIR"

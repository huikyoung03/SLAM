#!/usr/bin/env bash
set -euo pipefail

# Freeze visual MotionFilter selections under the exact original DROID EuRoC
# profile.  These CSVs are training inputs, not an IMU-assisted selection:
# droid.pth is deliberately used and no IMU options are supplied.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/SLAM/datasets/euroc_stage3/raw}"
PREPARED_ROOT="${PREPARED_ROOT:-/home/ubuntu/SLAM/datasets/euroc_stage3/prepared_motion_filter}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/home/ubuntu/SLAM/datasets/euroc_stage3/motion_filter_manifests_original_protocol}"
WEIGHTS="${WEIGHTS:-droid.pth}"
TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-MH_03_medium MH_04_difficult}"
# Frontend/backend are intentionally skipped while exporting manifests, so
# selected frames are never retired from DepthVideo. This capacity is only for
# that offline exporter; actual strict DROID/VI evaluations retain buffer=512.
MANIFEST_BUFFER="${MANIFEST_BUFFER:-4096}"

export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Checkpoint does not exist: $WEIGHTS" >&2
  exit 2
fi

read -r -a SEQUENCES <<< "$TRAIN_SEQUENCES"
if (( ${#SEQUENCES[@]} == 0 )); then
  echo "TRAIN_SEQUENCES must contain at least one EuRoC sequence" >&2
  exit 2
fi

mkdir -p "$PREPARED_ROOT" "$MANIFEST_ROOT"
for sequence in "${SEQUENCES[@]}"; do
  raw_mav0="$DATA_ROOT/$sequence/mav0"
  prepared="$PREPARED_ROOT/${sequence}_droid_cam0"
  manifest="$MANIFEST_ROOT/$sequence.csv"

  if [[ ! -f "$raw_mav0/cam0/data.csv" ]]; then
    echo "raw EuRoC input is missing: $raw_mav0/cam0/data.csv" >&2
    exit 2
  fi
  if [[ -e "$manifest" ]]; then
    echo "refusing to overwrite existing strict manifest: $manifest" >&2
    exit 2
  fi
  if [[ ! -d "$prepared" ]]; then
    "$PYTHON" tools/prepare_euroc_cam0.py --mav0 "$raw_mav0" --output "$prepared"
  fi
  for required in data calib.txt frames.csv; do
    if [[ ! -e "$prepared/$required" ]]; then
      echo "prepared input is incomplete: $prepared/$required" >&2
      exit 2
    fi
  done

  echo "[MOTION-FILTER / ORIGINAL-EUROC] $sequence -> $manifest"
  "$PYTHON" demo.py \
    --imagedir "$prepared/data" \
    --calib "$prepared/calib.txt" \
    --weights "$WEIGHTS" \
    --euroc_original_protocol \
    --buffer "$MANIFEST_BUFFER" \
    --disable_vis \
    --imu_prior "$prepared/imu_prior.csv" \
    --imu_frame_manifest "$prepared/frames.csv" \
    --motion_filter_manifest_only \
    --motion_filter_manifest_out "$manifest"
done

echo "[DONE] strict original-protocol manifests: $MANIFEST_ROOT"

#!/usr/bin/env bash
set -euo pipefail

# Freeze the exact *visual* MotionFilter selections used by the 15DoF DVI run.
#
# Candidate images are the normal EuRoC 10 Hz stream (raw 20 Hz cam0 with
# stride=2).  The CSV stores original raw frame IDs, so training can retain the
# true variable IMU interval between successive DROID-selected keyframes.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/SLAM/datasets/euroc_stage3/raw}"
PREPARED_ROOT="${PREPARED_ROOT:-/home/ubuntu/SLAM/datasets/euroc_stage3/prepared_motion_filter}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/home/ubuntu/SLAM/datasets/euroc_stage3/motion_filter_manifests_stride2}"
CKPT="${CKPT:-checkpoints/two_stage_15dof_conf_noise_6000_final_006000.pth}"
TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-MH_03_medium MH_04_difficult}"
STRIDE="${STRIDE:-2}"
BUFFER="${BUFFER:-4096}"

export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint does not exist: $CKPT" >&2
  exit 2
fi
if [[ "$STRIDE" != "2" ]]; then
  echo "This protocol is defined for the established EuRoC stride=2 candidate stream; got STRIDE=$STRIDE" >&2
  exit 2
fi

read -r -a SEQUENCES <<< "$TRAIN_SEQUENCES"
if (( ${#SEQUENCES[@]} == 0 )); then
  echo "TRAIN_SEQUENCES must contain at least one EuRoC sequence name" >&2
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
    echo "refusing to overwrite existing manifest: $manifest" >&2
    exit 2
  fi
  if [[ ! -d "$prepared" ]]; then
    echo "[PREPARE] $sequence"
    "$PYTHON" tools/prepare_euroc_cam0.py \
      --mav0 "$raw_mav0" \
      --output "$prepared"
  fi

  for required in data calib.txt frames.csv imu_prior.csv cam0_sensor.yaml imu0_sensor.yaml; do
    if [[ ! -e "$prepared/$required" ]]; then
      echo "prepared input is incomplete: $prepared/$required" >&2
      exit 2
    fi
  done

  echo "[MOTION-FILTER] $sequence -> $manifest"
  "$PYTHON" demo.py \
    --imagedir "$prepared/data" \
    --calib "$prepared/calib.txt" \
    --weights "$CKPT" \
    --stride "$STRIDE" \
    --buffer "$BUFFER" \
    --disable_vis \
    --imu_prior "$prepared/imu_prior.csv" \
    --imu_frame_manifest "$prepared/frames.csv" \
    --motion_filter_manifest_only \
    --motion_filter_manifest_out "$manifest"
done

echo "[DONE] frozen MotionFilter manifests: $MANIFEST_ROOT"

#!/usr/bin/env bash
set -euo pipefail

# Run final DVI four-way ablations sequentially. V2 starts only after V1 exits
# successfully. Supply either WEIGHTS directly or a selection.txt produced by
# select_dvi_stage3_checkpoint.sh; do not select on either final test set.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
WEIGHTS="${WEIGHTS:-}"
SELECTION_FILE="${SELECTION_FILE:-}"
V1_DATA_ROOT="${V1_DATA_ROOT:-/home/ubuntu/SLAM/datasets/euroc_heldout/prepared/V1_03_difficult_droid_cam0}"
V2_DATA_ROOT="${V2_DATA_ROOT:-/home/ubuntu/SLAM/datasets/euroc_heldout/prepared/V2_03_difficult_droid_cam0}"
STRIDE="${STRIDE:-2}"
MAX_FRAMES="${MAX_FRAMES:-0}"
TUNED_SCHEDULE="${TUNED_SCHEDULE:-0.25 0.50 0.75 1.00}"
OUT_ROOT="${OUT_ROOT:-outputs/euroc_dvi_imu_4way_test_pair_$(date +%Y%m%d_%H%M%S)}"

cd "$ROOT"
if [[ -z "$WEIGHTS" && -n "$SELECTION_FILE" ]]; then
  if [[ ! -f "$SELECTION_FILE" ]]; then
    echo "selection file does not exist: $SELECTION_FILE" >&2
    exit 2
  fi
  WEIGHTS="$(sed -n 's/^best_candidate_checkpoint=//p' "$SELECTION_FILE" | tail -n 1)"
fi
if [[ -z "$WEIGHTS" ]]; then
  echo "Set WEIGHTS or SELECTION_FILE before running final test sequences." >&2
  exit 2
fi
if [[ ! -f "$WEIGHTS" ]]; then
  echo "selected checkpoint does not exist: $WEIGHTS" >&2
  exit 2
fi
for path in "$V1_DATA_ROOT" "$V2_DATA_ROOT"; do
  if [[ ! -d "$path" ]]; then
    echo "test DATA_ROOT does not exist: $path" >&2
    exit 2
  fi
done

mkdir -p "$OUT_ROOT"
{
  echo "weights=$(realpath "$WEIGHTS")"
  echo "stride=$STRIDE"
  echo "max_frames=$MAX_FRAMES"
  echo "tuned_schedule=$TUNED_SCHEDULE"
  echo "v1_data_root=$V1_DATA_ROOT"
  echo "v2_data_root=$V2_DATA_ROOT"
  [[ -n "$SELECTION_FILE" ]] && echo "selection_file=$(realpath "$SELECTION_FILE")"
  git rev-parse HEAD
  git status --short
} > "$OUT_ROOT/run_manifest.txt"

run_sequence() {
  local label="$1"
  local data_root="$2"
  local output_dir="$OUT_ROOT/$label"
  mkdir -p "$output_dir"
  echo "[RUN] $label: $data_root"
  DATA_ROOT="$data_root" \
  WEIGHTS="$WEIGHTS" \
  STRIDE="$STRIDE" \
  MAX_FRAMES="$MAX_FRAMES" \
  TUNED_SCHEDULE="$TUNED_SCHEDULE" \
  OUT_DIR="$output_dir" \
  bash tools/run_euroc_dvi_imu_ablation.sh |& tee "$output_dir/run.log"
}

run_sequence v1_03_difficult "$V1_DATA_ROOT"
run_sequence v2_03_difficult "$V2_DATA_ROOT"

echo "[DONE] $OUT_ROOT"
echo "V1 report: $OUT_ROOT/v1_03_difficult/summary_common.md"
echo "V2 report: $OUT_ROOT/v2_03_difficult/summary_common.md"

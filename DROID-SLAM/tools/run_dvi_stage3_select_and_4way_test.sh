#!/usr/bin/env bash
set -euo pipefail

# One-command, leakage-free DVI experiment:
#   MH_03 training checkpoints
#       -> V1_02 + V2_02 validation checkpoint selection
#       -> V1_03 + V2_03 final feature-metric-free four-way evaluation.
#
# The final test sequences are never read until the checkpoint-selection file
# has been written.  This script performs work only when the user invokes it.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"

VAL_V1_RAW="${VAL_V1_RAW:-/home/ubuntu/SLAM/datasets/euroc_stage3/raw/V1_02_medium/mav0}"
VAL_V2_RAW="${VAL_V2_RAW:-/home/ubuntu/SLAM/datasets/euroc_stage3/raw/V2_02_medium/mav0}"
VAL_V1_ROOT="${VAL_V1_ROOT:-/home/ubuntu/SLAM/datasets/euroc_validation/prepared/V1_02_medium_droid_cam0}"
VAL_V2_ROOT="${VAL_V2_ROOT:-/home/ubuntu/SLAM/datasets/euroc_validation/prepared/V2_02_medium_droid_cam0}"
TEST_V1_ROOT="${TEST_V1_ROOT:-/home/ubuntu/SLAM/datasets/euroc_heldout/prepared/V1_03_difficult_droid_cam0}"
TEST_V2_ROOT="${TEST_V2_ROOT:-/home/ubuntu/SLAM/datasets/euroc_heldout/prepared/V2_03_difficult_droid_cam0}"

# Empty means the complete 1k...10k candidate series.  Use an explicit,
# whitespace-separated list only when deliberately reducing the search.
CHECKPOINTS="${CHECKPOINTS:-}"
STRIDE="${STRIDE:-2}"
MAX_FRAMES="${MAX_FRAMES:-0}"
IMU_BA_WEIGHT="${IMU_BA_WEIGHT:-0.002}"
IMU_CONFIDENCE_ONLY_WEIGHTING="${IMU_CONFIDENCE_ONLY_WEIGHTING:-0}"
DVI_IMU_QUALITY_FUSION="${DVI_IMU_QUALITY_FUSION:-0}"
MIN_COVERAGE_RATIO="${MIN_COVERAGE_RATIO:-0.95}"
TUNED_SCHEDULE="${TUNED_SCHEDULE:-0.25 0.50 0.75 1.00}"
RUN_FINAL_TEST="${RUN_FINAL_TEST:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/dvi_mh03_select_then_4way_$(date +%Y%m%d_%H%M%S)}"

cd "$ROOT"

ensure_fd_prior() {
  local data_root="$1"
  local prior="$data_root/imu_prior_fd_jacobian.csv"
  if [[ -s "$prior" ]]; then
    return
  fi
  for path in "$data_root/frames.csv" "$data_root/imu.csv" \
    "$data_root/cam0_sensor.yaml" "$data_root/imu0_sensor.yaml"; do
    if [[ ! -f "$path" ]]; then
      echo "cannot create IMU prior; required input does not exist: $path" >&2
      exit 2
    fi
  done
  echo "[PREPARE] finite-difference IMU prior: $prior"
  "$PYTHON" tools/imu_preintegrate.py \
    --frames "$data_root/frames.csv" \
    --imu "$data_root/imu.csv" \
    --output "$prior" \
    --cam_sensor_yaml "$data_root/cam0_sensor.yaml" \
    --imu_sensor_yaml "$data_root/imu0_sensor.yaml" \
    > "$data_root/imu_preintegrate_fd.log"
}

ensure_validation_root() {
  local raw_mav0="$1"
  local data_root="$2"
  if [[ ! -d "$data_root" ]]; then
    if [[ ! -d "$raw_mav0" ]]; then
      echo "validation input is missing: $data_root (and raw source is missing: $raw_mav0)" >&2
      exit 2
    fi
    echo "[PREPARE] validation input: $data_root"
    "$PYTHON" tools/prepare_euroc_cam0.py --mav0 "$raw_mav0" --output "$data_root"
  fi
  ensure_fd_prior "$data_root"
}

for data_root in "$TEST_V1_ROOT" "$TEST_V2_ROOT"; do
  if [[ ! -d "$data_root" ]]; then
    echo "final test input is missing: $data_root" >&2
    exit 2
  fi
done
for option in IMU_CONFIDENCE_ONLY_WEIGHTING DVI_IMU_QUALITY_FUSION RUN_FINAL_TEST; do
  value="${!option}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "$option must be 0 or 1, got: $value" >&2
    exit 2
  fi
done
if [[ "$IMU_CONFIDENCE_ONLY_WEIGHTING" == "1" && "$RUN_FINAL_TEST" == "1" ]]; then
  echo "confidence-only final four-way testing is not implemented in this legacy DVI runner; run checkpoint selection with RUN_FINAL_TEST=0 first" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"
{
  echo "protocol=MH_03 train -> V1_02,V2_02 select -> V1_03,V2_03 final test"
  echo "stride=$STRIDE"
  echo "max_frames=$MAX_FRAMES"
  echo "imu_ba_weight=$IMU_BA_WEIGHT"
  echo "imu_confidence_only_weighting=$IMU_CONFIDENCE_ONLY_WEIGHTING"
  echo "dvi_imu_quality_fusion=$DVI_IMU_QUALITY_FUSION"
  echo "run_final_test=$RUN_FINAL_TEST"
  echo "min_coverage_ratio=$MIN_COVERAGE_RATIO"
  echo "tuned_schedule=$TUNED_SCHEDULE"
  echo "val_v1_root=$VAL_V1_ROOT"
  echo "val_v2_root=$VAL_V2_ROOT"
  echo "test_v1_root=$TEST_V1_ROOT"
  echo "test_v2_root=$TEST_V2_ROOT"
  printf 'checkpoints=%s\n' "$CHECKPOINTS"
  git rev-parse HEAD
  git status --short
} > "$OUT_ROOT/run_manifest.txt"

ensure_validation_root "$VAL_V1_RAW" "$VAL_V1_ROOT"
ensure_validation_root "$VAL_V2_RAW" "$VAL_V2_ROOT"

run_validation() {
  local label="$1"
  local data_root="$2"
  local output_dir="$OUT_ROOT/validation/$label"
  mkdir -p "$output_dir"
  echo "[VALIDATION] $label"
  DATA_ROOT="$data_root" \
  CHECKPOINTS="$CHECKPOINTS" \
  STRIDE="$STRIDE" \
  MAX_FRAMES="$MAX_FRAMES" \
  IMU_BA_WEIGHT="$IMU_BA_WEIGHT" \
  IMU_CONFIDENCE_ONLY_WEIGHTING="$IMU_CONFIDENCE_ONLY_WEIGHTING" \
  DVI_IMU_QUALITY_FUSION="$DVI_IMU_QUALITY_FUSION" \
  MIN_COVERAGE_RATIO="$MIN_COVERAGE_RATIO" \
  OUT_DIR="$output_dir" \
  bash tools/select_dvi_stage3_checkpoint.sh |& tee "$output_dir/run.log"
}

# `set -e` means V2 cannot start unless V1 selection completed successfully.
run_validation v1_02 "$VAL_V1_ROOT"
run_validation v2_02 "$VAL_V2_ROOT"

SELECTION_FILE="$OUT_ROOT/checkpoint_selection.txt"
"$PYTHON" - \
  "$OUT_ROOT/validation/v1_02/summary_common.csv" \
  "$OUT_ROOT/validation/v1_02/summary_full.csv" \
  "$OUT_ROOT/validation/v1_02/candidate_map.csv" \
  "$OUT_ROOT/validation/v2_02/summary_common.csv" \
  "$OUT_ROOT/validation/v2_02/summary_full.csv" \
  "$SELECTION_FILE" \
  "$MIN_COVERAGE_RATIO" <<'PY'
import csv
import math
import sys

v1_common_path, v1_full_path, v1_map_path, v2_common_path, v2_full_path, output_path, coverage_ratio = sys.argv[1:]
coverage_ratio = float(coverage_ratio)

def read_rows(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def number(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan
    return value

def candidate_rows(common_path, full_path):
    common = {row['label']: row for row in read_rows(common_path) if row['label'].startswith('learned_')}
    full = {row['label']: row for row in read_rows(full_path) if row['label'].startswith('learned_')}
    if not common or not full:
        raise SystemExit(f'no learned candidate rows in {common_path}')
    max_coverage = max(number(row, 'raw_n_poses') for row in full.values())
    eligible = {
        label: common_row
        for label, common_row in common.items()
        if label in full
        and math.isfinite(number(common_row, 'se3_ate_rmse_m'))
        and number(full[label], 'raw_n_poses') >= coverage_ratio * max_coverage
    }
    if not eligible:
        raise SystemExit(f'no candidates meet the coverage guard in {common_path}')
    return eligible, full, max_coverage

mapping = {row['label']: row['checkpoint'] for row in read_rows(v1_map_path)}
v1, v1_full, v1_max_coverage = candidate_rows(v1_common_path, v1_full_path)
v2, v2_full, v2_max_coverage = candidate_rows(v2_common_path, v2_full_path)
labels = sorted(set(v1) & set(v2) & set(mapping))
if not labels:
    raise SystemExit('no candidate passed the metric and coverage guard on both validation sequences')

ranked = []
for label in labels:
    v1_ate = number(v1[label], 'se3_ate_rmse_m')
    v2_ate = number(v2[label], 'se3_ate_rmse_m')
    v1_rpe = number(v1[label], 'rpe_1s_trans_rmse_m')
    v2_rpe = number(v2[label], 'rpe_1s_trans_rmse_m')
    ranked.append((
        (v1_ate + v2_ate) / 2.0,
        (v1_rpe + v2_rpe) / 2.0,
        label,
        v1_ate,
        v2_ate,
        v1_rpe,
        v2_rpe,
    ))
ranked.sort()
best = ranked[0]

lines = [
    'selection_metric=mean of V1_02 and V2_02 common-frame metric SE(3) ATE RMSE',
    f'coverage_guard=raw_n_poses >= {coverage_ratio:.3f} of each sequence maximum',
    f'v1_02_max_raw_n_poses={v1_max_coverage:.0f}',
    f'v2_02_max_raw_n_poses={v2_max_coverage:.0f}',
    f'best_candidate_label={best[2]}',
    f'best_candidate_checkpoint={mapping[best[2]]}',
    f'best_candidate_mean_se3_ate_rmse_m={best[0]:.9f}',
    f'best_candidate_mean_rpe_1s_trans_rmse_m={best[1]:.9f}',
    f'best_candidate_v1_02_se3_ate_rmse_m={best[3]:.9f}',
    f'best_candidate_v2_02_se3_ate_rmse_m={best[4]:.9f}',
    f'best_candidate_v1_02_rpe_1s_trans_rmse_m={best[5]:.9f}',
    f'best_candidate_v2_02_rpe_1s_trans_rmse_m={best[6]:.9f}',
    'decision=checkpoint fixed before final V1_03 and V2_03 evaluation',
    'note=Sim(3) metrics were not used because they remove metric-scale error.',
]
lines.extend(
    f'rank_{index:02d}={label},mean_se3_ate_rmse_m={ate:.9f},mean_rpe_1s_trans_rmse_m={rpe:.9f}'
    for index, (ate, rpe, label, *_rest) in enumerate(ranked, start=1)
)
with open(output_path, 'w', encoding='utf-8') as handle:
    handle.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY

if [[ "$RUN_FINAL_TEST" == "0" ]]; then
  echo "[DONE] validation-only checkpoint selection: $OUT_ROOT"
  echo "checkpoint selection: $SELECTION_FILE"
  exit 0
fi

# This prior is absent in some older V1_03 prepared directories. It is created
# only after selection is fixed and before that final sequence is evaluated.
ensure_fd_prior "$TEST_V1_ROOT"
ensure_fd_prior "$TEST_V2_ROOT"

echo "[FINAL-TEST] selected checkpoint: $(sed -n 's/^best_candidate_checkpoint=//p' "$SELECTION_FILE")"
mkdir -p "$OUT_ROOT/final_4way"
SELECTION_FILE="$SELECTION_FILE" \
STRIDE="$STRIDE" \
MAX_FRAMES="$MAX_FRAMES" \
TUNED_SCHEDULE="$TUNED_SCHEDULE" \
V1_DATA_ROOT="$TEST_V1_ROOT" \
V2_DATA_ROOT="$TEST_V2_ROOT" \
OUT_ROOT="$OUT_ROOT/final_4way" \
bash tools/run_euroc_dvi_imu_4way_test_pair.sh |& tee "$OUT_ROOT/final_4way/run.log"

echo "[DONE] $OUT_ROOT"
echo "checkpoint selection: $SELECTION_FILE"
echo "V1_03 four-way report: $OUT_ROOT/final_4way/v1_03_difficult/summary_common.md"
echo "V2_03 four-way report: $OUT_ROOT/final_4way/v2_03_difficult/summary_common.md"

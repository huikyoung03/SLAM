#!/usr/bin/env bash
set -euo pipefail

# Select the DVI scalar-confidence checkpoint on a validation sequence only.
# Do not point DATA_ROOT at the V1_03/V2_03 reporting sets: those are the
# final test sequences and must remain unseen during selection.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to one prepared validation EuRoC sequence (V1_02 or V2_02).}"
IMAGEDIR="${IMAGEDIR:-${DATA_ROOT}/data}"
CALIB="${CALIB:-${DATA_ROOT}/calib.txt}"
IMU_PRIOR="${IMU_PRIOR:-${DATA_ROOT}/imu_prior_fd_jacobian.csv}"
FRAMES_CSV="${FRAMES_CSV:-${DATA_ROOT}/frames.csv}"
CAMERA_SENSOR_YAML="${CAMERA_SENSOR_YAML:-${DATA_ROOT}/cam0_sensor.yaml}"
IMU_SENSOR_YAML="${IMU_SENSOR_YAML:-${DATA_ROOT}/imu0_sensor.yaml}"
GT_CSV="${GT_CSV:-}"

# A whitespace-separated explicit list overrides the default 1k...10k series.
CHECKPOINTS="${CHECKPOINTS:-}"
STRIDE="${STRIDE:-2}"
MAX_FRAMES="${MAX_FRAMES:-0}"
BUFFER="${BUFFER:-4096}"
IMU_BA_WEIGHT="${IMU_BA_WEIGHT:-0.002}"
# New reliability formulation: the global IMU factor is fixed at one and the
# learned DVI confidence is the only adaptive attenuation.  Keep this opt-in
# so historical lambda-based selection reports remain reproducible.
IMU_CONFIDENCE_ONLY_WEIGHTING="${IMU_CONFIDENCE_ONLY_WEIGHTING:-0}"
# Opt-in so historical scalar-DVI evaluations remain reproducible. A
# quality-fusion checkpoint must be evaluated with this set to one.
DVI_IMU_QUALITY_FUSION="${DVI_IMU_QUALITY_FUSION:-0}"
# Candidates which track substantially less of the validation stream cannot
# win merely because their common trajectory is short and easy.
MIN_COVERAGE_RATIO="${MIN_COVERAGE_RATIO:-0.95}"
OUT_DIR="${OUT_DIR:-outputs/dvi_mh03_stage3_checkpoint_selection_$(basename "${DATA_ROOT}")_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "$GT_CSV" && -f "$DATA_ROOT/source.txt" ]]; then
  GT_CSV="$(sed -n 's/^groundtruth=//p' "$DATA_ROOT/source.txt" | head -n 1)"
fi
if [[ -z "$GT_CSV" && -f "$DATA_ROOT/source.txt" ]]; then
  SOURCE_MAV0="$(sed -n 's/^source_mav0=//p' "$DATA_ROOT/source.txt" | head -n 1)"
  if [[ -n "$SOURCE_MAV0" ]]; then
    GT_CSV="$SOURCE_MAV0/state_groundtruth_estimate0/data.csv"
  fi
fi

cd "$ROOT"
for path in "$IMAGEDIR" "$CALIB" "$IMU_PRIOR" "$FRAMES_CSV" \
  "$CAMERA_SENSOR_YAML" "$IMU_SENSOR_YAML" "$GT_CSV"; do
  if [[ ! -e "$path" ]]; then
    echo "required input does not exist: $path" >&2
    exit 2
  fi
done

if [[ -z "$CHECKPOINTS" ]]; then
  shopt -s nullglob
  CHECKPOINT_LIST=(checkpoints/dvi_mh03_10fps_stage3_10000_[0-9][0-9][0-9][0-9][0-9][0-9].pth)
  shopt -u nullglob
else
  # Intentional word splitting: CHECKPOINTS is a whitespace-separated list.
  read -r -a CHECKPOINT_LIST <<< "$CHECKPOINTS"
fi
for option in IMU_CONFIDENCE_ONLY_WEIGHTING DVI_IMU_QUALITY_FUSION; do
  value="${!option}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "$option must be 0 or 1, got: $value" >&2
    exit 2
  fi
done
if (( ${#CHECKPOINT_LIST[@]} == 0 )); then
  echo "No candidate DVI stage-3 checkpoints found." >&2
  exit 2
fi
for checkpoint in "${CHECKPOINT_LIST[@]}"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "candidate checkpoint does not exist: $checkpoint" >&2
    exit 2
  fi
done

mkdir -p "$OUT_DIR"
{
  echo "purpose=validation-only DVI stage-3 checkpoint selection"
  echo "data_root=$DATA_ROOT"
  echo "stride=$STRIDE"
  echo "max_frames=$MAX_FRAMES"
  echo "imu_ba_weight=$IMU_BA_WEIGHT"
  echo "imu_confidence_only_weighting=$IMU_CONFIDENCE_ONLY_WEIGHTING"
  echo "dvi_imu_quality_fusion=$DVI_IMU_QUALITY_FUSION"
  echo "min_coverage_ratio=$MIN_COVERAGE_RATIO"
  printf 'checkpoints='
  printf '%s ' "${CHECKPOINT_LIST[@]}"
  printf '\n'
  git rev-parse HEAD
  git status --short
} > "$OUT_DIR/run_manifest.txt"

MAX_ARGS=()
if [[ "$MAX_FRAMES" != "0" ]]; then
  MAX_ARGS=(--max_frames "$MAX_FRAMES")
fi

IMU_WEIGHT_ARGS=(--imu_ba_prior_weight "$IMU_BA_WEIGHT")
if [[ "$IMU_CONFIDENCE_ONLY_WEIGHTING" == "1" ]]; then
  IMU_WEIGHT_ARGS=(--imu_confidence_only_weighting)
fi
DVI_QUALITY_ARGS=()
if [[ "$DVI_IMU_QUALITY_FUSION" == "1" ]]; then
  DVI_QUALITY_ARGS=(--use_dvi_imu_quality_fusion)
fi

printf 'label,checkpoint\n' > "$OUT_DIR/candidate_map.csv"
RUN_ARGS=()
for checkpoint in "${CHECKPOINT_LIST[@]}"; do
  tag="$(basename "$checkpoint" .pth)"
  label="learned_${tag##*_}"
  recon="$OUT_DIR/${label}_recon.pth"
  debug="$OUT_DIR/${label}_debug.csv"

  echo "[RUN] $label ($checkpoint)"
  "$PYTHON" demo.py \
    --imagedir "$IMAGEDIR" \
    --calib "$CALIB" \
    --weights "$checkpoint" \
    --stride "$STRIDE" \
    "${MAX_ARGS[@]}" \
    --buffer "$BUFFER" \
    --disable_vis \
    --reconstruction_path "$recon" \
    --imu_prior "$IMU_PRIOR" \
    --imu_frame_manifest "$FRAMES_CSV" \
    --cam_sensor_yaml "$CAMERA_SENSOR_YAML" \
    --imu_sensor_yaml "$IMU_SENSOR_YAML" \
    --use_two_stage_imu_ba \
    --use_imu_noise_model \
    "${IMU_WEIGHT_ARGS[@]}" \
    --use_learned_imu_confidence \
    --use_dvi_imu_confidence \
    "${DVI_QUALITY_ARGS[@]}" \
    --imu_confidence_floor 0 \
    --imu_ba_debug \
    --imu_ba_debug_path "$debug"

  printf '%s,%s\n' "$label" "$(realpath "$checkpoint")" >> "$OUT_DIR/candidate_map.csv"
  RUN_ARGS+=(--run "$label=$recon,$debug")
done

"$PYTHON" tools/evaluate_euroc_recon.py \
  --frames-csv "$FRAMES_CSV" \
  --gt-csv "$GT_CSV" \
  --gt-frame body \
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML" \
  --common-frame-ids \
  --output-csv "$OUT_DIR/summary_common.csv" \
  --output-md "$OUT_DIR/summary_common.md" \
  "${RUN_ARGS[@]}"

"$PYTHON" tools/evaluate_euroc_recon.py \
  --frames-csv "$FRAMES_CSV" \
  --gt-csv "$GT_CSV" \
  --gt-frame body \
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML" \
  --output-csv "$OUT_DIR/summary_full.csv" \
  --output-md "$OUT_DIR/summary_full.md" \
  "${RUN_ARGS[@]}"

"$PYTHON" - "$OUT_DIR/summary_common.csv" "$OUT_DIR/summary_full.csv" \
  "$OUT_DIR/candidate_map.csv" "$OUT_DIR/selection.txt" "$MIN_COVERAGE_RATIO" <<'PY'
import csv
import math
import sys

common_path, full_path, mapping_path, output_path, coverage_ratio = sys.argv[1:]
coverage_ratio = float(coverage_ratio)

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def value(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan

mapping = {row['label']: row['checkpoint'] for row in read_csv(mapping_path)}
common = {row['label']: row for row in read_csv(common_path) if row['label'] in mapping}
full = {row['label']: row for row in read_csv(full_path) if row['label'] in mapping}
if not common or not full:
    raise SystemExit('selection failed: no evaluated DVI candidates')

max_coverage = max(value(row, 'raw_n_poses') for row in full.values())
minimum_coverage = coverage_ratio * max_coverage
candidates = []
for label, row in common.items():
    full_row = full.get(label, {})
    if not math.isfinite(value(row, 'se3_ate_rmse_m')):
        continue
    if value(full_row, 'raw_n_poses') < minimum_coverage:
        continue
    candidates.append((label, row, full_row))
if not candidates:
    raise SystemExit('selection failed: no metricized candidate met the coverage guard')

# The learned head supplies metric IMU information. Select with common-frame
# metric SE(3) ATE, not Sim(3), whose fitted scale masks VIO scale errors.
best_label, best_common, best_full = min(
    candidates,
    key=lambda item: (
        value(item[1], 'se3_ate_rmse_m'),
        value(item[1], 'rpe_1s_trans_rmse_m'),
        value(item[1], 'rpe_5s_trans_rmse_m'),
    ),
)

lines = [
    'selection_metric=common-frame metric SE(3) ATE RMSE',
    f'coverage_guard=raw_n_poses >= {coverage_ratio:.3f} * {max_coverage:.0f}',
    f'best_candidate_label={best_label}',
    f'best_candidate_checkpoint={mapping[best_label]}',
    f'best_candidate_common_se3_ate_rmse_m={value(best_common, "se3_ate_rmse_m"):.9f}',
    f'best_candidate_common_rpe_1s_trans_rmse_m={value(best_common, "rpe_1s_trans_rmse_m"):.9f}',
    f'best_candidate_common_rpe_5s_trans_rmse_m={value(best_common, "rpe_5s_trans_rmse_m"):.9f}',
    f'best_candidate_full_raw_n_poses={value(best_full, "raw_n_poses"):.0f}',
    'decision=checkpoint fixed before V1_03/V2_03 final evaluation',
    'note=The V1_03/V2_03 reporting sequences must not be used for this selection.',
]
with open(output_path, 'w', encoding='utf-8') as handle:
    handle.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY

echo "[DONE] $OUT_DIR"
echo "selection: $OUT_DIR/selection.txt"

#!/usr/bin/env bash
set -euo pipefail

# Rank frozen Hospital fine-tuning checkpoints on one EuRoC validation sequence.
# This script deliberately does not alter any runtime IMU weights or thresholds:
# every candidate uses the same fixed-bias 9DoF/noise-model configuration.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"

DATA_ROOT="${DATA_ROOT:-/home/ubuntu/SLAM/datasets/euroc_heldout/prepared/V2_01_easy_droid_cam0}"
ORIGINAL_WEIGHTS="${ORIGINAL_WEIGHTS:-droid.pth}"
# Be explicit about the candidate family. The old Hospital checkpoints predate
# the axis/state fixes, so silently selecting them would mix experiments.
CHECKPOINTS="${CHECKPOINTS:-}"
# Full-sequence validation is the default. Set MAX_FRAMES=450 only for a
# quick smoke test; a short prefix must not select the final checkpoint.
MAX_FRAMES="${MAX_FRAMES:-0}"
STRIDE="${STRIDE:-2}"
BUFFER="${BUFFER:-4096}"
RUN_PLY="${RUN_PLY:-0}"
OUT_DIR="${OUT_DIR:-outputs/euroc_checkpoint_selection_$(basename "${DATA_ROOT}")_$(date +%Y%m%d_%H%M%S)}"

IMAGEDIR="${IMAGEDIR:-${DATA_ROOT}/data}"
CALIB="${CALIB:-${DATA_ROOT}/calib.txt}"
IMU_PRIOR="${IMU_PRIOR:-${DATA_ROOT}/imu_prior.csv}"
FRAMES_CSV="${FRAMES_CSV:-${DATA_ROOT}/frames.csv}"
CAMERA_SENSOR_YAML="${CAMERA_SENSOR_YAML:-${DATA_ROOT}/cam0_sensor.yaml}"
IMU_SENSOR_YAML="${IMU_SENSOR_YAML:-${DATA_ROOT}/imu0_sensor.yaml}"
GT_CSV="${GT_CSV:-}"
if [[ -z "$GT_CSV" && -f "${DATA_ROOT}/source.txt" ]]; then
  GT_CSV="$(sed -n 's/^groundtruth=//p' "${DATA_ROOT}/source.txt" | head -n 1)"
fi

cd "$ROOT"

if [[ -z "$CHECKPOINTS" ]]; then
  echo "Set CHECKPOINTS to the frozen 9DoF checkpoints to rank." >&2
  exit 2
fi

for path in "$IMAGEDIR" "$CALIB" "$IMU_PRIOR" "$FRAMES_CSV" "$CAMERA_SENSOR_YAML" "$IMU_SENSOR_YAML" "$GT_CSV" "$ORIGINAL_WEIGHTS"; do
  if [[ ! -e "$path" ]]; then
    echo "required input does not exist: $path" >&2
    exit 2
  fi
done

mkdir -p "$OUT_DIR"

MAX_ARGS=()
if [[ "$MAX_FRAMES" != "0" ]]; then
  MAX_ARGS=(--max_frames "$MAX_FRAMES")
fi

RUN_ARGS=()
run_baseline() {
  local recon="$OUT_DIR/original_droid_recon.pth"
  echo "[RUN] original_droid"
  "$PYTHON" demo.py \
    --imagedir "$IMAGEDIR" \
    --calib "$CALIB" \
    --weights "$ORIGINAL_WEIGHTS" \
    --stride "$STRIDE" \
    "${MAX_ARGS[@]}" \
    --buffer "$BUFFER" \
    --disable_vis \
    --reconstruction_path "$recon"

  if [[ "$RUN_PLY" == "1" ]]; then
    "$PYTHON" export_reconstruction_ply.py "$recon" --output "$OUT_DIR/original_droid.ply"
  fi
  RUN_ARGS+=(--run "original_droid=${recon}")
}

run_imu_candidate() {
  local checkpoint="$1"
  local tag
  tag="$(basename "$checkpoint" .pth)"
  local label="imu_${tag}"
  local recon="$OUT_DIR/${label}_recon.pth"
  local debug="$OUT_DIR/${label}_debug.csv"

  if [[ ! -f "$checkpoint" ]]; then
    echo "candidate checkpoint does not exist: $checkpoint" >&2
    exit 2
  fi

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
    --cam_sensor_yaml "$CAMERA_SENSOR_YAML" \
    --imu_sensor_yaml "$IMU_SENSOR_YAML" \
    --use_pose_velocity_imu_ba \
    --imu_full_frontend \
    --use_learned_imu_confidence \
    --use_learned_imu_ba_weight \
    --use_learned_imu_residual_weights \
    --use_imu_noise_model \
    --use_vi_initialization \
    --imu_ba_debug \
    --imu_ba_debug_path "$debug"

  if [[ "$RUN_PLY" == "1" ]]; then
    "$PYTHON" export_reconstruction_ply.py "$recon" --output "$OUT_DIR/${label}.ply"
  fi

  printf '%s,%s\n' "$label" "$checkpoint" >> "$OUT_DIR/candidate_map.csv"
  RUN_ARGS+=(--run "${label}=${recon},${debug}")
}

printf 'label,checkpoint\n' > "$OUT_DIR/candidate_map.csv"
run_baseline

read -r -a CHECKPOINT_LIST <<< "$CHECKPOINTS"
if [[ "${#CHECKPOINT_LIST[@]}" -eq 0 ]]; then
  echo "CHECKPOINTS is empty" >&2
  exit 2
fi
for checkpoint in "${CHECKPOINT_LIST[@]}"; do
  run_imu_candidate "$checkpoint"
done

"$PYTHON" tools/evaluate_euroc_recon.py \
  --frames-csv "$FRAMES_CSV" \
  --gt-csv "$GT_CSV" \
  --gt-frame body \
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML" \
  --common-frame-ids \
  --output-csv "$OUT_DIR/summary.csv" \
  --output-md "$OUT_DIR/summary.md" \
  "${RUN_ARGS[@]}"

"$PYTHON" - "$OUT_DIR/summary.csv" "$OUT_DIR/candidate_map.csv" "$OUT_DIR/selection.txt" <<'PY'
import csv
import math
import sys

summary_path, mapping_path, output_path = sys.argv[1:]
with open(mapping_path, newline='', encoding='utf-8') as handle:
    mapping = {row['label']: row['checkpoint'] for row in csv.DictReader(handle)}
with open(summary_path, newline='', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))

def value(row, key):
    try:
        result = float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan
    return result

baseline = next((row for row in rows if row['label'] == 'original_droid'), None)
candidates = [
    row for row in rows
    if row['label'] in mapping
    and value(row, 'metric_scale_available') == 1.0
    and math.isfinite(value(row, 'se3_ate_rmse_m'))
]
if not candidates:
    raise SystemExit('selection failed: no metricized IMU candidate has finite SE(3) ATE')

# Checkpoints are selected by the metric VI result, not by a post-hoc Sim(3)
# fit that erases the scale information supplied by the IMU.
best = min(
    candidates,
    key=lambda row: (
        value(row, 'se3_ate_rmse_m'),
        value(row, 'rpe_1s_trans_rmse_m'),
    ),
)
baseline_shape_ate = value(baseline, 'sim3_ate_rmse_m') if baseline else math.nan
best_shape_ate = value(best, 'sim3_ate_rmse_m')
shape_relative = (
    100.0 * (best_shape_ate - baseline_shape_ate) / baseline_shape_ate
    if math.isfinite(baseline_shape_ate) and baseline_shape_ate > 0.0
    else math.nan
)

lines = [
    'selection_metric=common-frame metric SE(3) ATE RMSE (IMU candidates only)',
    f'baseline_sim3_ate_rmse_m={baseline_shape_ate:.9f}',
    f'best_candidate_label={best["label"]}',
    f'best_candidate_checkpoint={mapping[best["label"]]}',
    f'best_candidate_metric_se3_ate_rmse_m={value(best, "se3_ate_rmse_m"):.9f}',
    f'best_candidate_metric_origin_ate_rmse_m={value(best, "origin_ate_rmse_m"):.9f}',
    f'best_candidate_metric_rpe_1s_trans_rmse_m={value(best, "rpe_1s_trans_rmse_m"):.9f}',
    f'best_candidate_sim3_ate_rmse_m={best_shape_ate:.9f}',
    f'shape_relative_change_vs_baseline_percent={shape_relative:.4f}',
    'decision=ranking_only',
    'cross_sequence_validation_required=true',
    'note=Visual-only DROID has unobservable metric scale, so it is compared only by Sim(3) shape metrics.',
    'note=Candidate ranking is fixed before cross-sequence evaluation; no runtime weight or threshold was tuned here.',
]
with open(output_path, 'w', encoding='utf-8') as handle:
    handle.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY

echo "[DONE] $OUT_DIR"

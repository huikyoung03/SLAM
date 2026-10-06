#!/usr/bin/env bash
set -euo pipefail

# Evaluate stock DROID plus matched visual-only, fixed-bias 9DoF, and optional
# low-rate-bias variants on one prepared EuRoC sequence. This script never
# trains or tunes on the sequence it evaluates.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to a directory created by prepare_euroc_cam0.py}"
GT_CSV="${GT_CSV:-}"
if [[ -z "$GT_CSV" && -f "$DATA_ROOT/source.txt" ]]; then
  GT_CSV="$(sed -n 's/^groundtruth=//p' "$DATA_ROOT/source.txt" | head -n 1)"
fi
# Some older prepared folders were created before `groundtruth=` was written
# to source.txt.  They still retain the original mav0 root, so recover the
# canonical EuRoC ground-truth path without making the evaluation depend on a
# hand-entered host path.
if [[ -z "$GT_CSV" && -f "$DATA_ROOT/source.txt" ]]; then
  SOURCE_MAV0="$(sed -n 's/^source_mav0=//p' "$DATA_ROOT/source.txt" | head -n 1)"
  if [[ -n "$SOURCE_MAV0" ]]; then
    GT_CSV="$SOURCE_MAV0/state_groundtruth_estimate0/data.csv"
  fi
fi
IMAGEDIR="${IMAGEDIR:-${DATA_ROOT}/data}"
CALIB="${CALIB:-${DATA_ROOT}/calib.txt}"
IMU_PRIOR="${IMU_PRIOR:-${DATA_ROOT}/imu_prior.csv}"
FRAMES_CSV="${FRAMES_CSV:-${DATA_ROOT}/frames.csv}"
CAMERA_SENSOR_YAML="${CAMERA_SENSOR_YAML:-${DATA_ROOT}/cam0_sensor.yaml}"
IMU_SENSOR_YAML="${IMU_SENSOR_YAML:-${DATA_ROOT}/imu0_sensor.yaml}"

ORIGINAL_WEIGHTS="${ORIGINAL_WEIGHTS:-droid.pth}"
# Existing reconstructions may be reused only when they were produced with the
# same prepared sequence, stride and buffer. This avoids rerunning visual-only
# DROID or the frozen fixed-bias control when evaluating a new bias variant.
ORIGINAL_RECON="${ORIGINAL_RECON:-}"
FIXED_RECON="${FIXED_RECON:-}"
MATCHED_VISUAL_ONLY_RECON="${MATCHED_VISUAL_ONLY_RECON:-}"
LOW_RATE_BIAS_RECON="${LOW_RATE_BIAS_RECON:-}"
LOW_RATE_BIAS_DEBUG="${LOW_RATE_BIAS_DEBUG:-}"
# Older tensor bundles did not record their source sequence. Reusing one is
# disabled by default because a stale V2 reconstruction must never be mixed
# into a V1 held-out report. Set this only after manual provenance review.
ALLOW_LEGACY_RECON_REUSE="${ALLOW_LEGACY_RECON_REUSE:-0}"
RUN_MATCHED_VISUAL_ONLY="${RUN_MATCHED_VISUAL_ONLY:-1}"
# Pass the frozen checkpoint explicitly. A stale default can silently
# evaluate a different IMU formulation from the one selected on V2_01.
IMU_WEIGHTS="${IMU_WEIGHTS:-}"
# Full-sequence validation is the default. MAX_FRAMES=450 is only a smoke test.
MAX_FRAMES="${MAX_FRAMES:-0}"
STRIDE="${STRIDE:-2}"
BUFFER="${BUFFER:-4096}"
TURN_FRAME="${TURN_FRAME:-}"
RUN_PLY="${RUN_PLY:-0}"
USE_LEGACY_ADAPTIVE_GATE="${USE_LEGACY_ADAPTIVE_GATE:-0}"
USE_NOISE_MODEL="${USE_NOISE_MODEL:-1}"
USE_LEARNED_IMU_RESIDUAL_WEIGHTS="${USE_LEARNED_IMU_RESIDUAL_WEIGHTS:-1}"
RUN_SHARED_BIAS="${RUN_SHARED_BIAS:-1}"
BIAS_REFINE_INTERVAL="${BIAS_REFINE_INTERVAL:-20}"
BIAS_REFINE_WINDOW="${BIAS_REFINE_WINDOW:-30}"
BIAS_REFINE_ROBUST_DELTA="${BIAS_REFINE_ROBUST_DELTA:-3.0}"
BIAS_REFINE_MIN_EDGES="${BIAS_REFINE_MIN_EDGES:-8}"
BIAS_REFINE_MIN_DURATION="${BIAS_REFINE_MIN_DURATION:-0.5}"
BIAS_REFINE_MIN_ROTATION_DEG="${BIAS_REFINE_MIN_ROTATION_DEG:-2.0}"
BIAS_REFINE_MAX_CONDITION="${BIAS_REFINE_MAX_CONDITION:-1e8}"
BIAS_REFINE_MIN_RELATIVE_IMPROVEMENT="${BIAS_REFINE_MIN_RELATIVE_IMPROVEMENT:-1e-7}"
BIAS_REFINE_BACKEND="${BIAS_REFINE_BACKEND:-0}"
VI_INIT_SNAPSHOT_HISTORY_SIZE="${VI_INIT_SNAPSHOT_HISTORY_SIZE:-24}"
VI_INIT_SNAPSHOT_STRIDE="${VI_INIT_SNAPSHOT_STRIDE:-5}"
VI_INIT_SNAPSHOT_MAX_CANDIDATES="${VI_INIT_SNAPSHOT_MAX_CANDIDATES:-6}"
# A short recent-only window can become sparse after MotionFilter removes
# redundant frames. This is an initialization observation horizon, not a
# relaxation of scale/bias/RMS acceptance thresholds.
VI_INIT_WINDOW="${VI_INIT_WINDOW:-90}"
VI_INIT_BOOTSTRAP_FRAMES="${VI_INIT_BOOTSTRAP_FRAMES:-60}"
# Final held-out results must reflect the online frontend path.  The backend
# fallback remains available for diagnosis, but must be explicitly requested.
VI_INIT_BACKEND_FALLBACK="${VI_INIT_BACKEND_FALLBACK:-0}"
VI_INIT_BACKEND_WINDOW="${VI_INIT_BACKEND_WINDOW:-120}"
OUT_DIR="${OUT_DIR:-outputs/euroc_heldout_$(basename "${DATA_ROOT}")_$(date +%Y%m%d_%H%M%S)}"

cd "$ROOT"
mkdir -p "$OUT_DIR"

if [[ -z "$IMU_WEIGHTS" ]]; then
  echo "Set IMU_WEIGHTS to the frozen 9DoF checkpoint to validate." >&2
  exit 2
fi

for path in "$GT_CSV" "$CAMERA_SENSOR_YAML" "$IMU_SENSOR_YAML"; do
  if [[ ! -f "$path" ]]; then
    echo "required evaluation input does not exist: $path" >&2
    exit 2
  fi
done

MAX_ARGS=()
if [[ "$MAX_FRAMES" != "0" ]]; then
  MAX_ARGS=(--max_frames "$MAX_FRAMES")
fi

RUN_ARGS=()

validate_reconstruction_reuse() {
  local label="$1"
  local recon="$2"
  local require_imu="$3"

  "$PYTHON" - "$label" "$recon" "$IMAGEDIR" "$CALIB" "$IMU_PRIOR" \
    "$FRAMES_CSV" "$CAMERA_SENSOR_YAML" "$IMU_SENSOR_YAML" "$STRIDE" \
    "$MAX_FRAMES" "$require_imu" "$ALLOW_LEGACY_RECON_REUSE" <<'PY'
import hashlib
import os
import sys

import torch


def normalized_max_frames(value):
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def digest(path):
    path = os.path.realpath(os.path.expanduser(path))
    if not os.path.isfile(path):
        return None
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


(
    label,
    recon,
    image_dir,
    calib,
    imu_prior,
    frames_csv,
    camera_sensor_yaml,
    imu_sensor_yaml,
    stride,
    max_frames,
    require_imu,
    allow_legacy,
) = sys.argv[1:]
blob = torch.load(recon, map_location="cpu")
provenance = blob.get("input_provenance")
if not isinstance(provenance, dict):
    if allow_legacy == "1":
        print(
            f"[REUSE WARNING] {label}: accepting legacy reconstruction without "
            "input provenance because ALLOW_LEGACY_RECON_REUSE=1",
            file=sys.stderr,
        )
        raise SystemExit(0)
    print(
        f"[REUSE ERROR] {label}: reconstruction has no input_provenance. "
        "Regenerate it, or set ALLOW_LEGACY_RECON_REUSE=1 only after manual review.",
        file=sys.stderr,
    )
    raise SystemExit(2)

expected = {
    "image_directory": os.path.realpath(os.path.expanduser(image_dir)),
    "calibration_sha256": digest(calib),
    "stride": int(stride),
    "max_frames": normalized_max_frames(max_frames),
}
errors = []
for key, expected_value in expected.items():
    actual_value = provenance.get(key)
    if actual_value != expected_value:
        errors.append(f"{key}: expected={expected_value!r}, got={actual_value!r}")

if require_imu == "1":
    expected_imu_inputs = {
        "imu_prior_sha256": digest(imu_prior),
        "frames_csv_sha256": digest(frames_csv),
        "cam_sensor_yaml_sha256": digest(camera_sensor_yaml),
        "imu_sensor_yaml_sha256": digest(imu_sensor_yaml),
    }
    for key, expected_value in expected_imu_inputs.items():
        actual_value = provenance.get(key)
        if actual_value is None and allow_legacy == "1":
            print(
                f"[REUSE WARNING] {label}: accepting legacy IMU reconstruction "
                f"without {key} because ALLOW_LEGACY_RECON_REUSE=1",
                file=sys.stderr,
            )
            continue
        if actual_value != expected_value:
            errors.append(
                f"{key}: expected current IMU input, got={actual_value!r}"
            )

if errors:
    print(
        f"[REUSE ERROR] {label}: reconstruction does not match the active held-out input:\n"
        + "\n".join(f"  - {error}" for error in errors),
        file=sys.stderr,
    )
    raise SystemExit(2)

print(f"[REUSE] {label}: provenance matches current sequence", file=sys.stderr)
PY
}

run_case() {
  local label="$1"
  local weights="$2"
  shift 2
  local recon="$OUT_DIR/${label}_recon.pth"
  local debug="$OUT_DIR/${label}_debug.csv"

  echo "[RUN] $label"
  "$PYTHON" demo.py \
    --imagedir "$IMAGEDIR" \
    --calib "$CALIB" \
    --weights "$weights" \
    --stride "$STRIDE" \
    "${MAX_ARGS[@]}" \
    --buffer "$BUFFER" \
    --disable_vis \
    --reconstruction_path "$recon" \
    --imu_ba_debug_path "$debug" \
    "$@"

  if [[ "$RUN_PLY" == "1" ]]; then
    "$PYTHON" export_reconstruction_ply.py "$recon" --output "$OUT_DIR/${label}.ply"
  fi
  if [[ -f "$debug" ]]; then
    RUN_ARGS+=(--run "${label}=${recon},${debug}")
  else
    RUN_ARGS+=(--run "${label}=${recon}")
  fi
}

if [[ -n "$ORIGINAL_RECON" ]]; then
  if [[ ! -f "$ORIGINAL_RECON" ]]; then
    echo "ORIGINAL_RECON does not exist: $ORIGINAL_RECON" >&2
    exit 2
  fi
  validate_reconstruction_reuse "original_droid" "$ORIGINAL_RECON" 0
  echo "[REUSE] original_droid reconstruction=$ORIGINAL_RECON"
  RUN_ARGS+=(--run "original_droid=$ORIGINAL_RECON")
else
  run_case original_droid "$ORIGINAL_WEIGHTS"
fi

# This is the causal runtime control. It uses the exact checkpoint used by the
# IMU runs, but supplies no IMU prior or IMU optimization flags. Comparing it
# with imu_9dof_fixed separates IMU fusion from any visual-network fine-tuning.
if [[ "$RUN_MATCHED_VISUAL_ONLY" == "1" ]]; then
  if [[ -n "$MATCHED_VISUAL_ONLY_RECON" ]]; then
    if [[ ! -f "$MATCHED_VISUAL_ONLY_RECON" ]]; then
      echo "MATCHED_VISUAL_ONLY_RECON does not exist: $MATCHED_VISUAL_ONLY_RECON" >&2
      exit 2
    fi
    validate_reconstruction_reuse "finetuned_visual_only" "$MATCHED_VISUAL_ONLY_RECON" 0
    echo "[REUSE] finetuned_visual_only reconstruction=$MATCHED_VISUAL_ONLY_RECON"
    RUN_ARGS+=(--run "finetuned_visual_only=$MATCHED_VISUAL_ONLY_RECON")
  else
    run_case finetuned_visual_only "$IMU_WEIGHTS"
  fi
elif [[ "$RUN_MATCHED_VISUAL_ONLY" != "0" ]]; then
  echo "RUN_MATCHED_VISUAL_ONLY must be 0 or 1, got: $RUN_MATCHED_VISUAL_ONLY" >&2
  exit 2
fi

IMU_ARGS=(
  --imu_prior "$IMU_PRIOR"
  --imu_frame_manifest "$FRAMES_CSV"
  --cam_sensor_yaml "$CAMERA_SENSOR_YAML"
  --imu_sensor_yaml "$IMU_SENSOR_YAML"
  --use_pose_velocity_imu_ba
  --imu_full_frontend
  --use_learned_imu_confidence
  --use_learned_imu_ba_weight
  --use_vi_initialization
  --vi_init_window "$VI_INIT_WINDOW"
  --vi_init_bootstrap_frames "$VI_INIT_BOOTSTRAP_FRAMES"
  --vi_init_snapshot_history_size "$VI_INIT_SNAPSHOT_HISTORY_SIZE"
  --vi_init_snapshot_stride "$VI_INIT_SNAPSHOT_STRIDE"
  --vi_init_snapshot_max_candidates "$VI_INIT_SNAPSHOT_MAX_CANDIDATES"
  --imu_ba_debug
)

if [[ "$USE_LEGACY_ADAPTIVE_GATE" == "1" ]]; then
  if [[ "$USE_NOISE_MODEL" == "1" ]]; then
    echo "[IMU] disabling legacy threshold gate because USE_NOISE_MODEL=1" >&2
  else
    IMU_ARGS+=(--use_adaptive_imu_gate)
  fi
fi
if [[ "$USE_NOISE_MODEL" == "1" ]]; then
  IMU_ARGS+=(--use_imu_noise_model)
  if [[ "$USE_LEARNED_IMU_RESIDUAL_WEIGHTS" != "1" ]]; then
    echo "[IMU WARNING] covariance-mode run is using legacy position/velocity defaults because USE_LEARNED_IMU_RESIDUAL_WEIGHTS=0" >&2
  fi
else
  IMU_ARGS+=(--use_imu_info_weighting)
fi
if [[ "$USE_LEARNED_IMU_RESIDUAL_WEIGHTS" == "1" ]]; then
  IMU_ARGS+=(--use_learned_imu_residual_weights)
fi
if [[ "$VI_INIT_BACKEND_FALLBACK" == "1" ]]; then
  IMU_ARGS+=(--vi_init_backend_fallback --vi_init_backend_window "$VI_INIT_BACKEND_WINDOW")
elif [[ "$VI_INIT_BACKEND_FALLBACK" != "0" ]]; then
  echo "VI_INIT_BACKEND_FALLBACK must be 0 or 1, got: $VI_INIT_BACKEND_FALLBACK" >&2
  exit 2
fi

# This is the control: VI initialization estimates one global bias and it
# remains fixed while each local window optimizes T, v and dense depth.
if [[ -n "$FIXED_RECON" ]]; then
  if [[ ! -f "$FIXED_RECON" ]]; then
    echo "FIXED_RECON does not exist: $FIXED_RECON" >&2
    exit 2
  fi
  validate_reconstruction_reuse "imu_9dof_fixed" "$FIXED_RECON" 1
  echo "[REUSE] imu_9dof_fixed reconstruction=$FIXED_RECON"
  RUN_ARGS+=(--run "imu_9dof_fixed=$FIXED_RECON")
else
  run_case imu_9dof_fixed "$IMU_WEIGHTS" "${IMU_ARGS[@]}"
fi

# This path never uses the outer-relinearization diagnostic. It keeps the
# ordinary frontend local DBA window and updates one carried 6D bias only at a
# low rate. The termination-time backend update is separately opt-in.
if [[ "$RUN_SHARED_BIAS" == "1" ]]; then
  LOW_RATE_BIAS_ARGS=(
    "${IMU_ARGS[@]}"
    --use_low_rate_bias_refinement
    --imu_bias_refine_interval "$BIAS_REFINE_INTERVAL"
    --imu_bias_refine_window "$BIAS_REFINE_WINDOW"
    --imu_bias_refine_robust_delta "$BIAS_REFINE_ROBUST_DELTA"
    --imu_bias_refine_min_edges "$BIAS_REFINE_MIN_EDGES"
    --imu_bias_refine_min_duration "$BIAS_REFINE_MIN_DURATION"
    --imu_bias_refine_min_rotation_deg "$BIAS_REFINE_MIN_ROTATION_DEG"
    --imu_bias_refine_max_condition "$BIAS_REFINE_MAX_CONDITION"
    --imu_bias_refine_min_relative_improvement "$BIAS_REFINE_MIN_RELATIVE_IMPROVEMENT"
  )
  if [[ "$BIAS_REFINE_BACKEND" == "1" ]]; then
    LOW_RATE_BIAS_ARGS+=(--imu_bias_refine_backend)
  elif [[ "$BIAS_REFINE_BACKEND" != "0" ]]; then
    echo "BIAS_REFINE_BACKEND must be 0 or 1, got: $BIAS_REFINE_BACKEND" >&2
    exit 2
  fi
  if [[ -n "$LOW_RATE_BIAS_RECON" ]]; then
    if [[ ! -f "$LOW_RATE_BIAS_RECON" ]]; then
      echo "LOW_RATE_BIAS_RECON does not exist: $LOW_RATE_BIAS_RECON" >&2
      exit 2
    fi
    if [[ -n "$LOW_RATE_BIAS_DEBUG" && ! -f "$LOW_RATE_BIAS_DEBUG" ]]; then
      echo "LOW_RATE_BIAS_DEBUG does not exist: $LOW_RATE_BIAS_DEBUG" >&2
      exit 2
    fi
    validate_reconstruction_reuse "imu_9dof_lowrate_bias" "$LOW_RATE_BIAS_RECON" 1
    echo "[REUSE] imu_9dof_lowrate_bias reconstruction=$LOW_RATE_BIAS_RECON"
    if [[ -n "$LOW_RATE_BIAS_DEBUG" ]]; then
      RUN_ARGS+=(--run "imu_9dof_lowrate_bias=$LOW_RATE_BIAS_RECON,$LOW_RATE_BIAS_DEBUG")
    else
      RUN_ARGS+=(--run "imu_9dof_lowrate_bias=$LOW_RATE_BIAS_RECON")
    fi
  else
    run_case imu_9dof_lowrate_bias "$IMU_WEIGHTS" "${LOW_RATE_BIAS_ARGS[@]}"
  fi
elif [[ "$RUN_SHARED_BIAS" != "0" ]]; then
  echo "RUN_SHARED_BIAS must be 0 or 1, got: $RUN_SHARED_BIAS" >&2
  exit 2
fi

EVAL_BASE_ARGS=(
  --frames-csv "$FRAMES_CSV"
  --gt-csv "$GT_CSV"
  --gt-frame body
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML"
  "${RUN_ARGS[@]}"
)
if [[ -n "$TURN_FRAME" ]]; then
  EVAL_BASE_ARGS=(--turn-frame "$TURN_FRAME" "${EVAL_BASE_ARGS[@]}")
fi

# Keep the historical summary name for the fair paired comparison. The full
# report retains every valid pose in each run and therefore exposes tracking
# coverage differences that the common-frame report intentionally removes.
"$PYTHON" tools/evaluate_euroc_recon.py \
  --common-frame-ids \
  --output-csv "$OUT_DIR/summary.csv" \
  --output-md "$OUT_DIR/summary.md" \
  "${EVAL_BASE_ARGS[@]}"
"$PYTHON" tools/evaluate_euroc_recon.py \
  --output-csv "$OUT_DIR/summary_full.csv" \
  --output-md "$OUT_DIR/summary_full.md" \
  "${EVAL_BASE_ARGS[@]}"

echo "[DONE] $OUT_DIR"

#!/usr/bin/env bash
set -euo pipefail

# One held-out, apples-to-apples comparison:
#   stock visual DROID  vs.  one 15DoF confidence-only checkpoint.
#
# The reporting sequence is never used for checkpoint selection.  This runner
# intentionally has no fixed-IMU or tuned-IMU arm: its report answers only
# whether the learned model beats the requested visual-DROID baseline.

ROOT="${ROOT:-/home/ubuntu/DROID-SLAM}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/droid_env/bin/python}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to prepared V1_03/V2_03 *_droid_cam0}"
WEIGHTS="${WEIGHTS:?Set WEIGHTS to one learned 15DoF checkpoint}"
ORIGINAL_WEIGHTS="${ORIGINAL_WEIGHTS:-droid.pth}"
# An existing stock-DROID reconstruction may be reused only when it was made
# with this exact tracking/filling protocol. The check below deliberately
# rejects legacy 10 Hz dense reconstructions.
ORIGINAL_RECON="${ORIGINAL_RECON:-}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR to a new output directory}"
# Evaluation writes a manifest before launching the costly tracker.  If a
# process is interrupted before its first reconstruction is written, permit an
# *explicit* retry of that manifest-only directory.  Any directory containing
# a reconstruction, debug CSV, or report remains protected from overwrite.
ALLOW_INCOMPLETE_OUTPUT_REUSE="${ALLOW_INCOMPLETE_OUTPUT_REUSE:-0}"
IMAGEDIR="${IMAGEDIR:-${DATA_ROOT}/data}"
CALIB="${CALIB:-${DATA_ROOT}/calib.txt}"
# The held-out EuRoC protocol uses central-difference bias Jacobians when
# available.  Keep an explicitly supplied IMU_PRIOR authoritative, but do not
# silently fall back to the legacy analytic-Jacobian CSV merely because it has
# the shorter historical filename.
IMU_PRIOR="${IMU_PRIOR:-}"
if [[ -z "$IMU_PRIOR" ]]; then
  if [[ -f "$DATA_ROOT/imu_prior_fd_jacobian.csv" ]]; then
    IMU_PRIOR="$DATA_ROOT/imu_prior_fd_jacobian.csv"
  else
    IMU_PRIOR="$DATA_ROOT/imu_prior.csv"
  fi
fi
FRAMES_CSV="${FRAMES_CSV:-${DATA_ROOT}/frames.csv}"
CAMERA_SENSOR_YAML="${CAMERA_SENSOR_YAML:-${DATA_ROOT}/cam0_sensor.yaml}"
IMU_SENSOR_YAML="${IMU_SENSOR_YAML:-${DATA_ROOT}/imu0_sensor.yaml}"
GT_CSV="${GT_CSV:-}"
STRIDE="${STRIDE:-2}"
TRAJECTORY_FILL_STRIDE="${TRAJECTORY_FILL_STRIDE:-1}"
BUFFER="${BUFFER:-4096}"
MAX_FRAMES="${MAX_FRAMES:-0}"
# ``learned`` is the legacy sigmoid-confidence comparison. ``precision`` is
# the learned R/v/p covariance-relative attenuation model with no global IMU
# lambda. Its historical label is retained for command compatibility.
# ``fixed_one`` is
# a diagnostic control: it uses the identical 15DoF objective but supplies
# c_ij=1 for every valid IMU factor, without loading or invoking a confidence
# head. It isolates whether a result is limited by confidence prediction or
# by the underlying inertial BA/calibration.
CONFIDENCE_MODE="${CONFIDENCE_MODE:-learned}"
# ``fixed_one`` normally evaluates c=1.  A value in [0, 1] turns that arm
# into a controlled uniform-confidence diagnostic without changing the IMU
# factor, visual network, or two-stage schedule.  It is deliberately named
# confidence rather than lambda: the same component-wise confidence scaling
# is applied to residual and Jacobian rows.
FIXED_IMU_CONFIDENCE="${FIXED_IMU_CONFIDENCE:-1.0}"
# A residual-calibrated checkpoint must opt in at runtime.  It is valid for
# both learned and fixed-c=1 arms: when enabled, the latter becomes a true
# confidence ablation because p/v-vs-rotation calibration is shared and only
# c_ij differs.  Leave this off by default so older comparisons retain their
# exact fixed p/v settings.
LEARNED_RESIDUAL_WEIGHTS="${LEARNED_RESIDUAL_WEIGHTS:-0}"
# Select the checkpoint's learned confidence representation explicitly.
# Older DVI-style checkpoints use one scalar confidence plus the quality MLP;
# component checkpoints instead produce independent [rotation, velocity,
# position] confidences.  Do not silently evaluate a component checkpoint
# through its inactive scalar head.
USE_COMPONENT_IMU_CONFIDENCE="${USE_COMPONENT_IMU_CONFIDENCE:-0}"
# ``dense`` is the final paper-facing protocol: track every other raw 20 Hz
# frame but fill and evaluate every raw 20 Hz frame after global BA.
TRAJECTORY_SOURCE="${TRAJECTORY_SOURCE:-dense}"
# This profile makes both arms use the exact visual EuRoC protocol from the
# original DROID evaluator. IMU_PRIOR and CAMERA_SENSOR_YAML must then point
# to assets made by prepare_euroc_original_protocol_assets.py, because the
# rectification rotates the camera coordinate system.
EUROC_ORIGINAL_PROTOCOL="${EUROC_ORIGINAL_PROTOCOL:-0}"
# Opt-in objective variants. They apply identically to the fixed-c=1 and
# learned IMU arms, while the stock visual baseline remains untouched.
# Defaults retain the existing reproducible protocol.
USE_IMU_BIAS_RANDOM_WALK="${USE_IMU_BIAS_RANDOM_WALK:-0}"
USE_IMU_FULL_COVARIANCE_WEIGHTING="${USE_IMU_FULL_COVARIANCE_WEIGHTING:-0}"
# Diagnostic, GT-free post-global refinement of the one shared IMU bias.
# This is intentionally opt-in: the default remains the established 15DoF
# protocol.  When enabled, it estimates the common 6D accel/gyro bias from
# the completed visual/VI trajectory and IMU factors, then reruns the final
# global BA.  It never reads the evaluation ground truth.
USE_GLOBAL_BIAS_15DOF_REFINEMENT="${USE_GLOBAL_BIAS_15DOF_REFINEMENT:-0}"
GLOBAL_BIAS_POST_STEPS="${GLOBAL_BIAS_POST_STEPS:-2}"
GLOBAL_BIAS_REFINEMENT_WEIGHT="${GLOBAL_BIAS_REFINEMENT_WEIGHT:-1.0}"
# Unlike the post-hoc refinement above, this option introduces the shared 6D
# bias inside the final full 15DoF normal system and eliminates it by Schur
# complement before CUDA DBA. It is a separate structural experiment.
USE_JOINT_SHARED_BIAS_15DOF="${USE_JOINT_SHARED_BIAS_15DOF:-0}"
# Runtime two-stage schedule.  The default preserves the established
# visual:3/full:2 policy.  ``0`` full local iterations is an explicitly
# labelled diagnostic: it prevents online local bias drift while retaining
# the same final global 15DoF solve.
IMU_TWO_STAGE_VISUAL_STEPS="${IMU_TWO_STAGE_VISUAL_STEPS:-3}"
IMU_TWO_STAGE_FULL_STEPS="${IMU_TWO_STAGE_FULL_STEPS:-2}"

export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/droid_env/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

if [[ -z "$GT_CSV" && -f "$DATA_ROOT/source.txt" ]]; then
  GT_CSV="$(sed -n 's/^groundtruth=//p' "$DATA_ROOT/source.txt" | head -n 1)"
fi
if [[ -z "$GT_CSV" && -f "$DATA_ROOT/source.txt" ]]; then
  source_mav0="$(sed -n 's/^source_mav0=//p' "$DATA_ROOT/source.txt" | head -n 1)"
  if [[ -n "$source_mav0" ]]; then
    GT_CSV="$source_mav0/state_groundtruth_estimate0/data.csv"
  fi
fi

# The original-DROID EuRoC profile rectifies raw cam0 pixels on the fly.  A
# full IMU factor must therefore use the camera extrinsics and intrinsics in
# that *same* rectified frame.  Derive both files from the supplied rectified
# prior rather than leaving a caller to accidentally combine it with the raw
# camera YAML under DATA_ROOT (which is a physically different R_CI/t_CI).
if [[ "$EUROC_ORIGINAL_PROTOCOL" == "1" ]]; then
  # A diagnostic can place a derived IMU-prior CSV outside the immutable
  # rectified-asset directory.  In that case it must explicitly point back to
  # the camera calibration that produced the source prior.  Default behaviour
  # remains strict colocation, so normal evaluations cannot accidentally pair
  # an IMU prior with unrelated camera geometry.
  STRICT_ASSET_DIR="${STRICT_ASSET_DIR:-$(dirname "$IMU_PRIOR")}"
  STRICT_CAMERA_SENSOR_YAML="$STRICT_ASSET_DIR/cam0_sensor_rectified.yaml"
  STRICT_CALIB="$STRICT_ASSET_DIR/calib_rectified.txt"
  if [[ ! -f "$STRICT_CAMERA_SENSOR_YAML" || ! -f "$STRICT_CALIB" ]]; then
    echo "EUROC_ORIGINAL_PROTOCOL=1 requires rectified camera assets beside IMU_PRIOR:" >&2
    echo "  expected $STRICT_CAMERA_SENSOR_YAML" >&2
    echo "  expected $STRICT_CALIB" >&2
    exit 2
  fi
  CAMERA_SENSOR_YAML="$STRICT_CAMERA_SENSOR_YAML"
  CALIB="$STRICT_CALIB"
  echo "[STRICT ASSETS] using rectified calibration paired with IMU_PRIOR"
fi

for path in "$IMAGEDIR" "$CALIB" "$IMU_PRIOR" "$FRAMES_CSV" \
  "$CAMERA_SENSOR_YAML" "$IMU_SENSOR_YAML" "$GT_CSV" "$WEIGHTS"; do
  if [[ ! -e "$path" ]]; then
    echo "required input does not exist: $path" >&2
    exit 2
  fi
done
if [[ -n "$ORIGINAL_RECON" ]]; then
  if [[ ! -f "$ORIGINAL_RECON" ]]; then
    echo "ORIGINAL_RECON does not exist: $ORIGINAL_RECON" >&2
    exit 2
  fi
elif [[ ! -f "$ORIGINAL_WEIGHTS" ]]; then
  echo "required input does not exist: $ORIGINAL_WEIGHTS" >&2
  exit 2
fi
if [[ -e "$OUT_DIR" ]]; then
  if [[ "$ALLOW_INCOMPLETE_OUTPUT_REUSE" == "1" && -d "$OUT_DIR" && \
        -f "$OUT_DIR/run_manifest.txt" ]] && \
     [[ -z "$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 ! -name run_manifest.txt -print -quit)" ]]; then
    echo "[RETRY] reusing manifest-only incomplete output directory: $OUT_DIR"
  else
    echo "refusing to overwrite existing output directory: $OUT_DIR" >&2
    echo "Set ALLOW_INCOMPLETE_OUTPUT_REUSE=1 only when it contains run_manifest.txt and no result files." >&2
    exit 2
  fi
fi
if ! [[ "$STRIDE" =~ ^[1-9][0-9]*$ ]]; then
  echo "STRIDE must be a positive integer, got: $STRIDE" >&2
  exit 2
fi
if ! [[ "$TRAJECTORY_FILL_STRIDE" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAJECTORY_FILL_STRIDE must be a positive integer, got: $TRAJECTORY_FILL_STRIDE" >&2
  exit 2
fi
if [[ "$CONFIDENCE_MODE" != "learned" && "$CONFIDENCE_MODE" != "precision" && "$CONFIDENCE_MODE" != "fixed_one" ]]; then
  echo "CONFIDENCE_MODE must be learned, precision, or fixed_one, got: $CONFIDENCE_MODE" >&2
  exit 2
fi
if ! "$PYTHON" - "$FIXED_IMU_CONFIDENCE" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and 0.0 <= value <= 1.0 else 1)
PY
then
  echo "FIXED_IMU_CONFIDENCE must be finite and in [0, 1], got: $FIXED_IMU_CONFIDENCE" >&2
  exit 2
fi
if [[ "$LEARNED_RESIDUAL_WEIGHTS" != "0" && "$LEARNED_RESIDUAL_WEIGHTS" != "1" ]]; then
  echo "LEARNED_RESIDUAL_WEIGHTS must be 0 or 1, got: $LEARNED_RESIDUAL_WEIGHTS" >&2
  exit 2
fi
if [[ "$USE_COMPONENT_IMU_CONFIDENCE" != "0" && "$USE_COMPONENT_IMU_CONFIDENCE" != "1" ]]; then
  echo "USE_COMPONENT_IMU_CONFIDENCE must be 0 or 1, got: $USE_COMPONENT_IMU_CONFIDENCE" >&2
  exit 2
fi
if [[ "$CONFIDENCE_MODE" != "learned" && "$USE_COMPONENT_IMU_CONFIDENCE" == "1" ]]; then
  echo "USE_COMPONENT_IMU_CONFIDENCE=1 is only valid with CONFIDENCE_MODE=learned" >&2
  exit 2
fi
if [[ "$TRAJECTORY_SOURCE" != "keyframes" && "$TRAJECTORY_SOURCE" != "dense" ]]; then
  echo "TRAJECTORY_SOURCE must be keyframes or dense, got: $TRAJECTORY_SOURCE" >&2
  exit 2
fi
if [[ "$EUROC_ORIGINAL_PROTOCOL" != "0" && "$EUROC_ORIGINAL_PROTOCOL" != "1" ]]; then
  echo "EUROC_ORIGINAL_PROTOCOL must be 0 or 1, got: $EUROC_ORIGINAL_PROTOCOL" >&2
  exit 2
fi
if [[ "$USE_IMU_BIAS_RANDOM_WALK" != "0" && "$USE_IMU_BIAS_RANDOM_WALK" != "1" ]]; then
  echo "USE_IMU_BIAS_RANDOM_WALK must be 0 or 1" >&2
  exit 2
fi
if [[ "$USE_IMU_FULL_COVARIANCE_WEIGHTING" != "0" && "$USE_IMU_FULL_COVARIANCE_WEIGHTING" != "1" ]]; then
  echo "USE_IMU_FULL_COVARIANCE_WEIGHTING must be 0 or 1" >&2
  exit 2
fi
if [[ "$USE_GLOBAL_BIAS_15DOF_REFINEMENT" != "0" && "$USE_GLOBAL_BIAS_15DOF_REFINEMENT" != "1" ]]; then
  echo "USE_GLOBAL_BIAS_15DOF_REFINEMENT must be 0 or 1" >&2
  exit 2
fi
if [[ "$USE_JOINT_SHARED_BIAS_15DOF" != "0" && "$USE_JOINT_SHARED_BIAS_15DOF" != "1" ]]; then
  echo "USE_JOINT_SHARED_BIAS_15DOF must be 0 or 1" >&2
  exit 2
fi
if [[ "$USE_GLOBAL_BIAS_15DOF_REFINEMENT" == "1" && "$USE_JOINT_SHARED_BIAS_15DOF" == "1" ]]; then
  echo "choose either post-hoc global-bias refinement or joint shared 15DoF bias, not both" >&2
  exit 2
fi
if ! [[ "$GLOBAL_BIAS_POST_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GLOBAL_BIAS_POST_STEPS must be a positive integer, got: $GLOBAL_BIAS_POST_STEPS" >&2
  exit 2
fi
if ! [[ "$IMU_TWO_STAGE_VISUAL_STEPS" =~ ^[0-9]+$ ]]; then
  echo "IMU_TWO_STAGE_VISUAL_STEPS must be a non-negative integer, got: $IMU_TWO_STAGE_VISUAL_STEPS" >&2
  exit 2
fi
if ! [[ "$IMU_TWO_STAGE_FULL_STEPS" =~ ^[0-9]+$ ]]; then
  echo "IMU_TWO_STAGE_FULL_STEPS must be a non-negative integer, got: $IMU_TWO_STAGE_FULL_STEPS" >&2
  exit 2
fi
if ! "$PYTHON" - "$GLOBAL_BIAS_REFINEMENT_WEIGHT" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and value > 0.0 else 1)
PY
then
  echo "GLOBAL_BIAS_REFINEMENT_WEIGHT must be a finite positive number, got: $GLOBAL_BIAS_REFINEMENT_WEIGHT" >&2
  exit 2
fi
if [[ "$EUROC_ORIGINAL_PROTOCOL" == "1" ]]; then
  if [[ "$STRIDE" != "2" || "$TRAJECTORY_FILL_STRIDE" != "1" ]]; then
    echo "EUROC_ORIGINAL_PROTOCOL requires STRIDE=2 and TRAJECTORY_FILL_STRIDE=1" >&2
    exit 2
  fi
  if [[ "$BUFFER" != "512" ]]; then
    echo "EUROC_ORIGINAL_PROTOCOL requires BUFFER=512, got: $BUFFER" >&2
    exit 2
  fi
fi

profile_args=()
if [[ "$EUROC_ORIGINAL_PROTOCOL" == "1" ]]; then
  profile_args=(--euroc_original_protocol)
fi

case "$CONFIDENCE_MODE" in
  learned)
    RUN_LABEL="learned_15dof_confidence"
    CONFIDENCE_ARGS=(
      --imu_confidence_only_weighting
      --use_learned_imu_confidence
      --imu_confidence_floor 0
    )
    if [[ "$USE_COMPONENT_IMU_CONFIDENCE" == "1" ]]; then
      CONFIDENCE_ARGS+=(--use_component_imu_confidence)
    else
      CONFIDENCE_ARGS+=(
        --use_dvi_imu_confidence
        --use_dvi_imu_quality_fusion
      )
    fi
    ;;
  precision)
    if [[ "$LEARNED_RESIDUAL_WEIGHTS" != "0" ]]; then
      echo "precision mode learns R/v/p directly; LEARNED_RESIDUAL_WEIGHTS must be 0" >&2
      exit 2
    fi
    if [[ "$USE_IMU_BIAS_RANDOM_WALK" != "1" || "$USE_IMU_FULL_COVARIANCE_WEIGHTING" != "1" ]]; then
      echo "precision mode requires USE_IMU_BIAS_RANDOM_WALK=1 and USE_IMU_FULL_COVARIANCE_WEIGHTING=1" >&2
      exit 2
    fi
    RUN_LABEL="learned_15dof_attenuation"
    CONFIDENCE_ARGS=(--use_learned_imu_precision)
    ;;
  fixed_one)
    fixed_confidence_tag="$($PYTHON - "$FIXED_IMU_CONFIDENCE" <<'PY'
import sys
value = float(sys.argv[1])
print((f"{value:.6f}").rstrip("0").rstrip(".").replace(".", "p"))
PY
)"
    RUN_LABEL="fixed_c${fixed_confidence_tag}_15dof"
    CONFIDENCE_ARGS=(
      --imu_ba_prior_weight 1.0
      --imu_tuned_confidence_schedule "$FIXED_IMU_CONFIDENCE"
    )
    ;;
esac

if [[ "$LEARNED_RESIDUAL_WEIGHTS" == "1" ]]; then
  CONFIDENCE_ARGS+=(--use_learned_imu_residual_weights)
fi

IMU_OBJECTIVE_ARGS=()
if [[ "$USE_IMU_BIAS_RANDOM_WALK" == "1" ]]; then
  IMU_OBJECTIVE_ARGS+=(--use_imu_bias_random_walk)
fi
if [[ "$USE_IMU_FULL_COVARIANCE_WEIGHTING" == "1" ]]; then
  IMU_OBJECTIVE_ARGS+=(--use_imu_full_covariance_weighting)
fi

GLOBAL_BIAS_REFINEMENT_ARGS=()
if [[ "$USE_GLOBAL_BIAS_15DOF_REFINEMENT" == "1" ]]; then
  GLOBAL_BIAS_REFINEMENT_ARGS=(
    --use_global_bias_15dof_refinement
    --imu_global_bias_refine_post_steps "$GLOBAL_BIAS_POST_STEPS"
    --imu_global_bias_refine_weight "$GLOBAL_BIAS_REFINEMENT_WEIGHT"
  )
fi
if [[ "$USE_JOINT_SHARED_BIAS_15DOF" == "1" ]]; then
  GLOBAL_BIAS_REFINEMENT_ARGS+=(--use_joint_shared_bias_15dof)
fi

mkdir -p "$OUT_DIR"
{
  echo "purpose=held-out stock-visual-DROID versus 15DoF-IMU-control-or-confidence"
  echo "data_root=$(realpath "$DATA_ROOT")"
  echo "calib=$(realpath "$CALIB")"
  echo "camera_sensor_yaml=$(realpath "$CAMERA_SENSOR_YAML")"
  echo "imu_sensor_yaml=$(realpath "$IMU_SENSOR_YAML")"
  echo "imu_prior=$(realpath "$IMU_PRIOR")"
  echo "weights=$(realpath "$WEIGHTS")"
  echo "original_weights=$(realpath "$ORIGINAL_WEIGHTS")"
  echo "original_recon=${ORIGINAL_RECON:-generated_in_this_run}"
  echo "tracking_stride=$STRIDE"
  echo "trajectory_fill_stride=$TRAJECTORY_FILL_STRIDE"
  echo "buffer=$BUFFER"
  echo "max_frames=$MAX_FRAMES"
  echo "confidence_mode=$CONFIDENCE_MODE"
  echo "fixed_imu_confidence=$FIXED_IMU_CONFIDENCE"
  echo "learned_residual_weights=$LEARNED_RESIDUAL_WEIGHTS"
  echo "use_component_imu_confidence=$USE_COMPONENT_IMU_CONFIDENCE"
  echo "trajectory_source=$TRAJECTORY_SOURCE"
  echo "euroc_original_protocol=$EUROC_ORIGINAL_PROTOCOL"
  echo "use_imu_bias_random_walk=$USE_IMU_BIAS_RANDOM_WALK"
  echo "use_imu_full_covariance_weighting=$USE_IMU_FULL_COVARIANCE_WEIGHTING"
  echo "use_global_bias_15dof_refinement=$USE_GLOBAL_BIAS_15DOF_REFINEMENT"
  echo "global_bias_post_steps=$GLOBAL_BIAS_POST_STEPS"
  echo "global_bias_refinement_weight=$GLOBAL_BIAS_REFINEMENT_WEIGHT"
  echo "use_joint_shared_bias_15dof=$USE_JOINT_SHARED_BIAS_15DOF"
  echo "imu_two_stage_visual_steps=$IMU_TWO_STAGE_VISUAL_STEPS"
  echo "imu_two_stage_full_steps=$IMU_TWO_STAGE_FULL_STEPS"
  git rev-parse HEAD
  git status --short
} > "$OUT_DIR/run_manifest.txt"

max_args=()
if [[ "$MAX_FRAMES" != "0" ]]; then
  max_args=(--max_frames "$MAX_FRAMES")
fi

if [[ -n "$ORIGINAL_RECON" ]]; then
  "$PYTHON" - "$ORIGINAL_RECON" "$STRIDE" "$TRAJECTORY_FILL_STRIDE" "$EUROC_ORIGINAL_PROTOCOL" <<'PY'
import sys
import torch

path, expected_tracking, expected_fill, expected_profile = sys.argv[1:]
blob = torch.load(path, map_location="cpu")
protocol = blob.get("dense_protocol")
if not isinstance(protocol, dict):
    raise SystemExit(
        "refusing to reuse reconstruction without dense_protocol metadata; "
        "regenerate the visual baseline with the current runner"
    )
actual_tracking = int(protocol.get("tracking_stride", -1))
actual_fill = int(protocol.get("trajectory_fill_stride", -1))
if actual_tracking != int(expected_tracking) or actual_fill != int(expected_fill):
    raise SystemExit(
        "reused reconstruction protocol mismatch: "
        f"tracking={actual_tracking}, fill={actual_fill}; expected "
        f"tracking={expected_tracking}, fill={expected_fill}"
    )
actual_profile = int(bool(protocol.get("euroc_original_protocol", False)))
if actual_profile != int(expected_profile):
    raise SystemExit(
        "reused reconstruction EuRoC protocol mismatch: "
        f"strict={actual_profile}; expected strict={expected_profile}"
    )
PY
  original_recon="$ORIGINAL_RECON"
  echo "[REUSE] original_droid reconstruction: $original_recon"
else
  original_recon="$OUT_DIR/original_droid_recon.pth"
  echo "[RUN] original_droid"
  "$PYTHON" demo.py \
    --imagedir "$IMAGEDIR" \
    --calib "$CALIB" \
    --weights "$ORIGINAL_WEIGHTS" \
    --stride "$STRIDE" \
    --trajectory_fill_stride "$TRAJECTORY_FILL_STRIDE" \
    "${profile_args[@]}" \
    "${max_args[@]}" \
    --buffer "$BUFFER" \
    --disable_vis \
    --reconstruction_path "$original_recon"
fi

echo "[RUN] $RUN_LABEL"
# Training uses frozen keyframe manifests selected by the ordinary DROID
# MotionFilter at this same stride. Keep the runtime selection policy
# identical: two-stage 15DoF still performs VI alignment and full IMU BA, but
# does not force additional startup keyframes that the visual baseline and
# training clips never received.
"$PYTHON" demo.py \
  --imagedir "$IMAGEDIR" \
  --calib "$CALIB" \
  --weights "$WEIGHTS" \
  --stride "$STRIDE" \
  --trajectory_fill_stride "$TRAJECTORY_FILL_STRIDE" \
  "${profile_args[@]}" \
  "${max_args[@]}" \
  --buffer "$BUFFER" \
  --disable_vis \
  --reconstruction_path "$OUT_DIR/${RUN_LABEL}_recon.pth" \
  --imu_prior "$IMU_PRIOR" \
  --imu_frame_manifest "$FRAMES_CSV" \
  --cam_sensor_yaml "$CAMERA_SENSOR_YAML" \
  --imu_sensor_yaml "$IMU_SENSOR_YAML" \
  --use_two_stage_imu_ba \
  --imu_two_stage_visual_steps "$IMU_TWO_STAGE_VISUAL_STEPS" \
  --imu_two_stage_full_steps "$IMU_TWO_STAGE_FULL_STEPS" \
  --no_vi_init_bootstrap \
  --use_imu_noise_model \
  "${IMU_OBJECTIVE_ARGS[@]}" \
  "${GLOBAL_BIAS_REFINEMENT_ARGS[@]}" \
  "${CONFIDENCE_ARGS[@]}" \
  --imu_ba_debug \
  --imu_ba_debug_path "$OUT_DIR/${RUN_LABEL}_debug.csv"

eval_args=(
  --frames-csv "$FRAMES_CSV"
  --gt-csv "$GT_CSV"
  --gt-frame body
  --camera-sensor-yaml "$CAMERA_SENSOR_YAML"
  --run "original_droid=$original_recon"
  --run "$RUN_LABEL=$OUT_DIR/${RUN_LABEL}_recon.pth,$OUT_DIR/${RUN_LABEL}_debug.csv"
)

"$PYTHON" tools/evaluate_euroc_recon.py \
  --trajectory-source "$TRAJECTORY_SOURCE" \
  --common-frame-ids \
  --output-csv "$OUT_DIR/summary_common.csv" \
  --output-md "$OUT_DIR/summary_common.md" \
  "${eval_args[@]}"
"$PYTHON" tools/evaluate_euroc_recon.py \
  --trajectory-source "$TRAJECTORY_SOURCE" \
  --output-csv "$OUT_DIR/summary_full.csv" \
  --output-md "$OUT_DIR/summary_full.md" \
  "${eval_args[@]}"

echo "[DONE] $OUT_DIR"
echo "[DONE] comparison=$OUT_DIR/summary_common.md"

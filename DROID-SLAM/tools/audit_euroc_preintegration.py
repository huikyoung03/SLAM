#!/usr/bin/env python3
"""Audit a EuRoC preintegrated IMU prior against training-sequence ground truth.

This is a diagnostic tool, not an ATE evaluator and not a source of runtime
labels.  It checks whether the values consumed by the full 15DoF factor have
the same frame convention as EuRoC ground truth:

* ``dr`` is compared with the camera-frame relative rotation expected by the
  active ``dq_prev + inverse`` runtime convention;
* ``dv`` and ``dp`` are compared with GT finite-difference velocity and
  position changes, expressed in the start camera frame and with gravity
  removed exactly as in the BA residual.

Run this only on MH training sequences.  V1_03 and V2_03 stay untouched for
validation/final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ``droid_slam`` is intentionally a source directory, not an installed Python
# package.  Add it explicitly so this stand-alone audit uses the exact runtime
# compositor rather than a copied implementation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DROID_SOURCE = _REPO_ROOT / "droid_slam"
if str(_DROID_SOURCE) not in sys.path:
    sys.path.insert(0, str(_DROID_SOURCE))
from imu_prior import compose_imu_prior_rows


def _normalise(name: str) -> str:
    value = name.lstrip("\ufeff").lower().replace("#", "")
    for char in "[]()/":
        value = value.replace(char, "_")
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def _field(row: Dict[str, str], *names: str) -> str:
    wanted = {_normalise(name) for name in names}
    for key, value in row.items():
        if _normalise(key) in wanted and value not in (None, ""):
            return str(value)
    raise KeyError(f"missing one of {names}; available={list(row)}")


def _float(row: Dict[str, str], *names: str) -> float:
    return float(_field(row, *names))


def _int(row: Dict[str, str], *names: str) -> int:
    return int(float(_field(row, *names)))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_selected_manifest(path: Path) -> List[Dict[str, int]]:
    """Load the frozen MotionFilter decisions used by ordered training.

    The manifest stores original 20 Hz EuRoC camera-row indices.  It is not a
    new selection policy: it records the decisions made by the visual DROID
    MotionFilter under the strict original protocol.
    """

    rows = _read_csv(path)
    if len(rows) < 2:
        raise ValueError(f"selected-frame manifest needs at least two rows: {path}")
    selected = [
        {
            "frame_index": _int(row, "frame_index"),
            "timestamp_ns": _int(row, "timestamp_ns"),
        }
        for row in rows
    ]
    previous_index = -1
    previous_timestamp = -1
    for row in selected:
        if row["frame_index"] <= previous_index or row["timestamp_ns"] <= previous_timestamp:
            raise ValueError(
                "selected-frame manifest must be strictly ordered by frame_index "
                f"and timestamp: {path}"
            )
        previous_index = row["frame_index"]
        previous_timestamp = row["timestamp_ns"]
    return selected


def _load_selected_reconstruction(
    reconstruction_path: Path,
    frames_csv_path: Path,
) -> List[Dict[str, int]]:
    """Recover final global-BA keyframe intervals from a DROID reconstruction.

    ``tstamps`` holds original prepared-image indices. ``frames.csv`` maps
    those indices to EuRoC timestamps, so this produces the variable-duration
    intervals that the final backend actually received rather than all raw
    20 Hz camera intervals.
    """

    if not reconstruction_path.is_file():
        raise FileNotFoundError(f"reconstruction does not exist: {reconstruction_path}")
    if not frames_csv_path.is_file():
        raise FileNotFoundError(f"frames CSV does not exist: {frames_csv_path}")

    # Raw-prior audits do not need torch, so import it only for this mode.
    import torch

    blob = torch.load(reconstruction_path, map_location="cpu")
    if not isinstance(blob, dict) or "tstamps" not in blob:
        raise ValueError(
            "reconstruction must be a DROID reconstruction dictionary containing tstamps: "
            f"{reconstruction_path}"
        )
    keyframe_indices = np.asarray(blob["tstamps"], dtype=np.float64).reshape(-1)
    if keyframe_indices.size < 2:
        raise ValueError("reconstruction needs at least two surviving keyframes")

    frames: Dict[int, int] = {}
    for row in _read_csv(frames_csv_path):
        frame_index = _int(row, "frame_index", "frame_id")
        timestamp_ns = _int(row, "timestamp_ns")
        if frame_index in frames:
            raise ValueError(f"duplicate frame index {frame_index} in {frames_csv_path}")
        frames[frame_index] = timestamp_ns

    selected: List[Dict[str, int]] = []
    for value in keyframe_indices:
        frame_index = int(round(float(value)))
        if abs(float(value) - frame_index) > 1e-4:
            raise ValueError(
                "reconstruction timestamp is not an integer prepared-image index: "
                f"{value}"
            )
        if frame_index not in frames:
            raise ValueError(
                f"reconstruction keyframe {frame_index} has no entry in {frames_csv_path}"
            )
        selected.append(
            {"frame_index": frame_index, "timestamp_ns": frames[frame_index]}
        )

    previous_index = -1
    previous_timestamp = -1
    for row in selected:
        if row["frame_index"] <= previous_index or row["timestamp_ns"] <= previous_timestamp:
            raise ValueError(
                "reconstruction keyframes must be strictly ordered by original frame index "
                "and timestamp"
            )
        previous_index = row["frame_index"]
        previous_timestamp = row["timestamp_ns"]
    return selected


def _normalise_quaternion(quaternion: np.ndarray) -> np.ndarray:
    return quaternion / max(float(np.linalg.norm(quaternion)), 1e-15)


def _slerp(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    first = _normalise_quaternion(first)
    second = _normalise_quaternion(second)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalise_quaternion((1.0 - alpha) * first + alpha * second)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return (
        math.sin((1.0 - alpha) * theta) / sin_theta * first
        + math.sin(alpha * theta) / sin_theta * second
    )


def _quat_to_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalise_quaternion(quaternion_wxyz)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotvec_to_rotation(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    skew = np.array(
        [
            [0.0, -rotvec[2], rotvec[1]],
            [rotvec[2], 0.0, -rotvec[0]],
            [-rotvec[1], rotvec[0], 0.0],
        ],
        dtype=np.float64,
    )
    if theta < 1e-10:
        return np.eye(3) + skew
    return np.eye(3) + math.sin(theta) / theta * skew + (1.0 - math.cos(theta)) / (theta * theta) * (skew @ skew)


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = (float(np.trace(rotation)) - 1.0) * 0.5
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def _load_gt(path: Path):
    rows = _read_csv(path)
    if len(rows) < 3:
        raise ValueError(f"ground truth needs at least three rows: {path}")
    timestamps = []
    positions = []
    quaternions = []
    velocities = []
    has_velocity = True
    accel_biases = []
    gyro_biases = []
    has_bias = True
    for row in rows:
        timestamps.append(_int(row, "timestamp_ns", "#timestamp [ns]", "#timestamp"))
        positions.append(
            [
                _float(row, "p_RS_R_x [m]", "p_rs_r_x"),
                _float(row, "p_RS_R_y [m]", "p_rs_r_y"),
                _float(row, "p_RS_R_z [m]", "p_rs_r_z"),
            ]
        )
        quaternions.append(
            [
                _float(row, "q_RS_w []", "q_rs_w"),
                _float(row, "q_RS_x []", "q_rs_x"),
                _float(row, "q_RS_y []", "q_rs_y"),
                _float(row, "q_RS_z []", "q_rs_z"),
            ]
        )
        try:
            velocities.append(
                [
                    _float(row, "v_RS_R_x [m s^-1]", "v_rs_r_x"),
                    _float(row, "v_RS_R_y [m s^-1]", "v_rs_r_y"),
                    _float(row, "v_RS_R_z [m s^-1]", "v_rs_r_z"),
                ]
            )
        except KeyError:
            has_velocity = False
        try:
            gyro_biases.append(
                [
                    _float(row, "b_w_RS_S_x [rad s^-1]", "b_w_rs_s_x"),
                    _float(row, "b_w_RS_S_y [rad s^-1]", "b_w_rs_s_y"),
                    _float(row, "b_w_RS_S_z [rad s^-1]", "b_w_rs_s_z"),
                ]
            )
            accel_biases.append(
                [
                    _float(row, "b_a_RS_S_x [m s^-2]", "b_a_rs_s_x"),
                    _float(row, "b_a_RS_S_y [m s^-2]", "b_a_rs_s_y"),
                    _float(row, "b_a_RS_S_z [m s^-2]", "b_a_rs_s_z"),
                ]
            )
        except KeyError:
            has_bias = False
    timestamps = np.asarray(timestamps, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    positions = positions[order]
    quaternions = quaternions[order]
    velocities = np.asarray(velocities, dtype=np.float64)[order] if has_velocity else None
    accel_biases = np.asarray(accel_biases, dtype=np.float64)[order] if has_bias else None
    gyro_biases = np.asarray(gyro_biases, dtype=np.float64)[order] if has_bias else None
    seconds = timestamps.astype(np.float64) * 1e-9
    return timestamps, positions, quaternions, velocities, accel_biases, gyro_biases


def _body_to_imu(session: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read EuRoC/ROS ``T_BS``: IMU sensor coordinates into body coordinates."""

    sensor_yaml = session / "imu0_sensor.yaml"
    if not sensor_yaml.is_file():
        raise FileNotFoundError(f"IMU calibration does not exist: {sensor_yaml}")
    text = sensor_yaml.read_text(encoding="utf-8")
    block = re.search(r"T_BS\s*:\s*(.*?)(?:\n\S|\Z)", text, re.S)
    data_match = re.search(
        r"data\s*:\s*\[([^\]]+)\]", block.group(1) if block else text, re.S
    )
    if data_match is None:
        raise ValueError(f"cannot read T_BS.data from {sensor_yaml}")
    values = [
        float(value)
        for value in re.split(r"[,\s]+", data_match.group(1).strip())
        if value
    ]
    if len(values) != 16:
        raise ValueError(f"expected 16 T_BS values in {sensor_yaml}, got {len(values)}")
    transform = np.asarray(values, dtype=np.float64).reshape(4, 4)
    return transform[:3, :3], transform[:3, 3]


def _sample_gt(
    timestamp_ns: int,
    timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    velocities: np.ndarray,
    max_gap_ns: int,
):
    right = int(np.searchsorted(timestamps, timestamp_ns, side="left"))
    if right < len(timestamps) and timestamps[right] == timestamp_ns:
        return positions[right], quaternions[right], velocities[right]
    if right == 0 or right == len(timestamps):
        return None
    left = right - 1
    if timestamp_ns - timestamps[left] > max_gap_ns or timestamps[right] - timestamp_ns > max_gap_ns:
        return None
    alpha = (timestamp_ns - timestamps[left]) / float(timestamps[right] - timestamps[left])
    return (
        (1.0 - alpha) * positions[left] + alpha * positions[right],
        _slerp(quaternions[left], quaternions[right], alpha),
        (1.0 - alpha) * velocities[left] + alpha * velocities[right],
    )


def _matrix_from_prior(row: Dict[str, str]) -> np.ndarray:
    return np.asarray(
        [
            [_float(row, f"R_cam_imu_{r}{c}") for c in range(3)]
            for r in range(3)
        ],
        dtype=np.float64,
    )


def _matrix_from_bias_jacobian(row: Dict[str, object], prefix: str) -> np.ndarray:
    """Read the propagated 3x3 bias Jacobian stored in an IMU prior row."""

    return np.asarray(
        [
            [_float(row, f"{prefix}_{r}{c}") for c in range(3)]
            for r in range(3)
        ],
        dtype=np.float64,
    )


def _three_from_blob(blob: Dict[str, object], key: str) -> np.ndarray:
    """Read a finite physical 3-vector from a DROID reconstruction."""

    value = blob.get(key)
    if value is None:
        raise KeyError(f"reconstruction does not contain {key}")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if value.size != 3 or not np.isfinite(value).all():
        raise ValueError(f"{key} must be a finite 3-vector, got {value}")
    return value


def _load_runtime_biases(reconstruction_path: Path):
    """Load the exact global and local 15DoF bias states used at runtime."""

    import torch

    blob = torch.load(reconstruction_path, map_location="cpu")
    if not isinstance(blob, dict):
        raise ValueError(f"reconstruction is not a dictionary: {reconstruction_path}")
    tstamps = np.asarray(blob.get("tstamps"), dtype=np.float64).reshape(-1)
    local_accel = blob.get("bias_acc")
    local_gyro = blob.get("bias_gyro")
    if local_accel is None or local_gyro is None:
        raise KeyError("reconstruction does not contain local bias_acc/bias_gyro")
    if hasattr(local_accel, "detach"):
        local_accel = local_accel.detach().cpu().numpy()
    if hasattr(local_gyro, "detach"):
        local_gyro = local_gyro.detach().cpu().numpy()
    local_accel = np.asarray(local_accel, dtype=np.float64)
    local_gyro = np.asarray(local_gyro, dtype=np.float64)
    if local_accel.shape != (tstamps.size, 3) or local_gyro.shape != (tstamps.size, 3):
        raise ValueError(
            "incompatible reconstruction keyframe/bias arrays: "
            f"tstamps={tstamps.shape}, bias_acc={local_accel.shape}, "
            f"bias_gyro={local_gyro.shape}"
        )
    local_by_frame = {}
    for index, value in enumerate(tstamps):
        frame_index = int(round(float(value)))
        if abs(float(value) - frame_index) > 1e-4:
            raise ValueError(f"non-integer reconstruction timestamp: {value}")
        local_by_frame[frame_index] = (local_accel[index], local_gyro[index])
    return (
        _three_from_blob(blob, "imu_global_bias_acc"),
        _three_from_blob(blob, "imu_global_bias_gyro"),
        local_by_frame,
    )


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    norms = np.linalg.norm(values, axis=1)
    return {
        "mean": float(np.mean(norms)),
        "median": float(np.median(norms)),
        "p95": float(np.quantile(norms, 0.95)),
        "max": float(np.max(norms)),
        "signed_mean_x": float(np.mean(values[:, 0])),
        "signed_mean_y": float(np.mean(values[:, 1])),
        "signed_mean_z": float(np.mean(values[:, 2])),
    }


def _edge_row(
    row: Dict[str, object],
    previous_ns: int,
    now_ns: int,
    frame_index: int,
    timestamps: np.ndarray,
    imu_positions: np.ndarray,
    body_quaternions: np.ndarray,
    imu_velocities: np.ndarray,
    gravity: np.ndarray,
    max_gap_ns: int,
    global_accel_bias: np.ndarray,
    global_gyro_bias: np.ndarray,
    local_bias_by_frame: Dict[int, Tuple[np.ndarray, np.ndarray]],
    previous_frame_index: int,
):
    """Compare one raw or composed runtime prior with GT at its endpoints."""

    dt = _float(row, "dt")
    if dt <= 0.0:
        return None
    current = _sample_gt(
        now_ns,
        timestamps,
        imu_positions,
        body_quaternions,
        imu_velocities,
        max_gap_ns,
    )
    previous = _sample_gt(
        previous_ns,
        timestamps,
        imu_positions,
        body_quaternions,
        imu_velocities,
        max_gap_ns,
    )
    if current is None or previous is None:
        return None
    p_i, q_wb_i, v_i = previous
    p_j, q_wb_j, v_j = current
    # GT is root->ego/body.  Convert body orientation to IMU orientation,
    # then use the runtime's world-to-camera pose convention.
    R_ci = _matrix_from_prior(row)
    # ``R_bi`` is captured in the prior row below. This assignment is filled
    # by the caller to keep this helper independent of session I/O.
    R_bi = np.asarray(row["_audit_R_bi"], dtype=np.float64)
    R_wi_i = _quat_to_rotation(q_wb_i) @ R_bi
    R_wi_j = _quat_to_rotation(q_wb_j) @ R_bi
    R_iw_i = R_wi_i.T
    R_iw_j = R_wi_j.T
    R_cw_i = R_ci @ R_iw_i
    R_cw_j = R_ci @ R_iw_j

    # Match full_imu_factor_linearization(): the preintegrated deltas are
    # corrected by the same propagated Jacobians and the start-frame bias.
    local_accel_bias, local_gyro_bias = local_bias_by_frame.get(
        int(previous_frame_index),
        (np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)),
    )
    accel_bias = global_accel_bias + local_accel_bias
    gyro_bias = global_gyro_bias + local_gyro_bias
    J_R_bg = _matrix_from_bias_jacobian(row, "J_R_bg")
    J_v_ba = _matrix_from_bias_jacobian(row, "J_v_ba")
    J_v_bg = _matrix_from_bias_jacobian(row, "J_v_bg")
    J_p_ba = _matrix_from_bias_jacobian(row, "J_p_ba")
    J_p_bg = _matrix_from_bias_jacobian(row, "J_p_bg")

    # Active runtime convention: q_j = inv(dq) * q_i.
    R_delta_truth = R_cw_i @ R_cw_j.T
    dr = np.asarray([_float(row, "dr_x"), _float(row, "dr_y"), _float(row, "dr_z")])
    dr = dr + J_R_bg @ gyro_bias
    R_delta_prior = _rotvec_to_rotation(dr)
    rotation_error = _rotation_angle_degrees(R_delta_prior.T @ R_delta_truth)

    dv_truth = R_cw_i @ (v_j - v_i - gravity * dt)
    dp_truth = R_cw_i @ (p_j - p_i - v_i * dt - 0.5 * gravity * dt * dt)
    dv_prior = np.asarray([_float(row, "dv_x"), _float(row, "dv_y"), _float(row, "dv_z")])
    dp_prior = np.asarray([_float(row, "dp_x"), _float(row, "dp_y"), _float(row, "dp_z")])
    dv_prior = dv_prior + J_v_ba @ accel_bias + J_v_bg @ gyro_bias
    dp_prior = dp_prior + J_p_ba @ accel_bias + J_p_bg @ gyro_bias
    dv_error = dv_prior - dv_truth
    dp_error = dp_prior - dp_truth
    return {
        "frame_index": int(frame_index),
        "timestamp_ns": int(now_ns),
        "prev_timestamp_ns": int(previous_ns),
        "dt": float(dt),
        "timestamp_dt_error_ms": float(
            1e3 * (dt - (now_ns - previous_ns) * 1e-9)
        ),
        "rotation_error_deg": rotation_error,
        "dv_error": dv_error,
        "dp_error": dp_error,
        "applied_accel_bias": accel_bias,
        "applied_gyro_bias": gyro_bias,
    }


def audit(args: argparse.Namespace) -> Dict[str, object]:
    session = Path(args.session).expanduser().resolve()
    prior_path = session / args.prior_name
    if not prior_path.is_file():
        raise FileNotFoundError(f"IMU prior does not exist: {prior_path}")
    gt_path = Path(args.groundtruth).expanduser().resolve()
    if not gt_path.is_file():
        raise FileNotFoundError(f"ground truth does not exist: {gt_path}")

    (
        timestamps,
        body_positions,
        body_quaternions,
        body_velocities,
        gt_accel_biases,
        gt_gyro_biases,
    ) = _load_gt(gt_path)
    R_bi, t_bi = _body_to_imu(session)
    R_wb_all = np.asarray([_quat_to_rotation(q) for q in body_quaternions])
    # The MCAP-to-mav0 converter stores the root->ego/body GT pose, while the
    # IMU prior is integrated in the imu0 sensor frame.  Convert GT first;
    # treating p_RS directly as the IMU centre creates a false lever-arm and
    # orientation error.
    imu_positions = body_positions + np.einsum("nij,j->ni", R_wb_all, t_bi)
    seconds = timestamps.astype(np.float64) * 1e-9
    if body_velocities is None:
        imu_velocities = np.gradient(imu_positions, seconds, axis=0, edge_order=2)
        velocity_reference = "finite_difference_position"
    else:
        # The EuRoC IMU has zero body translation in the active calibration.
        # For a non-zero lever arm a rotational correction would be required.
        imu_velocities = body_velocities
        velocity_reference = "groundtruth_velocity"
    gravity = np.asarray(args.gravity, dtype=np.float64)
    max_gap_ns = int(args.max_gt_gap_ms * 1e6)
    valid_rows = 0
    skipped_rows = 0
    rotation_errors = []
    velocity_errors = []
    position_errors = []
    per_edge = []
    raw_rows = _read_csv(prior_path)
    interval_mode = "raw_camera_rows"
    selected_manifest_path = None
    reconstruction_path = None
    audit_rows: List[Tuple[Dict[str, object], int, int, int, int]] = []
    global_accel_bias = np.zeros(3, dtype=np.float64)
    global_gyro_bias = np.zeros(3, dtype=np.float64)
    local_bias_by_frame: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    runtime_bias_source = "none"
    if args.apply_reconstruction_bias:
        if args.reconstruction is None:
            raise ValueError("--apply-reconstruction-bias requires --reconstruction")
        reconstruction_for_bias = Path(args.reconstruction).expanduser().resolve()
        (
            global_accel_bias,
            global_gyro_bias,
            local_bias_by_frame,
        ) = _load_runtime_biases(reconstruction_for_bias)
        runtime_bias_source = str(reconstruction_for_bias)

    if args.reconstruction is not None:
        reconstruction_path = Path(args.reconstruction).expanduser().resolve()
        frames_csv_path = Path(args.frames_csv).expanduser().resolve()
        selected = _load_selected_reconstruction(reconstruction_path, frames_csv_path)
        raw_by_index = {_int(row, "frame_index"): row for row in raw_rows}
        interval_mode = "runtime_reconstruction_keyframe_intervals"
        for previous, current in zip(selected[:-1], selected[1:]):
            previous_index = previous["frame_index"]
            current_index = current["frame_index"]
            segment = []
            for index in range(previous_index + 1, current_index + 1):
                raw = raw_by_index.get(index)
                if raw is None:
                    raise ValueError(
                        "reconstruction keyframe interval has no contiguous raw IMU prior row: "
                        f"{previous_index}->{current_index}, missing frame_index={index}"
                    )
                segment.append(raw)
            composed = compose_imu_prior_rows(
                segment,
                prev_frame_index=previous_index,
                frame_index=current_index,
            )
            if composed is None or int(composed.get("imu_valid", 0)) != 1:
                continue
            audit_rows.append(
                (
                    dict(composed, _audit_R_bi=R_bi.tolist()),
                    previous["timestamp_ns"],
                    current["timestamp_ns"],
                    current_index,
                    previous_index,
                )
            )
    elif args.selected_manifest is None:
        for row in raw_rows:
            if int(float(row.get("imu_valid", "0"))) != 1:
                continue
            audit_rows.append(
                (
                    dict(row, _audit_R_bi=R_bi.tolist()),
                    _int(row, "prev_timestamp_ns"),
                    _int(row, "timestamp_ns"),
                    _int(row, "frame_index"),
                    _int(row, "frame_index") - 1,
                )
            )
    else:
        selected_manifest_path = Path(args.selected_manifest).expanduser().resolve()
        if not selected_manifest_path.is_file():
            raise FileNotFoundError(
                f"selected-frame manifest does not exist: {selected_manifest_path}"
            )
        selected = _load_selected_manifest(selected_manifest_path)
        raw_by_index = {_int(row, "frame_index"): row for row in raw_rows}
        interval_mode = "motion_filter_selected_intervals"
        for previous, current in zip(selected[:-1], selected[1:]):
            previous_index = previous["frame_index"]
            current_index = current["frame_index"]
            segment = []
            for index in range(previous_index + 1, current_index + 1):
                raw = raw_by_index.get(index)
                if raw is None:
                    raise ValueError(
                        "selected interval has no contiguous raw IMU prior row: "
                        f"{previous_index}->{current_index}, missing frame_index={index}"
                    )
                segment.append(raw)
            composed = compose_imu_prior_rows(
                segment,
                prev_frame_index=previous_index,
                frame_index=current_index,
            )
            if composed is None or int(composed.get("imu_valid", 0)) != 1:
                continue
            # The compositor intentionally carries the final CSV row as its
            # template.  For this audit, endpoints must be the selected pair,
            # not that final raw row's one-camera predecessor.
            composed = dict(composed, _audit_R_bi=R_bi.tolist())
            audit_rows.append(
                (
                    composed,
                    previous["timestamp_ns"],
                    current["timestamp_ns"],
                    current_index,
                    previous_index,
                )
            )

    for row, previous_ns, now_ns, frame_index, previous_frame_index in audit_rows:
        measurement = _edge_row(
            row,
            previous_ns,
            now_ns,
            frame_index,
            timestamps,
            imu_positions,
            body_quaternions,
            imu_velocities,
            gravity,
            max_gap_ns,
            global_accel_bias,
            global_gyro_bias,
            local_bias_by_frame,
            previous_frame_index,
        )
        if measurement is None:
            skipped_rows += 1
            continue
        rotation_errors.append(measurement["rotation_error_deg"])
        velocity_errors.append(measurement["dv_error"])
        position_errors.append(measurement["dp_error"])
        valid_rows += 1
        per_edge.append(
            {
                "frame_index": measurement["frame_index"],
                "prev_timestamp_ns": measurement["prev_timestamp_ns"],
                "timestamp_ns": measurement["timestamp_ns"],
                "dt": measurement["dt"],
                "timestamp_dt_error_ms": measurement["timestamp_dt_error_ms"],
                "rotation_error_deg": measurement["rotation_error_deg"],
                "dv_error_norm_mps": float(np.linalg.norm(measurement["dv_error"])),
                "dp_error_norm_m": float(np.linalg.norm(measurement["dp_error"])),
                "dv_error_x": float(measurement["dv_error"][0]),
                "dv_error_y": float(measurement["dv_error"][1]),
                "dv_error_z": float(measurement["dv_error"][2]),
                "dp_error_x": float(measurement["dp_error"][0]),
                "dp_error_y": float(measurement["dp_error"][1]),
                "dp_error_z": float(measurement["dp_error"][2]),
                "applied_accel_bias_x": float(measurement["applied_accel_bias"][0]),
                "applied_accel_bias_y": float(measurement["applied_accel_bias"][1]),
                "applied_accel_bias_z": float(measurement["applied_accel_bias"][2]),
                "applied_gyro_bias_x": float(measurement["applied_gyro_bias"][0]),
                "applied_gyro_bias_y": float(measurement["applied_gyro_bias"][1]),
                "applied_gyro_bias_z": float(measurement["applied_gyro_bias"][2]),
            }
        )
        if args.max_rows > 0 and valid_rows >= args.max_rows:
            break

    if not rotation_errors:
        raise RuntimeError("no valid prior/GT intervals were available for audit")

    rotation_errors_np = np.asarray(rotation_errors, dtype=np.float64)
    velocity_errors_np = np.asarray(velocity_errors, dtype=np.float64)
    position_errors_np = np.asarray(position_errors, dtype=np.float64)
    nominal_accel_biases = np.asarray(
        [
            [_float(row, "acc_bias_x"), _float(row, "acc_bias_y"), _float(row, "acc_bias_z")]
            for row, _, _, _, _ in audit_rows
        ],
        dtype=np.float64,
    )
    nominal_gyro_biases = np.asarray(
        [
            [_float(row, "gyro_bias_x"), _float(row, "gyro_bias_y"), _float(row, "gyro_bias_z")]
            for row, _, _, _, _ in audit_rows
        ],
        dtype=np.float64,
    )
    summary: Dict[str, object] = {
        "session": str(session),
        "groundtruth": str(gt_path),
        "interval_mode": interval_mode,
        "selected_manifest": str(selected_manifest_path) if selected_manifest_path else None,
        "reconstruction": str(reconstruction_path) if reconstruction_path else None,
        "runtime_convention": "dq_prev + inverse_imu=true",
        "runtime_bias_source": runtime_bias_source,
        "applied_global_accel_bias": [float(value) for value in global_accel_bias],
        "applied_global_gyro_bias": [float(value) for value in global_gyro_bias],
        "gt_pose_frame": "root->ego/body converted through imu0 T_BS",
        "velocity_reference": velocity_reference,
        "imu_to_body_translation_m": [float(value) for value in t_bi],
        "gravity_world_mps2": [float(value) for value in gravity],
        "intervals_used": valid_rows,
        "intervals_skipped_missing_gt": skipped_rows,
        "rotation_error_deg": {
            "mean": float(np.mean(rotation_errors_np)),
            "median": float(np.median(rotation_errors_np)),
            "p95": float(np.quantile(rotation_errors_np, 0.95)),
            "max": float(np.max(rotation_errors_np)),
        },
        "integration_timestamp_error_ms": {
            "mean": float(np.mean([edge["timestamp_dt_error_ms"] for edge in per_edge])),
            "max_abs": float(
                np.max(np.abs([edge["timestamp_dt_error_ms"] for edge in per_edge]))
            ),
        },
        "velocity_error_mps": _quantiles(velocity_errors_np),
        "position_error_m": _quantiles(position_errors_np),
        "preintegration_linearization_accel_bias_mps2": _quantiles(nominal_accel_biases),
        "preintegration_linearization_gyro_bias_radps": _quantiles(nominal_gyro_biases),
        "interpretation": (
            "Velocity references use ground-truth velocity when present, otherwise "
            "finite-difference GT position. "
            "Use the signed means and cross-sequence consistency to identify a "
            "systematic convention error; do not treat this as an ATE metric."
        ),
    }
    if gt_accel_biases is not None and gt_gyro_biases is not None:
        summary["groundtruth_accel_bias_mps2"] = _quantiles(gt_accel_biases)
        summary["groundtruth_gyro_bias_radps"] = _quantiles(gt_gyro_biases)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "preintegration_audit_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        with (output_dir / "preintegration_audit_edges.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_edge[0]))
            writer.writeheader()
            writer.writerows(per_edge)

    print("[PREINTEGRATION-AUDIT]", json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="prepared EuRoC DROID session")
    parser.add_argument("--groundtruth", required=True, help="EuRoC state_groundtruth_estimate0/data.csv")
    parser.add_argument("--prior-name", default="imu_prior_fd_jacobian.csv")
    interval_group = parser.add_mutually_exclusive_group()
    interval_group.add_argument(
        "--selected-manifest",
        default=None,
        help=(
            "optional frozen MotionFilter CSV (frame_index,timestamp_ns). "
            "When supplied, compose exactly the raw priors between adjacent "
            "selected frames, matching the runtime full-IMU factor interval."
        ),
    )
    interval_group.add_argument(
        "--reconstruction",
        default=None,
        help=(
            "DROID reconstruction .pth. Final global-BA keyframes are mapped through "
            "--frames-csv and audited as the runtime factor intervals."
        ),
    )
    parser.add_argument(
        "--frames-csv",
        default=None,
        help="prepared frames.csv; required with --reconstruction",
    )
    parser.add_argument(
        "--apply-reconstruction-bias",
        action="store_true",
        help=(
            "apply the saved global and per-keyframe 15DoF bias states through "
            "the propagated IMU Jacobians before GT comparison; requires "
            "--reconstruction"
        ),
    )
    parser.add_argument("--gravity", type=float, nargs=3, default=(0.0, 0.0, -9.81))
    parser.add_argument("--max-gt-gap-ms", type=float, default=5.0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.reconstruction is not None and args.frames_csv is None:
        parser.error("--reconstruction requires --frames-csv")
    if args.apply_reconstruction_bias and args.reconstruction is None:
        parser.error("--apply-reconstruction-bias requires --reconstruction")
    return args


if __name__ == "__main__":
    audit(parse_args())

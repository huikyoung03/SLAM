#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from pathlib import Path

import numpy as np
import torch


TRAJ_KEYS = [
    "raw_n_poses",
    "n_poses",
    "path_len",
    "start_end_path",
    "step_p95",
    "rot_p95_deg",
    "turn_spread",
    "return_gap_median_path",
    "return_gap_p90",
]

GT_KEYS = [
    "gt_matches",
    "gt_pose_mode",
    "gt_frame",
    # A DROID reconstruction may be normalized after visual-inertial alignment.
    # These fields state whether the exported poses were converted back to
    # metres before metric quantities below were computed.
    "metric_scale_available",
    "pose_unit_scale",
    "metricization",
    # Legacy aliases retained for existing reports. They always refer to the
    # scale-corrected reconstruction comparison below.
    "ate_rmse_m",
    "ate_median_m",
    "ate_p90_m",
    "sim3_ate_rmse_m",
    "sim3_ate_median_m",
    "sim3_ate_p90_m",
    "sim3_scale",
    "metric_scale_error",
    "sim3_rpe_1s_pairs",
    "sim3_rpe_1s_trans_rmse_m",
    "sim3_rpe_1s_trans_median_m",
    "sim3_rpe_1s_rot_rmse_deg",
    "sim3_rpe_5s_pairs",
    "sim3_rpe_5s_trans_rmse_m",
    "sim3_rpe_5s_trans_median_m",
    "sim3_rpe_5s_rot_rmse_deg",
    "se3_ate_rmse_m",
    "se3_ate_median_m",
    "se3_ate_p90_m",
    "se3_rot_rmse_deg",
    "origin_ate_rmse_m",
    "origin_ate_median_m",
    "origin_ate_p90_m",
    "origin_rot_rmse_deg",
    "rpe_1s_pairs",
    "rpe_1s_trans_rmse_m",
    "rpe_1s_trans_median_m",
    "rpe_1s_rot_rmse_deg",
    "rpe_5s_pairs",
    "rpe_5s_trans_rmse_m",
    "rpe_5s_trans_median_m",
    "rpe_5s_rot_rmse_deg",
    "max_time_diff_ms",
]

IMU_KEYS = [
    "imu_rows",
    "imu_used",
    "imu_skip_dt",
    "imu_skip_rot",
    "imu_used_ratio",
    "imu_conf_mean",
    "imu_weight_mean",
    "imu_r_R_deg_mean",
    "imu_r_p_norm_mean",
    "imu_r_v_norm_mean",
    "imu_r_ba_norm_mean",
    "imu_r_bg_norm_mean",
]

VI_KEYS = [
    "vi_init_requested",
    "vi_init_source",
    "vi_init_edges",
    "vi_init_backend_fallback",
    "bias_refine_attempts",
    "bias_refine_updates",
]


def parse_run(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=RECON[,DEBUG]")

    label, rest = value.split("=", 1)
    label = label.strip()
    parts = [p.strip() for p in rest.split(",", 1)]

    if not label or not parts[0]:
        raise argparse.ArgumentTypeError("run must be LABEL=RECON[,DEBUG]")

    recon = parts[0]
    debug = parts[1] if len(parts) > 1 and parts[1] else ""
    return label, recon, debug


def q_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1e-12, None)


def quat_angle_deg(q1, q2):
    q1 = q_normalize(q1)
    q2 = q_normalize(q2)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot) * 180.0 / np.pi


def qxyzw_to_rotmat(q):
    q = q_normalize(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[..., 0, 1] = 2.0 * (x * y - z * w)
    R[..., 0, 2] = 2.0 * (x * z + y * w)
    R[..., 1, 0] = 2.0 * (x * y + z * w)
    R[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[..., 1, 2] = 2.0 * (y * z - x * w)
    R[..., 2, 0] = 2.0 * (x * z - y * w)
    R[..., 2, 1] = 2.0 * (y * z + x * w)
    R[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R


def rotation_angle_deg(R):
    """Return the angle of one or more SO(3) matrices in degrees."""
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def read_frames(path):
    frames = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frames[int(row["frame_id"])] = int(row["timestamp_ns"])
    return frames


def _normalise_column(name):
    name = name.lstrip("\ufeff#").lower()
    name = re.sub(r"\[[^\]]*\]", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _find_column(header, candidates, required=True):
    columns = {_normalise_column(name): index for index, name in enumerate(header)}
    for name in candidates:
        index = columns.get(_normalise_column(name))
        if index is not None:
            return index
    if required:
        raise ValueError(
            f"could not find one of {tuple(candidates)} in GT header={header}"
        )
    return None


def read_gt_poses(path):
    """Read EuRoC-style body poses, retaining orientation when available."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = csv.reader(f)
        header = next(rows, None)
        if not header:
            raise ValueError(f"ground-truth CSV is empty: {path}")

        ts_idx = _find_column(header, ("timestamp", "timestamp_ns"))
        px_idx = _find_column(header, ("p_RS_R_x", "px", "p_x"))
        py_idx = _find_column(header, ("p_RS_R_y", "py", "p_y"))
        pz_idx = _find_column(header, ("p_RS_R_z", "pz", "p_z"))
        qw_idx = _find_column(header, ("q_RS_w", "qw", "q_w"), required=False)
        qx_idx = _find_column(header, ("q_RS_x", "qx", "q_x"), required=False)
        qy_idx = _find_column(header, ("q_RS_y", "qy", "q_y"), required=False)
        qz_idx = _find_column(header, ("q_RS_z", "qz", "q_z"), required=False)

        have_rotation = all(index is not None for index in (qw_idx, qx_idx, qy_idx, qz_idx))
        timestamps, positions, quaternions = [], [], []
        for row in rows:
            if not row or row[0].startswith("#"):
                continue
            try:
                stamp = int(float(row[ts_idx]))
                position = [float(row[px_idx]), float(row[py_idx]), float(row[pz_idx])]
                if have_rotation:
                    quaternions.append(
                        [float(row[qx_idx]), float(row[qy_idx]), float(row[qz_idx]), float(row[qw_idx])]
                    )
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid GT row in {path}: {row}") from exc
            timestamps.append(stamp)
            positions.append(position)

    if not timestamps:
        raise ValueError(f"ground-truth CSV has no data rows: {path}")

    order = np.argsort(np.asarray(timestamps, dtype=np.int64))
    timestamps = np.asarray(timestamps, dtype=np.int64)[order]
    positions = np.asarray(positions, dtype=np.float64)[order]
    rotations = None
    if have_rotation:
        rotations = qxyzw_to_rotmat(np.asarray(quaternions, dtype=np.float64)[order])
    return timestamps, positions, rotations


def read_camera_to_body_transform(path):
    """Read EuRoC's T_BS transform, which maps camera coordinates into body."""
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"T_BS:.*?data:\s*\[([^\]]+)\]", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"could not find T_BS.data in {path}")
    values = [
        float(value)
        for value in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", match.group(1))
    ]
    if len(values) != 16:
        raise ValueError(f"T_BS in {path} must contain 16 values, got {len(values)}")
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def body_to_camera_ground_truth(positions, rotations, camera_to_body):
    """Convert EuRoC body/IMU GT into cam0 world poses."""
    if rotations is None:
        raise ValueError("body-frame GT requires orientation to apply camera extrinsics")
    R_bc = camera_to_body[:3, :3]
    t_bc = camera_to_body[:3, 3]
    camera_positions = positions + np.einsum("nij,j->ni", rotations, t_bc)
    camera_rotations = rotations @ R_bc
    return camera_positions, camera_rotations


def _scalar_from_blob(blob, key, default=np.nan):
    value = blob.get(key, default)
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1)
        if value.numel() == 0:
            return float(default)
        value = value[0].item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def reconstruction_metric_metadata(blob, auto_metricize_imu=True):
    """Describe the length unit stored in one reconstruction blob.

    DROID normalizes monocular pose/depth scale during backend optimization.
    Once VI alignment has succeeded, ``imu_unit_scale`` records the number of
    current DROID length units per metre.  It must be divided out of pose
    translations before a metric ATE/RPE claim is made.  A visual-only run has
    no such observable scale, so it remains eligible only for Sim(3) metrics.
    """

    aligned = int(_scalar_from_blob(blob, "imu_alignment_ready", 0.0) > 0.5)
    requested_value = _scalar_from_blob(blob, "imu_vi_initialization_requested")
    requested = int(requested_value > 0.5) if math.isfinite(requested_value) else None
    unit_scale = _scalar_from_blob(blob, "imu_unit_scale")
    usable = bool(aligned and np.isfinite(unit_scale) and unit_scale > 1e-12)
    applied = bool(usable and auto_metricize_imu)
    alignment_metrics = blob.get("imu_alignment_metrics", {})
    if not isinstance(alignment_metrics, dict):
        alignment_metrics = {}

    history = blob.get("imu_bias_refinement_history", [])
    if not isinstance(history, (list, tuple)):
        history = []
    attempts = int(_scalar_from_blob(
        blob, "imu_bias_refinement_attempts", float(len(history))
    ))
    updates = int(_scalar_from_blob(
        blob,
        "imu_bias_refinement_updates",
        float(sum(
            int(isinstance(entry, dict) and float(entry.get("updated", 0.0)) > 0.5)
            for entry in history
        )),
    ))

    source = blob.get("imu_alignment_source", "")
    if not isinstance(source, str) or not source:
        if not aligned:
            source = (
                "not_aligned" if requested == 1
                else "visual_only" if requested == 0
                else "unknown_nonmetric"
            )
        elif float(alignment_metrics.get("backend_fallback", 0.0)) > 0.5:
            source = "backend_fallback"
        elif float(alignment_metrics.get("history_snapshot", 0.0)) > 0.5:
            source = "history"
        else:
            source = "live"

    return {
        "metric_scale_available": int(applied),
        "pose_unit_scale": float(unit_scale) if usable else np.nan,
        "metricization": (
            "imu_alignment_unit_scale" if applied
            else "imu_alignment_not_applied" if usable
            else "monocular_scale_unobservable"
        ),
        "vi_init_requested": requested if requested is not None else np.nan,
        "vi_init_source": source,
        "vi_init_edges": int(float(alignment_metrics.get("edges", 0.0))),
        "vi_init_backend_fallback": int(
            float(alignment_metrics.get("backend_fallback", 0.0)) > 0.5
        ),
        "bias_refine_attempts": attempts,
        "bias_refine_updates": updates,
    }


def load_recon(path, auto_metricize_imu=True):
    blob = torch.load(path, map_location="cpu")
    poses = blob["poses"].detach().cpu().float().numpy()
    if "tstamps" in blob:
        tstamps = blob["tstamps"].detach().cpu().numpy()
    else:
        tstamps = np.arange(len(poses), dtype=np.float64)

    valid = np.isfinite(poses).all(axis=1)
    poses = poses[valid]
    tstamps = tstamps[valid]
    metadata = reconstruction_metric_metadata(blob, auto_metricize_imu)
    if metadata["metric_scale_available"] and auto_metricize_imu:
        # ``poses`` stores world-to-camera translations.  A global length-unit
        # conversion therefore acts directly on t_cw; rotations are unchanged.
        poses = poses.copy()
        poses[:, :3] /= metadata["pose_unit_scale"]
    return tstamps, poses, metadata


def pairwise_nn(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.nan, np.nan, np.nan

    dists = []
    for a in A:
        d = np.linalg.norm(B - a[None, :], axis=1)
        dists.append(np.min(d))

    dists = np.asarray(dists)
    return (
        float(np.mean(dists)),
        float(np.median(dists)),
        float(np.percentile(dists, 90)),
    )


def trajectory_metrics(tstamps, positions, rotations, turn_frame):
    if len(positions) < 5:
        return {key: np.nan for key in TRAJ_KEYS}

    positions = np.asarray(positions, dtype=np.float64)
    trans_step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if rotations is None:
        rot_step = None
    else:
        relative_rotations = np.transpose(rotations[:-1], (0, 2, 1)) @ rotations[1:]
        rot_step = rotation_angle_deg(relative_rotations)

    path_len = float(np.sum(trans_step))
    start_end = float(np.linalg.norm(positions[-1] - positions[0]))
    turn_idx = int(np.argmin(np.abs(tstamps - turn_frame)))

    before = positions[: max(0, turn_idx - 5)]
    after = positions[min(len(positions), turn_idx + 5) :]
    _, return_gap_median, return_gap_p90 = pairwise_nn(after, before)

    a = max(0, turn_idx - 10)
    b = min(len(positions), turn_idx + 10)
    turn_local = positions[a:b]
    turn_spread = (
        float(np.linalg.norm(np.ptp(turn_local, axis=0)))
        if len(turn_local) > 1
        else np.nan
    )

    return {
        "n_poses": int(len(positions)),
        "path_len": path_len,
        "start_end_path": start_end / path_len if path_len > 1e-12 else np.nan,
        "step_p95": float(np.percentile(trans_step, 95)),
        "rot_p95_deg": float(np.percentile(rot_step, 95)) if rot_step is not None else np.nan,
        "turn_spread": turn_spread,
        "return_gap_median_path": (
            return_gap_median / path_len if path_len > 1e-12 else np.nan
        ),
        "return_gap_p90": return_gap_p90,
    }


def nearest_gt_indices(frame_ids, frames, gt_ts, max_dt_ns):
    recon_indices = []
    gt_indices = []
    time_diffs = []

    for idx, frame_id in enumerate(frame_ids):
        frame_id = int(round(float(frame_id)))
        if frame_id not in frames:
            continue

        timestamp = frames[frame_id]
        pos = int(np.searchsorted(gt_ts, timestamp))
        candidates = []
        if pos < len(gt_ts):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        if not candidates:
            continue

        best = min(candidates, key=lambda k: abs(int(gt_ts[k]) - timestamp))
        dt = abs(int(gt_ts[best]) - timestamp)
        if dt <= max_dt_ns:
            recon_indices.append(idx)
            gt_indices.append(best)
            time_diffs.append(dt / 1e9)

    return (
        np.asarray(recon_indices, dtype=np.int64),
        np.asarray(gt_indices, dtype=np.int64),
        np.asarray(time_diffs, dtype=np.float64),
    )


def similarity_alignment(source, target, correct_scale):
    """Fit target ~= scale * R * source + t with optional scale correction."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean

    var = np.mean(np.sum(src_centered * src_centered, axis=1))
    cov = (tgt_centered.T @ src_centered) / len(source)
    U, singular, Vt = np.linalg.svd(cov)

    sign = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        sign[-1, -1] = -1.0

    R = U @ sign @ Vt
    if correct_scale:
        scale = np.trace(np.diag(singular) @ sign) / max(var, 1e-12)
    else:
        scale = 1.0
    trans = tgt_mean - scale * R @ src_mean
    return R, trans, float(scale)


def apply_alignment(positions, rotations, R, trans, scale=1.0):
    aligned_positions = (scale * (R @ positions.T)).T + trans
    aligned_rotations = None if rotations is None else R[None] @ rotations
    return aligned_positions, aligned_rotations


def sim3_align(source, target):
    R, trans, scale = similarity_alignment(source, target, correct_scale=True)
    aligned, _ = apply_alignment(source, None, R, trans, scale)
    return aligned, scale


def estimate_camera_poses(poses, mode):
    """Recover camera centers and world-frame camera rotations from DROID poses."""
    pose_t = np.asarray(poses[:, :3], dtype=np.float64)
    R_cw = qxyzw_to_rotmat(poses[:, 3:7])

    if mode == "pose_t":
        return pose_t, R_cw, "pose_t"

    # DROID stores a world-to-camera transform.  Camera-center coordinates and
    # camera-to-world orientation are the physical trajectory representation.
    center = -(np.transpose(R_cw, (0, 2, 1)) @ pose_t[..., None])[..., 0]
    return center, np.transpose(R_cw, (0, 2, 1)), "inv_center"


def _error_summary(estimate, target):
    error = np.linalg.norm(estimate - target, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "median": float(np.median(error)),
        "p90": float(np.percentile(error, 90)),
    }


def _orientation_rmse(estimate, target):
    if estimate is None or target is None:
        return np.nan
    error_rotations = np.transpose(target, (0, 2, 1)) @ estimate
    error = rotation_angle_deg(error_rotations)
    return float(np.sqrt(np.mean(error * error)))


def _rpe_pair_indices(timestamps_sec, delta_sec):
    pairs = []
    tolerance = max(0.02, 0.10 * float(delta_sec))
    for i, timestamp in enumerate(timestamps_sec):
        wanted = timestamp + float(delta_sec)
        j = int(np.searchsorted(timestamps_sec, wanted))
        candidates = [candidate for candidate in (j - 1, j) if i < candidate < len(timestamps_sec)]
        if not candidates:
            continue
        best = min(candidates, key=lambda candidate: abs(timestamps_sec[candidate] - wanted))
        if abs(timestamps_sec[best] - wanted) <= tolerance:
            pairs.append((i, best))
    return pairs


def rpe_metrics(
    estimate_positions,
    estimate_rotations,
    target_positions,
    target_rotations,
    timestamps_sec,
    delta_sec,
    prefix=None,
):
    prefix = prefix or f"rpe_{int(delta_sec)}s"
    out = {
        f"{prefix}_pairs": 0,
        f"{prefix}_trans_rmse_m": np.nan,
        f"{prefix}_trans_median_m": np.nan,
        f"{prefix}_rot_rmse_deg": np.nan,
    }
    if estimate_rotations is None or target_rotations is None:
        return out

    pairs = _rpe_pair_indices(timestamps_sec, delta_sec)
    if not pairs:
        return out

    trans_errors, rot_errors = [], []
    for i, j in pairs:
        R_est_rel = estimate_rotations[i].T @ estimate_rotations[j]
        t_est_rel = estimate_rotations[i].T @ (estimate_positions[j] - estimate_positions[i])
        R_gt_rel = target_rotations[i].T @ target_rotations[j]
        t_gt_rel = target_rotations[i].T @ (target_positions[j] - target_positions[i])
        R_error = R_gt_rel.T @ R_est_rel
        t_error = R_gt_rel.T @ (t_est_rel - t_gt_rel)
        trans_errors.append(float(np.linalg.norm(t_error)))
        rot_errors.append(float(rotation_angle_deg(R_error)))

    trans_errors = np.asarray(trans_errors, dtype=np.float64)
    rot_errors = np.asarray(rot_errors, dtype=np.float64)
    out.update(
        {
            f"{prefix}_pairs": int(len(pairs)),
            f"{prefix}_trans_rmse_m": float(np.sqrt(np.mean(trans_errors * trans_errors))),
            f"{prefix}_trans_median_m": float(np.median(trans_errors)),
            f"{prefix}_rot_rmse_deg": float(np.sqrt(np.mean(rot_errors * rot_errors))),
        }
    )
    return out


def ate_metrics(
    tstamps,
    poses,
    frames,
    gt_ts,
    gt_pos,
    gt_rot,
    max_dt_ns,
    mode,
    gt_frame,
    reconstruction_metadata,
):
    recon_idx, gt_idx, time_diffs = nearest_gt_indices(
        tstamps, frames, gt_ts, max_dt_ns
    )

    if len(recon_idx) < 5:
        out = {key: np.nan for key in GT_KEYS}
        out["gt_matches"] = int(len(recon_idx))
        out["gt_pose_mode"] = "none"
        out["gt_frame"] = gt_frame
        out.update(reconstruction_metadata)
        return out

    source, source_rot, pose_mode = estimate_camera_poses(poses[recon_idx], mode)
    target = gt_pos[gt_idx]
    target_rot = None if gt_rot is None else gt_rot[gt_idx]

    R_sim3, t_sim3, sim3_scale = similarity_alignment(source, target, correct_scale=True)
    sim3_positions, sim3_rotations = apply_alignment(source, source_rot, R_sim3, t_sim3, sim3_scale)
    sim3_error = _error_summary(sim3_positions, target)

    timestamps_sec = np.asarray(
        [frames[int(round(float(frame_id)))] / 1.0e9 for frame_id in tstamps[recon_idx]],
        dtype=np.float64,
    )
    sim3_rpe_1s = rpe_metrics(
        sim3_positions,
        sim3_rotations,
        target,
        target_rot,
        timestamps_sec,
        1.0,
        prefix="sim3_rpe_1s",
    )
    sim3_rpe_5s = rpe_metrics(
        sim3_positions,
        sim3_rotations,
        target,
        target_rot,
        timestamps_sec,
        5.0,
        prefix="sim3_rpe_5s",
    )

    metric_available = bool(reconstruction_metadata["metric_scale_available"])
    se3_error = {"rmse": np.nan, "median": np.nan, "p90": np.nan}
    origin_error = {"rmse": np.nan, "median": np.nan, "p90": np.nan}
    se3_rot_rmse = np.nan
    origin_rot_rmse = np.nan
    metric_rpe_1s = rpe_metrics(None, None, target, target_rot, timestamps_sec, 1.0)
    metric_rpe_5s = rpe_metrics(None, None, target, target_rot, timestamps_sec, 5.0)
    metric_scale_error = np.nan

    if metric_available:
        R_se3, t_se3, _ = similarity_alignment(source, target, correct_scale=False)
        se3_positions, se3_rotations = apply_alignment(source, source_rot, R_se3, t_se3)
        se3_error = _error_summary(se3_positions, target)
        se3_rot_rmse = _orientation_rmse(se3_rotations, target_rot)
        metric_scale_error = abs(sim3_scale - 1.0)
        metric_rpe_1s = rpe_metrics(
            se3_positions,
            se3_rotations,
            target,
            target_rot,
            timestamps_sec,
            1.0,
        )
        metric_rpe_5s = rpe_metrics(
            se3_positions,
            se3_rotations,
            target,
            target_rot,
            timestamps_sec,
            5.0,
        )

        if source_rot is not None and target_rot is not None:
            R_origin = target_rot[0] @ source_rot[0].T
            t_origin = target[0] - R_origin @ source[0]
            origin_positions, origin_rotations = apply_alignment(
                source,
                source_rot,
                R_origin,
                t_origin,
            )
            origin_error = _error_summary(origin_positions, target)
            origin_rot_rmse = _orientation_rmse(origin_rotations, target_rot)

    row = {
        "gt_matches": int(len(recon_idx)),
        "gt_pose_mode": pose_mode,
        "gt_frame": gt_frame,
        **reconstruction_metadata,
        # Compatibility aliases: this is deliberately the Sim(3) result only.
        "ate_rmse_m": sim3_error["rmse"],
        "ate_median_m": sim3_error["median"],
        "ate_p90_m": sim3_error["p90"],
        "sim3_ate_rmse_m": sim3_error["rmse"],
        "sim3_ate_median_m": sim3_error["median"],
        "sim3_ate_p90_m": sim3_error["p90"],
        "sim3_scale": sim3_scale,
        "metric_scale_error": metric_scale_error,
        "se3_ate_rmse_m": se3_error["rmse"],
        "se3_ate_median_m": se3_error["median"],
        "se3_ate_p90_m": se3_error["p90"],
        "se3_rot_rmse_deg": se3_rot_rmse,
        "origin_ate_rmse_m": origin_error["rmse"],
        "origin_ate_median_m": origin_error["median"],
        "origin_ate_p90_m": origin_error["p90"],
        "origin_rot_rmse_deg": origin_rot_rmse,
        "max_time_diff_ms": float(1000.0 * np.max(time_diffs)),
    }
    row.update(sim3_rpe_1s)
    row.update(sim3_rpe_5s)
    row.update(metric_rpe_1s)
    row.update(metric_rpe_5s)
    return row


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if math.isfinite(value) else np.nan


def mean_field(rows, field, used_only=True):
    values = []
    for row in rows:
        if used_only and row.get("reason") != "used":
            continue
        value = finite_float(row.get(field))
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else np.nan


def imu_debug_metrics(path):
    out = {key: np.nan for key in IMU_KEYS}
    out.update({"imu_rows": 0, "imu_used": 0, "imu_skip_dt": 0, "imu_skip_rot": 0})

    if not path:
        return out

    debug_path = Path(path)
    if not debug_path.exists():
        return out

    with debug_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    reasons = {}
    for row in rows:
        reason = row.get("reason", "")
        reasons[reason] = reasons.get(reason, 0) + 1

    used = reasons.get("used", 0)
    out.update(
        {
            "imu_rows": int(len(rows)),
            "imu_used": int(used),
            "imu_skip_dt": int(reasons.get("skip_dt", 0)),
            "imu_skip_rot": int(reasons.get("skip_rot", 0)),
            "imu_used_ratio": float(used / len(rows)) if rows else np.nan,
            "imu_conf_mean": mean_field(rows, "confidence"),
            "imu_weight_mean": mean_field(rows, "weight"),
            "imu_r_R_deg_mean": mean_field(rows, "r_R_deg"),
            "imu_r_p_norm_mean": mean_field(rows, "r_p_norm"),
            "imu_r_v_norm_mean": mean_field(rows, "r_v_norm"),
            "imu_r_ba_norm_mean": mean_field(rows, "r_ba_norm"),
            "imu_r_bg_norm_mean": mean_field(rows, "r_bg_norm"),
        }
    )
    return out


def evaluate_run(label, recon_path, debug_path, args, frames, gt_ts, gt_pos, gt_rot):
    tstamps, poses, reconstruction_metadata = load_recon(
        recon_path,
        auto_metricize_imu=args.auto_metricize_imu_recon,
    )
    return evaluate_loaded_run(
        label,
        recon_path,
        debug_path,
        tstamps,
        poses,
        args,
        frames,
        gt_ts,
        gt_pos,
        gt_rot,
        reconstruction_metadata,
    )


def evaluate_loaded_run(
    label,
    recon_path,
    debug_path,
    tstamps,
    poses,
    args,
    frames,
    gt_ts,
    gt_pos,
    gt_rot,
    reconstruction_metadata,
    raw_n_poses=None,
):
    row = {
        "label": label,
        "recon": recon_path,
        "debug": debug_path,
        "raw_n_poses": int(len(poses)) if raw_n_poses is None else int(raw_n_poses),
    }
    trajectory_positions, trajectory_rotations, _ = estimate_camera_poses(poses, args.pose_mode)
    row.update(
        trajectory_metrics(
            tstamps,
            trajectory_positions,
            trajectory_rotations,
            args.turn_frame,
        )
    )
    row.update(
        ate_metrics(
            tstamps,
            poses,
            frames,
            gt_ts,
            gt_pos,
            gt_rot,
            int(args.max_time_diff_ms * 1e6),
            args.pose_mode,
            args.gt_frame,
            reconstruction_metadata,
        )
    )
    row.update(imu_debug_metrics(debug_path))
    return row


def common_reconstruction_frame_ids(loaded_runs, frames):
    """Return input frame IDs shared by every reconstruction in a comparison.

    A tracker can expose a slightly different set of valid keyframes per run.
    Comparing separately aligned ATE values over those different sets can make a
    one-frame bookkeeping difference look like an IMU improvement or regression.
    Frame IDs are stored as float tensors in reconstruction files, but originate
    from integer input-image indices.
    """

    common = set(frames)
    for _, _, _, tstamps, _, _ in loaded_runs:
        ids = np.rint(np.asarray(tstamps, dtype=np.float64)).astype(np.int64)
        common.intersection_update(int(frame_id) for frame_id in ids)

    return common


def filter_reconstruction_to_frame_ids(tstamps, poses, frame_ids):
    """Keep one pose per requested input frame while preserving trajectory order."""

    ids = np.rint(np.asarray(tstamps, dtype=np.float64)).astype(np.int64)
    keep = []
    seen = set()
    for index, frame_id in enumerate(ids):
        frame_id = int(frame_id)
        if frame_id in frame_ids and frame_id not in seen:
            keep.append(index)
            seen.add(frame_id)

    keep = np.asarray(keep, dtype=np.int64)
    return tstamps[keep], poses[keep]


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def write_csv(path, rows):
    keys = ["label", "recon", "debug"] + TRAJ_KEYS + GT_KEYS + VI_KEYS + IMU_KEYS
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_markdown(path, rows, common_frame_ids=False, common_count=None):
    keys = [
        "label",
        "raw_n_poses",
        "n_poses",
        "metricization",
        "pose_unit_scale",
        "vi_init_requested",
        "vi_init_source",
        "vi_init_edges",
        "bias_refine_attempts",
        "bias_refine_updates",
        "sim3_ate_rmse_m",
        "sim3_rpe_1s_trans_rmse_m",
        "sim3_rpe_5s_trans_rmse_m",
        "se3_ate_rmse_m",
        "origin_ate_rmse_m",
        "metric_scale_error",
        "rpe_1s_trans_rmse_m",
        "rpe_5s_trans_rmse_m",
        "rpe_1s_rot_rmse_deg",
        "rpe_5s_rot_rmse_deg",
        "gt_pose_mode",
        "start_end_path",
        "rot_p95_deg",
        "return_gap_median_path",
        "imu_used",
        "imu_used_ratio",
        "imu_conf_mean",
        "imu_weight_mean",
    ]

    lines = [
        "# EuRoC Ablation Summary",
        "",
        "Sim(3) ATE is a scale-corrected reconstruction-shape metric only.",
        "Sim(3) RPE compares local drift after an oracle global scale fit.",
        "Metric SE(3)/origin ATE and metric RPE are reported only when the run",
        "contains a successful VI alignment and its DROID unit scale is converted to metres.",
    ]

    if common_frame_ids:
        lines.append(
            "Runs are compared on the same "
            f"{common_count} reconstruction frame IDs before alignment."
        )

    lines.extend(
        [
            "",
            "| " + " | ".join(keys) + " |",
            "| " + " | ".join(["---"] + ["---:"] * (len(keys) - 1)) + " |",
        ]
    )

    for row in rows:
        values = [fmt(row.get(key, "")) for key in keys]
        lines.append("| " + " | ".join(values) + " |")

    best = [
        row
        for row in rows
        if isinstance(row.get("se3_ate_rmse_m"), float) and math.isfinite(row["se3_ate_rmse_m"])
    ]
    if best:
        best_row = min(best, key=lambda row: row["se3_ate_rmse_m"])
        lines.extend(
            [
                "",
                f"Lowest metric SE(3) ATE: `{best_row['label']}` = "
                f"{best_row['se3_ate_rmse_m']:.6f} m",
            ]
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(rows):
    keys = [
        "label",
        "metricization",
        "sim3_ate_rmse_m",
        "sim3_rpe_1s_trans_rmse_m",
        "se3_ate_rmse_m",
        "origin_ate_rmse_m",
        "rpe_1s_trans_rmse_m",
        "n_poses",
        "imu_used_ratio",
    ]
    print(" | ".join(keys))
    print("-" * 86)
    for row in rows:
        print(" | ".join(fmt(row.get(key, "")) for key in keys))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-csv", required=True)
    parser.add_argument("--gt-csv", required=True)
    parser.add_argument("--turn-frame", type=int, default=223)
    parser.add_argument(
        "--pose-mode",
        choices=["auto", "pose_t", "inv_center"],
        default="inv_center",
        help="DROID pose convention; inv_center is the physical camera-center trajectory",
    )
    parser.add_argument(
        "--gt-frame",
        choices=["camera", "body"],
        default="camera",
        help="whether --gt-csv poses are already camera poses or EuRoC body/IMU poses",
    )
    parser.add_argument(
        "--camera-sensor-yaml",
        default="",
        help="EuRoC cam0 sensor.yaml; required with --gt-frame body",
    )
    parser.add_argument("--max-time-diff-ms", type=float, default=30.0)
    parser.add_argument(
        "--no-auto-imu-metricize",
        dest="auto_metricize_imu_recon",
        action="store_false",
        help=(
            "debug only: do not divide aligned reconstruction translations by "
            "the exported imu_unit_scale"
        ),
    )
    parser.set_defaults(auto_metricize_imu_recon=True)
    parser.add_argument(
        "--common-frame-ids",
        action="store_true",
        help=(
            "evaluate every run on the intersection of valid reconstruction "
            "input-frame IDs"
        ),
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", default="")
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    args = parser.parse_args()

    frames = read_frames(args.frames_csv)
    gt_ts, gt_pos, gt_rot = read_gt_poses(args.gt_csv)
    if args.gt_frame == "body":
        if not args.camera_sensor_yaml:
            raise ValueError("--gt-frame body requires --camera-sensor-yaml")
        camera_to_body = read_camera_to_body_transform(args.camera_sensor_yaml)
        gt_pos, gt_rot = body_to_camera_ground_truth(gt_pos, gt_rot, camera_to_body)

    common_count = None
    if args.common_frame_ids:
        loaded_runs = [
            (
                label,
                recon,
                debug,
                *load_recon(
                    recon,
                    auto_metricize_imu=args.auto_metricize_imu_recon,
                ),
            )
            for label, recon, debug in args.run
        ]
        common_ids = common_reconstruction_frame_ids(loaded_runs, frames)
        common_count = len(common_ids)
        if common_count < 5:
            raise ValueError(
                "fewer than five common reconstruction frame IDs are available "
                f"across runs: {common_count}"
            )

        print(f"[EVAL] using {common_count} common reconstruction frame IDs")
        rows = []
        for label, recon, debug, tstamps, poses, reconstruction_metadata in loaded_runs:
            raw_n_poses = len(poses)
            tstamps, poses = filter_reconstruction_to_frame_ids(
                tstamps,
                poses,
                common_ids,
            )
            rows.append(
                evaluate_loaded_run(
                    label,
                    recon,
                    debug,
                    tstamps,
                    poses,
                    args,
                    frames,
                    gt_ts,
                    gt_pos,
                    gt_rot,
                    reconstruction_metadata,
                    raw_n_poses=raw_n_poses,
                )
            )
    else:
        rows = []
        for label, recon, debug in args.run:
            rows.append(evaluate_run(label, recon, debug, args, frames, gt_ts, gt_pos, gt_rot))

    write_csv(args.output_csv, rows)
    if args.output_md:
        write_markdown(
            args.output_md,
            rows,
            common_frame_ids=args.common_frame_ids,
            common_count=common_count,
        )

    print_table(rows)
    print(f"wrote {args.output_csv}")
    if args.output_md:
        print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()

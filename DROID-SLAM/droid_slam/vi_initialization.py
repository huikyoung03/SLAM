"""Visual-inertial alignment and fixed-bias refinement helpers.

The runtime DBA state stays deliberately small: pose, velocity and dense
inverse depth are optimized per keyframe.  This module estimates the global
metric quantities that should not be duplicated at every keyframe:

* visual-to-metric scale,
* gravity expressed in the visual world frame,
* one accelerometer bias and one gyroscope bias.

The initializer is intentionally outside the CUDA DBA loop.  It runs after a
short visual-only warm-up, metricizes the current visual state, then lets the
existing 9D pose/velocity CUDA path consume the resulting fixed parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from geom.imu_factor import (
    camera_center_from_pose_data,
    imu_center_from_pose_data,
    predict_imu_rotation,
    quat_inverse,
    quat_multiply,
    quat_rotate,
    quat_to_rotmat,
    quat_to_rotvec,
    rotvec_to_quat,
    select_imu_bias_jacobian,
    skew_symmetric,
    so3_left_jacobian,
    so3_left_jacobian_inv,
)


GRAVITY_MPS2 = 9.81


@dataclass
class VIAlignmentResult:
    """Result of one visual-inertial alignment attempt."""

    success: bool
    reason: str
    scale: float = 0.0
    gravity_internal: Optional[torch.Tensor] = None
    velocities_internal: Optional[torch.Tensor] = None
    velocity_indices: Optional[torch.Tensor] = None
    accel_bias_metric: Optional[torch.Tensor] = None
    gyro_bias_metric: Optional[torch.Tensor] = None
    metrics: Optional[Dict[str, float]] = None


@dataclass
class _SnapshotCounter:
    """Small ``multiprocessing.Value``-compatible counter for a snapshot."""

    value: int


@dataclass
class VIInitializationSnapshot:
    """A bounded, immutable visual-IMU window retained for VI initialization.

    The frontend can delete redundant live keyframes before a later motion
    segment makes scale and gravity observable.  Keeping a few compact
    snapshots lets the initializer retry a previously well-conditioned
    window without reintroducing deleted frames into the active DBA state.
    """

    poses: torch.Tensor
    imu_delta: torch.Tensor
    imu_valid: torch.Tensor
    imu_bias_jacobian: torch.Tensor
    imu_unit_scale: torch.Tensor
    imu_camera_translation: torch.Tensor
    tstamp: torch.Tensor
    counter: _SnapshotCounter
    imu_alignment_ready: bool = False


@dataclass
class _AlignmentLayout:
    """Column layout for one sparse temporal-edge alignment system."""

    frame_ids: np.ndarray
    velocity_dim: int
    gravity_col: Optional[int]
    scale_col: int
    bias_col: int
    total_dim: int


def _as_vector(value, device, dtype) -> torch.Tensor:
    if value is None:
        return torch.zeros(3, device=device, dtype=dtype)
    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if tensor.numel() < 3:
        return torch.zeros(3, device=device, dtype=dtype)
    return tensor[:3]


def _alignment_ready(video) -> bool:
    ready = getattr(video, "imu_alignment_ready", None)
    if ready is None:
        return False
    if hasattr(ready, "value"):
        return bool(ready.value)
    if torch.is_tensor(ready):
        return bool(ready.detach().reshape(-1)[0].cpu().item())
    return bool(ready)


def _set_alignment_ready(video, value: bool) -> None:
    ready = getattr(video, "imu_alignment_ready", None)
    if ready is None:
        return
    if hasattr(ready, "value"):
        ready.value = int(bool(value))
    elif torch.is_tensor(ready):
        ready.fill_(int(bool(value)))


@torch.no_grad()
def capture_vi_initialization_snapshot(video, window: int) -> Optional[VIInitializationSnapshot]:
    """Copy the latest pre-alignment window before frontend keyframe removal.

    Only pose and IMU quantities needed by the linear alignment solve are
    copied.  The first local row has no predecessor inside the snapshot, so
    its IMU factor is deliberately invalidated.
    """

    count = int(video.counter.value)
    if count < 2:
        return None

    end = count
    start = max(0, end - max(2, int(window)))
    size = end - start
    if size < 2:
        return None

    device = video.poses.device
    dtype = video.poses.dtype
    jacobian = getattr(video, "imu_bias_jacobian", None)
    if jacobian is None:
        jacobian_slice = torch.zeros(size, 5, 3, 3, device=device, dtype=dtype)
    else:
        jacobian_slice = jacobian[start:end].detach().clone()

    camera_translation = getattr(video, "imu_camera_translation", None)
    if camera_translation is None:
        camera_translation = torch.zeros(3, device=device, dtype=dtype)

    snapshot = VIInitializationSnapshot(
        poses=video.poses[start:end].detach().clone(),
        imu_delta=video.imu_delta[start:end].detach().clone(),
        imu_valid=video.imu_valid[start:end].detach().clone(),
        imu_bias_jacobian=jacobian_slice,
        imu_unit_scale=video.imu_unit_scale.detach().clone(),
        imu_camera_translation=torch.as_tensor(
            camera_translation, device=device, dtype=dtype
        ).detach().clone(),
        tstamp=video.tstamp[start:end].detach().clone(),
        counter=_SnapshotCounter(size),
    )
    snapshot.imu_delta[0].zero_()
    snapshot.imu_valid[0] = False
    snapshot.imu_bias_jacobian[0].zero_()
    return snapshot


def _valid_temporal_edges(video, start, end, max_dt) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return consecutive valid edges in `[start, end)` as `(src, dst)`."""

    device = video.poses.device
    if end - start < 2:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty

    dst = torch.arange(max(start + 1, 1), end, device=device, dtype=torch.long)
    src = dst - 1
    valid = video.imu_valid[dst].bool()
    dt = video.imu_delta[dst, 0]
    valid &= torch.isfinite(dt) & (dt > 1e-6)
    if max_dt > 0.0:
        valid &= dt <= float(max_dt)
    return src[valid], dst[valid]


@torch.no_grad()
def vi_initialization_observability(
    video,
    min_frames: int,
    window: int,
    max_dt: float,
    min_edges: int,
    min_duration: float,
    min_rotation_deg: float,
    min_visual_span_ratio: float,
) -> Dict[str, float]:
    """Score whether a live or retained window is worth a full VI solve."""

    count = int(video.counter.value)
    metrics: Dict[str, float] = {
        "candidate_frames": float(count),
        "candidate_edges": 0.0,
        "candidate_duration_sec": 0.0,
        "imu_rotation_excitation_deg": 0.0,
        "visual_span_ratio": 0.0,
        "eligible": 0.0,
        "score": 0.0,
    }
    if count < int(min_frames):
        metrics["reason"] = "insufficient_frames"
        return metrics

    start = max(0, count - max(int(window), int(min_frames)))
    src, dst = _valid_temporal_edges(video, start, count, max_dt)
    edge_count = int(dst.numel())
    duration = float(video.imu_delta[dst, 0].sum().detach().cpu().item())
    rotation_deg = float(
        (torch.linalg.norm(video.imu_delta[dst, 1:4], dim=-1).sum() * 180.0 / np.pi)
        .detach()
        .cpu()
        .item()
    )
    metrics.update(
        {
            "candidate_edges": float(edge_count),
            "candidate_duration_sec": duration,
            "imu_rotation_excitation_deg": rotation_deg,
        }
    )
    if edge_count < max(8, int(min_edges)):
        metrics["reason"] = "insufficient_imu_edges"
        return metrics
    if min_duration > 0.0 and duration < float(min_duration):
        metrics["reason"] = "insufficient_imu_duration"
        return metrics
    if min_rotation_deg > 0.0 and rotation_deg < float(min_rotation_deg):
        metrics["reason"] = "insufficient_imu_rotation"
        return metrics

    centers = camera_center_from_pose_data(video.poses[:count])
    used_frame_ids = torch.unique(torch.cat([src, dst]))
    visual_points = centers[used_frame_ids]
    visual_span = float(torch.linalg.vector_norm(
        visual_points.amax(dim=0) - visual_points.amin(dim=0)
    ).detach().cpu().item())
    edge_lengths = torch.linalg.vector_norm(centers[dst] - centers[src], dim=-1)
    median_edge = float(edge_lengths.median().detach().cpu().item()) if edge_lengths.numel() else 0.0
    span_ratio = visual_span / max(median_edge, 1e-12)
    metrics.update(
        {
            "visual_span_internal": visual_span,
            "median_visual_edge_internal": median_edge,
            "visual_span_ratio": span_ratio,
        }
    )
    if median_edge <= 1e-8 or span_ratio < float(min_visual_span_ratio):
        metrics["reason"] = "insufficient_visual_excitation"
        return metrics

    # Cap each contribution so an extreme but brittle segment cannot crowd
    # out every other observable candidate.
    score = (
        min(edge_count / max(float(min_edges), 1.0), 3.0)
        + min(duration / max(float(min_duration), 1e-6), 3.0)
        + min(rotation_deg / max(float(min_rotation_deg), 1e-6), 3.0)
        + min(span_ratio / max(float(min_visual_span_ratio), 1e-6), 3.0)
    )
    metrics.update({"eligible": 1.0, "score": score, "reason": "candidate"})
    return metrics


def rebase_vi_alignment_result(
    result: VIAlignmentResult,
    snapshot: VIInitializationSnapshot,
    video,
) -> VIAlignmentResult:
    """Express a snapshot solve in the current DROID visual-length unit.

    DROID may normalize its internal length unit after the snapshot was
    captured.  Biases remain metric quantities, while scale, gravity and
    velocities must be multiplied by ``current_unit / snapshot_unit`` before
    committing the alignment to the live video buffer.
    """

    if not result.success:
        return result

    source_scale = float(snapshot.imu_unit_scale.detach().reshape(-1)[0].cpu().item())
    target_scale = float(video.imu_unit_scale.detach().reshape(-1)[0].cpu().item())
    if not (
        np.isfinite(source_scale)
        and np.isfinite(target_scale)
        and source_scale > 1e-12
        and target_scale > 1e-12
    ):
        return result

    ratio = target_scale / source_scale
    result.scale *= ratio
    if result.gravity_internal is not None:
        result.gravity_internal = result.gravity_internal * ratio

    velocities = result.velocities_internal
    indices = result.velocity_indices
    if velocities is not None and indices is not None and snapshot.tstamp.numel() > 0:
        live_count = int(video.counter.value)
        live_times = video.tstamp[:live_count].detach().cpu().tolist()
        live_index = {int(round(float(value))): ix for ix, value in enumerate(live_times)}
        snapshot_times = snapshot.tstamp.detach().cpu().tolist()
        source_indices = indices.detach().cpu().tolist()
        keep_rows = []
        target_indices = []
        for row, source_index in enumerate(source_indices):
            source_index = int(source_index)
            if source_index < 0 or source_index >= len(snapshot_times):
                continue
            matched = live_index.get(int(round(float(snapshot_times[source_index]))))
            if matched is not None:
                keep_rows.append(row)
                target_indices.append(matched)
        if keep_rows:
            keep = torch.as_tensor(keep_rows, device=velocities.device, dtype=torch.long)
            result.velocities_internal = velocities[keep] * ratio
            result.velocity_indices = torch.as_tensor(
                target_indices, device=indices.device, dtype=torch.long
            )
        else:
            result.velocities_internal = None
            result.velocity_indices = None
    elif velocities is not None:
        result.velocities_internal = velocities * ratio

    metrics = dict(result.metrics or {})
    for key in (
        "gravity_norm_internal",
        "residual_rms_internal",
        "visual_span_internal",
        "median_visual_edge_internal",
    ):
        if key in metrics and np.isfinite(metrics[key]):
            metrics[key] *= ratio
    metrics.update(
        {
            "history_snapshot": 1.0,
            "snapshot_unit_scale": source_scale,
            "live_unit_scale": target_scale,
            "snapshot_to_live_scale": ratio,
            "scale": result.scale,
        }
    )
    result.metrics = metrics
    return result


def _edge_bias_jacobian(video, dst: torch.Tensor) -> torch.Tensor:
    """Select propagated bias Jacobians with the established fallback."""

    if dst.numel() == 0:
        return torch.empty(
            0,
            5,
            3,
            3,
            device=video.poses.device,
            dtype=video.poses.dtype,
        )

    dt = video.imu_delta[dst, 0:1].view(1, -1, 1)
    return select_imu_bias_jacobian(
        getattr(video, "imu_bias_jacobian", None),
        dst,
        batch=1,
        num_frames=int(video.counter.value),
        device=video.poses.device,
        dtype=video.poses.dtype,
        dt=dt,
    )[0]


def _rotation_residuals(
    video,
    src: torch.Tensor,
    dst: torch.Tensor,
    gyro_bias: torch.Tensor,
    compose_order: str,
    inverse_imu: bool,
) -> torch.Tensor:
    """Current visual-versus-IMU rotation residual for selected edges."""

    poses = video.poses
    jacobian = _edge_bias_jacobian(video, dst)
    q_i = poses[src, 3:7]
    q_j = poses[dst, 3:7]
    rotvec = video.imu_delta[dst, 1:4] + torch.matmul(
        jacobian[:, 0], gyro_bias.view(1, 3, 1)
    ).squeeze(-1)
    q_pred = predict_imu_rotation(
        q_i,
        rotvec_to_quat(rotvec),
        compose_order=compose_order,
        inverse_imu=inverse_imu,
    )
    return quat_to_rotvec(quat_multiply(quat_inverse(q_pred), q_j))


def estimate_gyro_bias(
    video,
    src: torch.Tensor,
    dst: torch.Tensor,
    initial_bias: Optional[torch.Tensor],
    compose_order: str,
    inverse_imu: bool,
    max_rotation_deg: float = 30.0,
    damping: float = 1e-5,
    iterations: int = 3,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Estimate one shared gyro bias by robust Gauss-Newton on `r_R`."""

    if dst.numel() < 3:
        return None, {"gyro_edges": float(dst.numel())}

    device = video.poses.device
    dtype = video.poses.dtype
    bias = _as_vector(initial_bias, device, dtype).clone()
    eps = 1e-4
    max_rotation = np.deg2rad(max(float(max_rotation_deg), 1e-3))
    accepted_edges = 0

    for _ in range(max(1, int(iterations))):
        residual = _rotation_residuals(
            video, src, dst, bias, compose_order, inverse_imu
        )
        norms = torch.linalg.norm(residual, dim=-1)
        mask = torch.isfinite(norms) & (norms <= max_rotation)
        if int(mask.sum().item()) < 3:
            return None, {"gyro_edges": float(mask.sum().item())}

        residual = residual[mask]
        src_used = src[mask]
        dst_used = dst[mask]
        norms = norms[mask]
        accepted_edges = int(mask.sum().item())

        jac_cols = []
        for axis in range(3):
            delta = torch.zeros_like(bias)
            delta[axis] = eps
            plus = _rotation_residuals(
                video,
                src_used,
                dst_used,
                bias + delta,
                compose_order,
                inverse_imu,
            )
            minus = _rotation_residuals(
                video,
                src_used,
                dst_used,
                bias - delta,
                compose_order,
                inverse_imu,
            )
            jac_cols.append(((plus - minus) / (2.0 * eps)).reshape(-1))

        jacobian = torch.stack(jac_cols, dim=-1)
        robust = (max_rotation / norms.clamp_min(max_rotation)).clamp(max=1.0)
        sqrt_weight = robust.sqrt().repeat_interleave(3)
        weighted_jacobian = jacobian * sqrt_weight[:, None]
        weighted_residual = residual.reshape(-1) * sqrt_weight
        hessian = weighted_jacobian.T @ weighted_jacobian
        hessian += float(damping) * torch.eye(3, device=device, dtype=dtype)
        gradient = weighted_jacobian.T @ weighted_residual
        try:
            step = torch.linalg.solve(hessian, -gradient)
        except RuntimeError:
            return None, {"gyro_edges": float(accepted_edges)}

        if not torch.isfinite(step).all():
            return None, {"gyro_edges": float(accepted_edges)}

        bias += step.clamp(min=-0.1, max=0.1)
        if float(torch.linalg.norm(step).detach().cpu().item()) < 1e-6:
            break

    final_residual = _rotation_residuals(
        video, src, dst, bias, compose_order, inverse_imu
    )
    final_norm = torch.linalg.norm(final_residual, dim=-1)
    good = torch.isfinite(final_norm) & (final_norm <= max_rotation)
    if int(good.sum().item()) < 3:
        return None, {"gyro_edges": float(good.sum().item())}

    return bias, {
        "gyro_edges": float(good.sum().item()),
        "gyro_residual_deg": float(
            (final_norm[good].mean() * 180.0 / np.pi).detach().cpu().item()
        ),
    }


def _build_alignment_system(
    centers: np.ndarray,
    rotations_wc: np.ndarray,
    delta_metric: np.ndarray,
    jac_metric: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    gyro_bias: np.ndarray,
    gravity_direction: Optional[np.ndarray] = None,
    imu_to_camera_translation: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, _AlignmentLayout]:
    """Build the linear visual-inertial system in the current visual scale."""

    frame_ids = np.unique(np.concatenate([edge_src, edge_dst])).astype(np.int64)
    frame_to_local = {int(frame): index for index, frame in enumerate(frame_ids.tolist())}
    velocity_dim = 3 * len(frame_ids)
    constrained = gravity_direction is not None
    extra_dim = 4 if constrained else 7  # scale, ba or gravity, scale, ba
    total_dim = velocity_dim + extra_dim
    rows = []
    values = []

    if constrained:
        gravity_direction = np.asarray(gravity_direction, dtype=np.float64)
        gravity_direction /= max(np.linalg.norm(gravity_direction), 1e-12)
        scale_col = velocity_dim
        bias_col = velocity_dim + 1
    else:
        gravity_col = velocity_dim
        scale_col = velocity_dim + 3
        bias_col = velocity_dim + 4

    layout = _AlignmentLayout(
        frame_ids=frame_ids,
        velocity_dim=velocity_dim,
        gravity_col=None if constrained else gravity_col,
        scale_col=scale_col,
        bias_col=bias_col,
        total_dim=total_dim,
    )

    lever_arm = np.zeros(3, dtype=np.float64)
    if imu_to_camera_translation is not None:
        lever_arm = np.asarray(imu_to_camera_translation, dtype=np.float64).reshape(-1)
        if lever_arm.size != 3 or not np.isfinite(lever_arm).all():
            raise ValueError("imu_to_camera_translation must contain three finite values")

    for src, dst in zip(edge_src.tolist(), edge_dst.tolist()):
        src_local = frame_to_local[int(src)]
        dst_local = frame_to_local[int(dst)]
        dt = float(delta_metric[dst, 0])
        R_wc = rotations_wc[src]
        dp = delta_metric[dst, 7:10]
        dv = delta_metric[dst, 4:7]
        J_v_ba = jac_metric[dst, 1]
        J_v_bg = jac_metric[dst, 2]
        J_p_ba = jac_metric[dst, 3]
        J_p_bg = jac_metric[dst, 4]
        dp_corrected = dp + J_p_bg @ gyro_bias
        dv_corrected = dv + J_v_bg @ gyro_bias
        # The visual trajectory stores camera centers.  The IMU preintegrated
        # position belongs to the IMU origin, whose visual-unit displacement
        # has an additional scale * (R_WC,j t_CI - R_WC,i t_CI) term.
        lever_delta = rotations_wc[dst] @ lever_arm - rotations_wc[src] @ lever_arm
        position_scale_term = R_wc @ dp_corrected - lever_delta

        for axis in range(3):
            row = np.zeros(total_dim, dtype=np.float64)
            row[3 * src_local : 3 * src_local + 3] = dt * np.eye(3)[axis]
            if constrained:
                row[scale_col] = (
                    0.5 * dt * dt * GRAVITY_MPS2 * gravity_direction[axis]
                    + position_scale_term[axis]
                )
            else:
                row[gravity_col : gravity_col + 3] = (
                    0.5 * dt * dt * np.eye(3)[axis]
                )
                row[scale_col] = position_scale_term[axis]
            row[bias_col : bias_col + 3] = (R_wc @ J_p_ba)[axis]
            rows.append(row)
            values.append((centers[dst] - centers[src])[axis])

        for axis in range(3):
            row = np.zeros(total_dim, dtype=np.float64)
            row[3 * src_local : 3 * src_local + 3] = -np.eye(3)[axis]
            row[3 * dst_local : 3 * dst_local + 3] = np.eye(3)[axis]
            if constrained:
                row[scale_col] = (
                    -dt * GRAVITY_MPS2 * gravity_direction[axis]
                    - (R_wc @ dv_corrected)[axis]
                )
            else:
                row[gravity_col : gravity_col + 3] = -dt * np.eye(3)[axis]
                row[scale_col] = -(R_wc @ dv_corrected)[axis]
            row[bias_col : bias_col + 3] = -(R_wc @ J_v_ba)[axis]
            rows.append(row)
            values.append(0.0)

    if not rows:
        return np.empty((0, total_dim)), np.empty(0), layout
    return np.asarray(rows), np.asarray(values), layout


def _solve_lstsq(A: np.ndarray, b: np.ndarray) -> Tuple[Optional[np.ndarray], int, float]:
    if A.shape[0] < A.shape[1]:
        return None, 0, np.inf
    try:
        solution, _, rank, singular = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, 0, np.inf

    if singular.size == 0 or singular[-1] <= 1e-14:
        condition = np.inf
    else:
        condition = float(singular[0] / singular[-1])
    return solution, int(rank), condition


def _alignment_residual_rms(
    centers: np.ndarray,
    rotations_wc: np.ndarray,
    delta_metric: np.ndarray,
    jac_metric: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    velocities: np.ndarray,
    velocity_indices: np.ndarray,
    gravity: np.ndarray,
    scale: float,
    accel_bias_internal: np.ndarray,
    gyro_bias: np.ndarray,
    imu_to_camera_translation: Optional[np.ndarray] = None,
) -> float:
    errors = []
    frame_to_local = {int(frame): index for index, frame in enumerate(velocity_indices.tolist())}
    lever_arm = np.zeros(3, dtype=np.float64)
    if imu_to_camera_translation is not None:
        lever_arm = np.asarray(imu_to_camera_translation, dtype=np.float64).reshape(-1)
        if lever_arm.size != 3 or not np.isfinite(lever_arm).all():
            return np.inf
    for src, dst in zip(edge_src.tolist(), edge_dst.tolist()):
        src_local = frame_to_local[int(src)]
        dst_local = frame_to_local[int(dst)]
        dt = float(delta_metric[dst, 0])
        R_wc = rotations_wc[src]
        dp = delta_metric[dst, 7:10]
        dv = delta_metric[dst, 4:7]
        J_v_ba = jac_metric[dst, 1]
        J_v_bg = jac_metric[dst, 2]
        J_p_ba = jac_metric[dst, 3]
        J_p_bg = jac_metric[dst, 4]
        p_pred = (
            velocities[src_local] * dt
            + 0.5 * gravity * dt * dt
            + R_wc @ (scale * (dp + J_p_bg @ gyro_bias) + J_p_ba @ accel_bias_internal)
        )
        v_pred = (
            gravity * dt
            + R_wc @ (scale * (dv + J_v_bg @ gyro_bias) + J_v_ba @ accel_bias_internal)
        )
        lever_delta = rotations_wc[dst] @ lever_arm - rotations_wc[src] @ lever_arm
        errors.append(centers[dst] - centers[src] + scale * lever_delta - p_pred)
        errors.append(velocities[dst_local] - velocities[src_local] - v_pred)
    if not errors:
        return np.inf
    return float(np.sqrt(np.mean(np.square(np.concatenate(errors)))))


@torch.no_grad()
def try_visual_inertial_initialization(
    video,
    initial_gyro_bias: Optional[torch.Tensor] = None,
    initial_accel_bias: Optional[torch.Tensor] = None,
    compose_order: str = "dq_prev",
    inverse_imu: bool = True,
    min_frames: int = 20,
    window: int = 30,
    max_dt: float = 0.5,
    min_edges: int = 20,
    min_duration: float = 1.5,
    min_rotation_deg: float = 5.0,
    min_visual_span_ratio: float = 3.0,
    max_rms_m: float = 0.5,
    min_scale: float = 0.05,
    max_scale: float = 20.0,
    max_condition: float = 1e9,
    max_accel_bias_norm: float = 1.0,
    max_gyro_bias_norm: float = 0.2,
) -> VIAlignmentResult:
    """Estimate global VI parameters and metricize the current visual state.

    The first solve keeps gravity unconstrained only to obtain its direction.
    The second solve enforces `|g| = scale * 9.81`, which removes the usual
    scale/gravity ambiguity while retaining one shared accelerometer bias.
    """

    if _alignment_ready(video):
        return VIAlignmentResult(False, "already_aligned")

    count = int(video.counter.value)
    if count < int(min_frames):
        return VIAlignmentResult(False, "insufficient_frames")

    start = max(0, count - max(int(window), int(min_frames)))
    src, dst = _valid_temporal_edges(video, start, count, max_dt)
    edge_count = int(dst.numel())
    edge_duration = float(video.imu_delta[dst, 0].sum().detach().cpu().item())
    rotation_excitation_deg = float(
        (torch.linalg.norm(video.imu_delta[dst, 1:4], dim=-1).sum() * 180.0 / np.pi)
        .detach()
        .cpu()
        .item()
    )
    observability_metrics = {
        "candidate_edges": float(edge_count),
        "candidate_duration_sec": edge_duration,
        "imu_rotation_excitation_deg": rotation_excitation_deg,
    }
    if edge_count < max(8, int(min_edges)):
        return VIAlignmentResult(False, "insufficient_imu_edges", metrics=observability_metrics)
    if min_duration > 0.0 and edge_duration < float(min_duration):
        return VIAlignmentResult(False, "insufficient_imu_duration", metrics=observability_metrics)
    if min_rotation_deg > 0.0 and rotation_excitation_deg < float(min_rotation_deg):
        return VIAlignmentResult(False, "insufficient_imu_rotation", metrics=observability_metrics)

    device = video.poses.device
    dtype = video.poses.dtype
    gyro_bias, gyro_metrics = estimate_gyro_bias(
        video,
        src,
        dst,
        initial_gyro_bias,
        compose_order=compose_order,
        inverse_imu=inverse_imu,
    )
    if gyro_bias is None:
        return VIAlignmentResult(False, "gyro_bias_failed", metrics={**observability_metrics, **gyro_metrics})

    rotation = _rotation_residuals(
        video, src, dst, gyro_bias, compose_order, inverse_imu
    )
    rotation_deg = torch.linalg.norm(rotation, dim=-1) * (180.0 / np.pi)
    usable = torch.isfinite(rotation_deg) & (rotation_deg <= 30.0)
    if int(usable.sum().item()) < 8:
        return VIAlignmentResult(False, "rotation_inconsistent", metrics={**observability_metrics, **gyro_metrics})
    src = src[usable]
    dst = dst[usable]

    poses = video.poses[:count]
    centers = camera_center_from_pose_data(poses).detach().double().cpu().numpy()
    rotations_wc = (
        quat_to_rotmat(poses[:, 3:7]).transpose(-1, -2).detach().double().cpu().numpy()
    )
    delta = video.imu_delta[:count].detach().double().cpu().numpy().copy()
    jacobian = video.imu_bias_jacobian[:count].detach().double().cpu().numpy().copy()
    current_unit_scale = float(video.imu_unit_scale.detach().reshape(-1)[0].cpu().item())
    current_unit_scale = max(current_unit_scale, 1e-12)
    delta[:, 4:10] /= current_unit_scale
    jacobian[:, 1:5] /= current_unit_scale
    lever_arm_metric = (
        video.imu_camera_translation.detach().double().cpu().numpy().copy()
        if hasattr(video, "imu_camera_translation")
        else np.zeros(3, dtype=np.float64)
    )

    used_frame_ids = torch.unique(torch.cat([src, dst])).detach().cpu().numpy()
    visual_points = centers[used_frame_ids]
    visual_span = float(np.linalg.norm(np.ptp(visual_points, axis=0)))
    visual_edge_lengths = np.linalg.norm(centers[dst.detach().cpu().numpy()] - centers[src.detach().cpu().numpy()], axis=1)
    median_visual_edge = float(np.median(visual_edge_lengths)) if visual_edge_lengths.size else 0.0
    visual_span_ratio = visual_span / max(median_visual_edge, 1e-12)
    observability_metrics.update({
        "visual_span_internal": visual_span,
        "median_visual_edge_internal": median_visual_edge,
        "visual_span_ratio": visual_span_ratio,
    })
    if median_visual_edge <= 1e-8 or visual_span_ratio < float(min_visual_span_ratio):
        return VIAlignmentResult(False, "insufficient_visual_excitation", metrics=observability_metrics)

    src_np = src.detach().cpu().numpy()
    dst_np = dst.detach().cpu().numpy()
    gyro_np = gyro_bias.detach().double().cpu().numpy()

    A_raw, b_raw, raw_layout = _build_alignment_system(
        centers,
        rotations_wc,
        delta,
        jacobian,
        src_np,
        dst_np,
        gyro_np,
        gravity_direction=None,
        imu_to_camera_translation=lever_arm_metric,
    )
    raw_solution, raw_rank, raw_condition = _solve_lstsq(A_raw, b_raw)
    if (
        raw_solution is None
        or raw_rank < raw_layout.total_dim
        or raw_condition > float(max_condition)
    ):
        return VIAlignmentResult(
            False,
            "ill_conditioned_alignment",
            metrics={**observability_metrics, **gyro_metrics, "rank": float(raw_rank), "condition": raw_condition},
        )

    raw_gravity = raw_solution[
        raw_layout.gravity_col : raw_layout.gravity_col + 3
    ]
    raw_gravity_norm = float(np.linalg.norm(raw_gravity))
    if not np.isfinite(raw_gravity_norm) or raw_gravity_norm < 1e-6:
        return VIAlignmentResult(False, "unobservable_gravity", metrics={**observability_metrics, **gyro_metrics})
    gravity_direction = raw_gravity / raw_gravity_norm

    A, b, constrained_layout = _build_alignment_system(
        centers,
        rotations_wc,
        delta,
        jacobian,
        src_np,
        dst_np,
        gyro_np,
        gravity_direction=gravity_direction,
        imu_to_camera_translation=lever_arm_metric,
    )
    solution, rank, condition = _solve_lstsq(A, b)
    if (
        solution is None
        or rank < constrained_layout.total_dim
        or condition > float(max_condition)
    ):
        return VIAlignmentResult(
            False,
            "ill_conditioned_constrained_alignment",
            metrics={**observability_metrics, **gyro_metrics, "rank": float(rank), "condition": condition},
        )

    velocities = solution[: constrained_layout.velocity_dim].reshape(-1, 3)
    scale = float(solution[constrained_layout.scale_col])
    accel_bias_internal = solution[
        constrained_layout.bias_col : constrained_layout.bias_col + 3
    ]
    if not np.isfinite(scale) or not (float(min_scale) <= scale <= float(max_scale)):
        return VIAlignmentResult(
            False,
            "scale_out_of_range",
            scale=scale,
            metrics={**observability_metrics, **gyro_metrics, "rank": float(rank), "condition": condition},
        )

    gravity_internal = scale * GRAVITY_MPS2 * gravity_direction
    accel_bias_metric = accel_bias_internal / scale
    if not (
        np.isfinite(velocities).all()
        and np.isfinite(gravity_internal).all()
        and np.isfinite(accel_bias_metric).all()
        and np.isfinite(gyro_np).all()
    ):
        return VIAlignmentResult(False, "nonfinite_alignment", scale=scale)
    if (
        max_accel_bias_norm > 0.0
        and float(np.linalg.norm(accel_bias_metric)) > float(max_accel_bias_norm)
    ):
        return VIAlignmentResult(False, "accel_bias_out_of_range", scale=scale)
    if (
        max_gyro_bias_norm > 0.0
        and float(np.linalg.norm(gyro_np)) > float(max_gyro_bias_norm)
    ):
        return VIAlignmentResult(False, "gyro_bias_out_of_range", scale=scale)

    rms = _alignment_residual_rms(
        centers,
        rotations_wc,
        delta,
        jacobian,
        src_np,
        dst_np,
        velocities,
        constrained_layout.frame_ids,
        gravity_internal,
        scale,
        accel_bias_internal,
        gyro_np,
        imu_to_camera_translation=lever_arm_metric,
    )
    rms_metric = rms / max(scale, 1e-12)
    if max_rms_m > 0.0 and (not np.isfinite(rms_metric) or rms_metric > float(max_rms_m)):
        return VIAlignmentResult(
            False,
            "alignment_residual_too_large",
            scale=scale,
            metrics={
                **observability_metrics,
                **gyro_metrics,
                "rank": float(rank),
                "condition": condition,
                "residual_rms_internal": rms,
                "residual_rms_metric": rms_metric,
            },
        )
    metrics = {
        **observability_metrics,
        **gyro_metrics,
        "edges": float(len(dst_np)),
        "scale": scale,
        "gravity_norm_internal": float(np.linalg.norm(gravity_internal)),
        "condition": condition,
        "rank": float(rank),
        "residual_rms_internal": rms,
        "residual_rms_metric": rms_metric,
        "visual_span_internal": visual_span,
        "mean_rotation_residual_deg": float(rotation_deg[usable].mean().detach().cpu().item()),
        "imu_lever_arm_norm_m": float(np.linalg.norm(lever_arm_metric)),
    }

    return VIAlignmentResult(
        True,
        "ok",
        scale=scale,
        gravity_internal=torch.as_tensor(gravity_internal, device=device, dtype=dtype),
        velocities_internal=torch.as_tensor(velocities, device=device, dtype=dtype),
        velocity_indices=torch.as_tensor(
            constrained_layout.frame_ids, device=device, dtype=torch.long
        ),
        accel_bias_metric=torch.as_tensor(accel_bias_metric, device=device, dtype=dtype),
        gyro_bias_metric=gyro_bias,
        metrics=metrics,
    )


def apply_visual_inertial_alignment(video, result: VIAlignmentResult) -> bool:
    """Commit a successful result while preserving DROID's scale invariance."""

    if not result.success or result.scale <= 0.0:
        return False

    count = int(video.counter.value)
    if count <= 0:
        return False

    scale = float(result.scale)
    with video.get_lock():
        # Return preintegrated translational terms to metres first.  The visual
        # poses/depths are then reparameterized into those same metric units.
        if hasattr(video, "set_imu_unit_scale"):
            video.set_imu_unit_scale(1.0, count=count)
        else:
            old = video.imu_unit_scale[0].clamp_min(1e-12)
            video.imu_delta[:count, 4:10] /= old
            video.imu_info[:count, 1:3] *= old * old
            video.imu_bias_jacobian[:count, 1:5] /= old
            video.imu_unit_scale.fill_(1.0)

        video.poses[:count, :3] /= scale
        video.disps[:count] *= scale
        if hasattr(video, "disps_up"):
            video.disps_up[:count] *= scale
        # Only connected IMU segments participate in the alignment solve. Give
        # disconnected states a neutral value so the existing local velocity
        # initializer can seed them from nearby visual motion on demand.
        video.velocities[:count].zero_()
        if result.velocity_indices is not None:
            indices = result.velocity_indices.to(device=video.velocities.device)
            video.velocities[indices] = result.velocities_internal / scale
        video.bias_acc[:count].zero_()
        video.bias_gyro[:count].zero_()

        video.imu_gravity[:] = result.gravity_internal / scale
        video.imu_global_bias_acc[:] = result.accel_bias_metric
        video.imu_global_bias_gyro[:] = result.gyro_bias_metric
        video.imu_alignment_scale.fill_(scale)
        _set_alignment_ready(video, True)
        video.dirty[:count] = True

    video.imu_alignment_metrics = dict(result.metrics or {})
    return True


def _full_bias_residual_and_jacobian(
    video,
    src: torch.Tensor,
    dst: torch.Tensor,
    accel_bias: torch.Tensor,
    gyro_bias: torch.Tensor,
    gravity: torch.Tensor,
    compose_order: str,
    inverse_imu: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return residuals and a shared-bias Jacobian for selected IMU edges.

    The returned Jacobian follows DROID's normal-equation convention
    ``r - J db``. It is therefore the negative residual derivative. This is
    the same convention used by the pose/velocity blocks in ``FactorGraph``
    and makes the six-variable shared-bias step directly compatible with the
    local IMU DBA objective.
    """

    poses = video.poses
    centers = imu_center_from_pose_data(
        poses,
        video.imu_camera_translation_in_current_units()
        if hasattr(video, "imu_camera_translation_in_current_units")
        else None,
    )
    jacobian = _edge_bias_jacobian(video, dst)
    dt = video.imu_delta[dst, 0:1]
    q_i = poses[src, 3:7]
    q_j = poses[dst, 3:7]
    p_i = centers[src]
    p_j = centers[dst]
    v_i = video.velocities[src]
    v_j = video.velocities[dst]

    rotvec = video.imu_delta[dst, 1:4] + torch.matmul(
        jacobian[:, 0], gyro_bias.view(1, 3, 1)
    ).squeeze(-1)
    dv_imu = (
        video.imu_delta[dst, 4:7]
        + torch.matmul(jacobian[:, 1], accel_bias.view(1, 3, 1)).squeeze(-1)
        + torch.matmul(jacobian[:, 2], gyro_bias.view(1, 3, 1)).squeeze(-1)
    )
    dp_imu = (
        video.imu_delta[dst, 7:10]
        + torch.matmul(jacobian[:, 3], accel_bias.view(1, 3, 1)).squeeze(-1)
        + torch.matmul(jacobian[:, 4], gyro_bias.view(1, 3, 1)).squeeze(-1)
    )
    q_pred = predict_imu_rotation(
        q_i,
        rotvec_to_quat(rotvec),
        compose_order=compose_order,
        inverse_imu=inverse_imu,
    )
    r_R = quat_to_rotvec(quat_multiply(quat_inverse(q_pred), q_j))
    r_p = quat_rotate(q_i, p_j - p_i - v_i * dt - 0.5 * gravity * dt * dt) - dp_imu
    r_v = quat_rotate(q_i, v_j - v_i - gravity * dt) - dv_imu
    residual = torch.cat([r_p, r_v, r_R], dim=-1)

    # Bias correction Jacobian for the normal-equation model r - J db.
    # Translational terms are direct consequences of the preintegrated bias
    # Jacobians. The rotation term carries the SO(3) log-map and composition
    # convention used by the full pose-velocity IMU factor.
    J = torch.zeros(
        (src.numel(), 9, 6), device=poses.device, dtype=poses.dtype
    )
    J[:, 0:3, 0:3] = jacobian[:, 3]  # J_p_ba
    J[:, 0:3, 3:6] = jacobian[:, 4]  # J_p_bg
    J[:, 3:6, 0:3] = jacobian[:, 1]  # J_v_ba
    J[:, 3:6, 3:6] = jacobian[:, 2]  # J_v_bg

    R_i = quat_to_rotmat(q_i)
    R_pred = quat_to_rotmat(q_pred)
    Jl_inv = so3_left_jacobian_inv(r_R)
    gyro_sign = -1.0 if inverse_imu else 1.0
    effective_rotvec = gyro_sign * rotvec
    preintegration_map = gyro_sign * so3_left_jacobian(effective_rotvec)
    if compose_order == "dq_prev":
        gyro_map = R_pred.transpose(-1, -2)
    elif compose_order == "prev_dq":
        gyro_map = R_pred.transpose(-1, -2) @ R_i
    else:
        raise ValueError(f"Unknown compose_order: {compose_order}")
    J[:, 6:9, 3:6] = (
        Jl_inv @ gyro_map @ preintegration_map @ jacobian[:, 0]
    )
    return residual, J


def _full_bias_residuals(
    video,
    src: torch.Tensor,
    dst: torch.Tensor,
    accel_bias: torch.Tensor,
    gyro_bias: torch.Tensor,
    gravity: torch.Tensor,
    compose_order: str,
    inverse_imu: bool,
) -> torch.Tensor:
    """Return `[r_p, r_v, r_R]` for callers that only need residuals."""

    residual, _ = _full_bias_residual_and_jacobian(
        video,
        src,
        dst,
        accel_bias,
        gyro_bias,
        gravity,
        compose_order,
        inverse_imu,
    )
    return residual


def _shared_bias_confidence(
    imu_confidence: Optional[torch.Tensor],
    dst: torch.Tensor,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Select direct temporal GRU confidence for the selected IMU factors."""

    if imu_confidence is None:
        return torch.ones(dst.numel(), device=device, dtype=dtype)

    confidence = imu_confidence.detach().to(device=device, dtype=dtype)
    if confidence.ndim >= 2:
        confidence = confidence.reshape(-1, confidence.shape[-1])[0]
    else:
        confidence = confidence.reshape(-1)
    if confidence.numel() < num_frames:
        return torch.ones(dst.numel(), device=device, dtype=dtype)
    return confidence[dst].clamp(0.0, 1.0)


def _shared_bias_information_scale(
    video,
    dst: torch.Tensor,
    info_reference: Optional[torch.Tensor],
    use_imu_info_weighting: bool,
    info_weight_clip: float,
    info_weight_eps: float,
) -> torch.Tensor:
    """Return per-edge `[p, v, R]` covariance-relative residual scales."""

    device = video.poses.device
    dtype = video.poses.dtype
    if not use_imu_info_weighting or getattr(video, "imu_info", None) is None:
        return torch.ones((dst.numel(), 9), device=device, dtype=dtype)

    info = video.imu_info[dst].detach().to(device=device, dtype=dtype)
    valid = torch.isfinite(info) & (info > float(info_weight_eps))
    if info_reference is None:
        reference = torch.ones(3, device=device, dtype=dtype)
        for term in range(3):
            values = info[:, term][valid[:, term]]
            if values.numel() > 0:
                reference[term] = values.median().clamp_min(float(info_weight_eps))
    else:
        reference = torch.as_tensor(
            info_reference, device=device, dtype=dtype
        ).reshape(-1)
        if reference.numel() < 3:
            reference = torch.ones(3, device=device, dtype=dtype)
        else:
            reference = reference[:3].clamp_min(float(info_weight_eps))

    ratio = torch.ones_like(info)
    ratio[valid] = info[valid] / reference.expand_as(info)[valid]
    if info_weight_clip > 0.0:
        ratio = ratio.clamp(
            min=1.0 / float(info_weight_clip),
            max=float(info_weight_clip),
        )
    else:
        ratio = ratio.clamp_min(0.0)
    scale = ratio.sqrt()
    return torch.cat(
        [
            scale[:, 2:3].expand(-1, 3),
            scale[:, 1:2].expand(-1, 3),
            scale[:, 0:1].expand(-1, 3),
        ],
        dim=-1,
    )


def _huber_cost_and_irls(norm: torch.Tensor, delta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return vector Huber costs and IRLS factors for factor norms."""

    if delta <= 0.0:
        return 0.5 * norm.square(), torch.ones_like(norm)
    delta_t = norm.new_tensor(float(delta))
    quadratic = norm <= delta_t
    cost = torch.where(
        quadratic,
        0.5 * norm.square(),
        delta_t * (norm - 0.5 * delta_t),
    )
    irls = torch.where(quadratic, torch.ones_like(norm), delta_t / norm.clamp_min(1e-12))
    return cost, irls


def _shared_bias_prior_std(
    video,
    dst: torch.Tensor,
    elapsed_sec: float,
    accel_random_walk: float,
    gyro_random_walk: float,
    accel_prior_std: float,
    gyro_prior_std: float,
) -> torch.Tensor:
    """Build a bias random-walk trust region around the current estimate."""

    device = video.poses.device
    dtype = video.poses.dtype
    random_walk = getattr(video, "imu_bias_random_walk", None)
    if random_walk is None:
        measured_accel_rw = 0.0
        measured_gyro_rw = 0.0
    else:
        values = random_walk[dst].detach().to(device=device, dtype=dtype)
        measured_accel = values[:, 0][torch.isfinite(values[:, 0]) & (values[:, 0] > 0.0)]
        measured_gyro = values[:, 1][torch.isfinite(values[:, 1]) & (values[:, 1] > 0.0)]
        measured_accel_rw = (
            float(measured_accel.median().item()) if measured_accel.numel() > 0 else 0.0
        )
        measured_gyro_rw = (
            float(measured_gyro.median().item()) if measured_gyro.numel() > 0 else 0.0
        )

    accel_rw = measured_accel_rw if accel_random_walk < 0.0 else float(accel_random_walk)
    gyro_rw = measured_gyro_rw if gyro_random_walk < 0.0 else float(gyro_random_walk)
    elapsed = max(0.0, float(elapsed_sec))
    accel_std = np.hypot(max(0.0, float(accel_prior_std)), max(0.0, accel_rw) * np.sqrt(elapsed))
    gyro_std = np.hypot(max(0.0, float(gyro_prior_std)), max(0.0, gyro_rw) * np.sqrt(elapsed))
    return torch.as_tensor(
        [accel_std, accel_std, accel_std, gyro_std, gyro_std, gyro_std],
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def refine_window_biases(
    video,
    compose_order: str = "dq_prev",
    inverse_imu: bool = True,
    window: int = 30,
    max_dt: float = 0.5,
    pos_weight: float = 0.05,
    vel_weight: float = 0.05,
    damping: float = 1e-4,
    step_scale: float = 0.5,
    max_accel_step: float = 0.15,
    max_gyro_step: float = 0.015,
    max_accel_norm: float = 0.5,
    max_gyro_norm: float = 0.1,
    global_weight: float = 1.0,
    imu_confidence: Optional[torch.Tensor] = None,
    use_imu_info_weighting: bool = False,
    imu_info_reference: Optional[torch.Tensor] = None,
    imu_info_weight_clip: float = 4.0,
    imu_info_weight_eps: float = 1e-12,
    robust_delta: float = 3.0,
    min_edges: int = 8,
    min_duration: float = 0.5,
    min_rotation_deg: float = 2.0,
    max_condition: float = 1e8,
    min_relative_improvement: float = 0.0,
    prior_elapsed_sec: Optional[float] = None,
    accel_random_walk: float = -1.0,
    gyro_random_walk: float = -1.0,
    accel_prior_std: float = 0.05,
    gyro_prior_std: float = 0.005,
) -> Dict[str, float]:
    """Refine one shared `(b_a, b_g)` inside the local 9DoF IMU objective.

    The pose/velocity/depth CUDA DBA and this six-variable block are solved
    alternately. This is a block-coordinate Gauss-Newton step over
    ``{T_i, v_i, d_i}_{i in window} + {b_a, b_g}``, avoiding duplicated
    per-keyframe bias states while preserving the preintegration coupling.
    """

    if not _alignment_ready(video):
        return {"updated": 0.0, "reason": "not_aligned"}
    if global_weight <= 0.0 or not np.isfinite(global_weight):
        return {"updated": 0.0, "reason": "zero_global_weight"}

    count = int(video.counter.value)
    start = max(0, count - max(2, int(window)))
    src, dst = _valid_temporal_edges(video, start, count, max_dt)
    if dst.numel() < max(1, int(min_edges)):
        return {"updated": 0.0, "reason": "insufficient_edges"}

    device = video.poses.device
    dtype = video.poses.dtype
    accel_bias = video.imu_global_bias_acc.detach().to(device=device, dtype=dtype).clone()
    gyro_bias = video.imu_global_bias_gyro.detach().to(device=device, dtype=dtype).clone()
    gravity = video.imu_gravity.detach().to(device=device, dtype=dtype)

    duration = float(video.imu_delta[dst, 0].sum().detach().cpu().item())
    if duration < float(min_duration):
        return {
            "updated": 0.0,
            "reason": "insufficient_duration",
            "edges": float(dst.numel()),
            "duration_sec": duration,
        }
    rotation_excitation_deg = float(
        video.imu_delta[dst, 1:4].norm(dim=-1).sum().detach().cpu().item()
        * (180.0 / np.pi)
    )
    if rotation_excitation_deg < float(min_rotation_deg):
        return {
            "updated": 0.0,
            "reason": "insufficient_rotation",
            "edges": float(dst.numel()),
            "duration_sec": duration,
            "rotation_excitation_deg": rotation_excitation_deg,
        }

    residual, jacobian = _full_bias_residual_and_jacobian(
        video, src, dst, accel_bias, gyro_bias, gravity, compose_order, inverse_imu
    )
    base_scale = residual.new_tensor(
        [pos_weight] * 3 + [vel_weight] * 3 + [1.0] * 3
    ).clamp_min(0.0).sqrt()
    information_scale = _shared_bias_information_scale(
        video,
        dst,
        imu_info_reference,
        use_imu_info_weighting,
        imu_info_weight_clip,
        imu_info_weight_eps,
    )
    residual_scale = base_scale.view(1, 9) * information_scale
    confidence = _shared_bias_confidence(
        imu_confidence, dst, count, device, dtype
    )
    row_weight = video.imu_weight[dst].detach().to(device=device, dtype=dtype)
    row_scale = (row_weight / 0.001).clamp(0.0, 1.0)
    edge_weight = float(global_weight) * confidence * row_scale
    valid = (
        torch.isfinite(residual).all(dim=-1)
        & torch.isfinite(jacobian).all(dim=(-1, -2))
        & torch.isfinite(residual_scale).all(dim=-1)
        & torch.isfinite(edge_weight)
        & (edge_weight > 0.0)
    )
    if int(valid.sum().item()) < max(1, int(min_edges)):
        return {"updated": 0.0, "reason": "insufficient_weighted_edges"}

    src = src[valid]
    dst = dst[valid]
    residual = residual[valid]
    jacobian = jacobian[valid]
    residual_scale = residual_scale[valid]
    confidence = confidence[valid]
    edge_weight = edge_weight[valid]
    normalized_residual = residual * residual_scale
    residual_norm = normalized_residual.norm(dim=-1)
    _, irls = _huber_cost_and_irls(residual_norm, robust_delta)
    sqrt_weight = (edge_weight * irls).clamp_min(0.0).sqrt()
    weighted_residual = normalized_residual * sqrt_weight[:, None]
    weighted_jacobian = jacobian * residual_scale[:, :, None] * sqrt_weight[:, None, None]

    hessian_data = torch.einsum("eri,erj->ij", weighted_jacobian, weighted_jacobian)
    gradient = torch.einsum("eri,er->i", weighted_jacobian, weighted_residual)
    hessian_sym = 0.5 * (hessian_data + hessian_data.T)
    try:
        eigenvalues = torch.linalg.eigvalsh(hessian_sym)
    except RuntimeError:
        return {"updated": 0.0, "reason": "eigendecomposition_failed"}
    largest = float(eigenvalues[-1].detach().cpu().item()) if eigenvalues.numel() else 0.0
    rank_threshold = max(largest * 1e-6, 1e-12)
    rank = int((eigenvalues > rank_threshold).sum().detach().cpu().item())
    smallest = float(eigenvalues[0].detach().cpu().item()) if eigenvalues.numel() else 0.0
    condition = largest / max(smallest, rank_threshold) if largest > 0.0 else float("inf")
    if rank < 6 or not np.isfinite(condition) or condition > float(max_condition):
        return {
            "updated": 0.0,
            "reason": "poor_observability",
            "edges": float(dst.numel()),
            "rank": float(rank),
            "condition": condition,
            "rotation_excitation_deg": rotation_excitation_deg,
        }

    elapsed = duration if prior_elapsed_sec is None else float(prior_elapsed_sec)
    prior_std = _shared_bias_prior_std(
        video,
        dst,
        elapsed,
        accel_random_walk,
        gyro_random_walk,
        accel_prior_std,
        gyro_prior_std,
    )
    prior_precision = torch.zeros(6, device=device, dtype=dtype)
    usable_prior = torch.isfinite(prior_std) & (prior_std > 0.0)
    prior_precision[usable_prior] = 1.0 / prior_std[usable_prior].square()
    hessian = hessian_sym + torch.diag(prior_precision)
    hessian += float(damping) * torch.eye(6, device=device, dtype=dtype)
    try:
        # The Jacobian uses DROID's `r - J db` convention, so the normal
        # equation is H db = J^T r. The older finite-difference implementation
        # differentiated r directly and therefore used the opposite sign.
        step = torch.linalg.solve(hessian, gradient) * float(step_scale)
    except RuntimeError:
        return {"updated": 0.0, "reason": "singular"}
    if not torch.isfinite(step).all():
        return {"updated": 0.0, "reason": "nonfinite_step"}

    accel_step = step[:3]
    gyro_step = step[3:]
    accel_norm = accel_step.norm().clamp_min(1e-12)
    gyro_norm = gyro_step.norm().clamp_min(1e-12)
    if max_accel_step > 0.0:
        accel_step = accel_step * min(1.0, float(max_accel_step) / float(accel_norm.item()))
    if max_gyro_step > 0.0:
        gyro_step = gyro_step * min(1.0, float(max_gyro_step) / float(gyro_norm.item()))

    candidate_accel = accel_bias + accel_step
    candidate_gyro = gyro_bias + gyro_step
    if max_accel_norm > 0.0 and float(candidate_accel.norm().item()) > float(max_accel_norm):
        candidate_accel *= float(max_accel_norm) / float(candidate_accel.norm().item())
    if max_gyro_norm > 0.0 and float(candidate_gyro.norm().item()) > float(max_gyro_norm):
        candidate_gyro *= float(max_gyro_norm) / float(candidate_gyro.norm().item())

    def objective(accel, gyro, prior_step=None):
        candidate_residual, _ = _full_bias_residual_and_jacobian(
            video, src, dst, accel, gyro, gravity, compose_order, inverse_imu
        )
        norm = (candidate_residual * residual_scale).norm(dim=-1)
        cost, _ = _huber_cost_and_irls(norm, robust_delta)
        total = torch.sum(edge_weight * cost)
        if prior_step is not None:
            total = total + 0.5 * torch.sum(prior_precision * prior_step.square())
        return total

    before_t = objective(accel_bias, gyro_bias)
    candidate_step = torch.cat(
        [candidate_accel - accel_bias, candidate_gyro - gyro_bias], dim=0
    )
    after_t = objective(candidate_accel, candidate_gyro, candidate_step)
    before = float(before_t.detach().cpu().item())
    after = float(after_t.detach().cpu().item())
    relative_improvement = (before - after) / max(abs(before), 1e-12)
    if not np.isfinite(after) or after >= before:
        return {
            "updated": 0.0,
            "reason": "no_improvement",
            "edges": float(len(dst)),
            "cost_before": before,
            "cost_after": after,
            "relative_improvement": relative_improvement,
        }
    if (
        float(min_relative_improvement) > 0.0
        and relative_improvement < float(min_relative_improvement)
    ):
        return {
            "updated": 0.0,
            "reason": "insufficient_relative_improvement",
            "edges": float(len(dst)),
            "cost_before": before,
            "cost_after": after,
            "relative_improvement": relative_improvement,
        }

    accel_step = candidate_accel - accel_bias
    gyro_step = candidate_gyro - gyro_bias
    video.imu_global_bias_acc[:] = candidate_accel
    video.imu_global_bias_gyro[:] = candidate_gyro
    return {
        "updated": 1.0,
        "edges": float(len(dst)),
        "duration_sec": duration,
        "rotation_excitation_deg": rotation_excitation_deg,
        "condition": condition,
        "rank": float(rank),
        "confidence_mean": float(confidence.mean().detach().cpu().item()),
        "prior_accel_std": float(prior_std[:3].mean().detach().cpu().item()),
        "prior_gyro_std": float(prior_std[3:].mean().detach().cpu().item()),
        "cost_before": before,
        "cost_after": after,
        "relative_improvement": relative_improvement,
        "accel_step_norm": float(accel_step.norm().detach().cpu().item()),
        "gyro_step_norm": float(gyro_step.norm().detach().cpu().item()),
        "accel_bias_norm": float(candidate_accel.norm().detach().cpu().item()),
        "gyro_bias_norm": float(candidate_gyro.norm().detach().cpu().item()),
    }


def record_window_bias_refinement(video, result: Dict[str, float], stage: str) -> None:
    """Persist scalar shared-bias updates for reconstruction-side analysis."""

    history = getattr(video, "imu_bias_refinement_history", None)
    if history is None:
        history = []
        video.imu_bias_refinement_history = history

    entry = {"stage": str(stage), "frame": int(video.counter.value)}
    for key, value in result.items():
        if isinstance(value, (str, int, float, np.integer, np.floating)):
            entry[key] = value.item() if isinstance(value, np.generic) else value
    accel = video.imu_global_bias_acc.detach().cpu().tolist()
    gyro = video.imu_global_bias_gyro.detach().cpu().tolist()
    for axis, value in enumerate(accel):
        entry[f"accel_bias_{axis}"] = float(value)
    for axis, value in enumerate(gyro):
        entry[f"gyro_bias_{axis}"] = float(value)
    history.append(entry)
    if len(history) > 4096:
        del history[:-4096]

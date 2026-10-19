"""
Rotation-only IMU factor prototype for DROID-SLAM.

This module is intentionally independent from the CUDA DBA backend. It provides
the first testable form of the inertial residual. DROID poses store quaternions
in xyzw order, but the correct way to compose an IMU delta depends on the pose
convention used by the caller. Keep that convention explicit with
`compose_order` and `inverse_imu` instead of baking one formula into every IMU
path.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch


IMU_BIAS_JACOBIAN_NAMES = ("J_R_bg", "J_v_ba", "J_v_bg", "J_p_ba", "J_p_bg")

# The learned confidence head uses one value for each physical part of a full
# preintegrated IMU factor.  Keep this order shared by the network, training
# loss, and both DBA implementations; residuals themselves remain ordered as
# [position, velocity, rotation, accel-bias, gyro-bias].
IMU_CONFIDENCE_COMPONENTS = ("rotation", "velocity", "position")
IMU_QUALITY_FEATURE_DIM = 7


def quat_normalize(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    xyz = -q[..., :3]
    return torch.cat([xyz, q[..., 3:4]], dim=-1)


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product for xyzw quaternions."""

    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)

    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)

    out = torch.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dim=-1,
    )
    return quat_normalize(out)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate 3D vectors by xyzw quaternions."""

    q = quat_normalize(q)
    u = q[..., :3]
    w = q[..., 3:4]

    while u.ndim < v.ndim:
        u = u.unsqueeze(-2)
        w = w.unsqueeze(-2)

    uv = torch.cross(u, v, dim=-1)
    uuv = torch.cross(u, uv, dim=-1)
    return v + 2.0 * (w * uv + uuv)


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """Return the skew matrix `[v]x` for vectors with shape `[..., 3]`."""

    z = torch.zeros_like(v[..., 0])
    x, y, z_axis = v.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack([z, -z_axis, y], dim=-1),
            torch.stack([z_axis, z, -x], dim=-1),
            torch.stack([-y, x, z], dim=-1),
        ],
        dim=-2,
    )


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert xyzw quaternions to rotation matrices."""

    q = quat_normalize(q)
    x, y, z, w = q.unbind(dim=-1)

    one = torch.ones_like(w)
    two = 2.0

    r00 = one - two * (y * y + z * z)
    r01 = two * (x * y - z * w)
    r02 = two * (x * z + y * w)

    r10 = two * (x * y + z * w)
    r11 = one - two * (x * x + z * z)
    r12 = two * (y * z - x * w)

    r20 = two * (x * z - y * w)
    r21 = two * (y * z + x * w)
    r22 = one - two * (x * x + y * y)

    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def so3_left_jacobian_inv(phi: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Inverse left Jacobian of SO(3) for rotation vectors `phi`."""

    K = skew_symmetric(phi)
    K2 = K @ K

    eye = torch.eye(3, device=phi.device, dtype=phi.dtype)
    while eye.ndim < K.ndim:
        eye = eye.unsqueeze(0)
    eye = eye.expand(K.shape)

    theta2 = torch.sum(phi * phi, dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(float(eps) * float(eps)))
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    theta2_safe = theta2.clamp_min(float(eps) * float(eps))
    denom = (2.0 * theta * sin_theta).clamp_min(float(eps))
    coef = 1.0 / theta2_safe - (1.0 + cos_theta) / denom
    small_coef = 1.0 / 12.0 + theta2 / 720.0
    coef = torch.where(theta2 < float(eps) * float(eps), small_coef, coef)

    return eye - 0.5 * K + coef[..., None] * K2


def so3_left_jacobian(phi: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Left Jacobian of SO(3) for rotation vectors ``phi``.

    This maps an increment in exponential coordinates to a left-multiplied
    rotation perturbation. Preintegrated gyro-bias Jacobians need this term
    before they pass through the residual log-map Jacobian.
    """

    K = skew_symmetric(phi)
    K2 = K @ K

    eye = torch.eye(3, device=phi.device, dtype=phi.dtype)
    while eye.ndim < K.ndim:
        eye = eye.unsqueeze(0)
    eye = eye.expand(K.shape)

    theta2 = torch.sum(phi * phi, dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(float(eps) * float(eps)))
    theta2_safe = theta2.clamp_min(float(eps) * float(eps))
    theta3_safe = (theta2_safe * theta).clamp_min(float(eps) ** 3)
    a = (1.0 - torch.cos(theta)) / theta2_safe
    b = (theta - torch.sin(theta)) / theta3_safe
    small = theta2 < float(eps) * float(eps)
    a_small = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
    b_small = 1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0
    a = torch.where(small, a_small, a)
    b = torch.where(small, b_small, b)
    return eye + a[..., None] * K + b[..., None] * K2


def default_imu_bias_jacobian(dt: torch.Tensor) -> torch.Tensor:
    """
    First-order fallback bias Jacobians for `[J_R_bg, J_v_ba, J_v_bg, J_p_ba, J_p_bg]`.

    Older `imu_prior.csv` files do not contain propagated bias Jacobians. In
    that case this preserves the original constant-bias approximation:
        dR/dbg ~= -dt I, dv/dba ~= -dt I, dp/dba ~= -0.5 dt^2 I.
    """

    dt = dt.clamp_min(0.0)
    out = torch.zeros(dt.shape[:-1] + (len(IMU_BIAS_JACOBIAN_NAMES), 3, 3), device=dt.device, dtype=dt.dtype)
    eye = torch.eye(3, device=dt.device, dtype=dt.dtype)
    while eye.ndim < out.ndim - 2:
        eye = eye.unsqueeze(0)

    out[..., 0, :, :] = -dt[..., None] * eye
    out[..., 1, :, :] = -dt[..., None] * eye
    out[..., 3, :, :] = -0.5 * dt[..., None] * dt[..., None] * eye
    return out


def select_imu_bias_jacobian(
    imu_bias_jacobian: Optional[torch.Tensor],
    dst: torch.Tensor,
    batch: int,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Select per-edge preintegration bias Jacobians with old-CSV fallback."""

    if imu_bias_jacobian is None:
        return default_imu_bias_jacobian(dt)

    jac = imu_bias_jacobian.to(device=device, dtype=dtype)

    if jac.ndim == 4:
        if jac.shape[0] < num_frames or jac.shape[1:4] != (len(IMU_BIAS_JACOBIAN_NAMES), 3, 3):
            raise ValueError(
                "imu_bias_jacobian must have shape [N, 5, 3, 3] or [B, N, 5, 3, 3], "
                f"got {tuple(imu_bias_jacobian.shape)} for frames={num_frames}"
            )
        selected = jac[dst][None].expand(batch, -1, -1, -1, -1)

    elif jac.ndim == 5:
        if jac.shape[0] == 1 and batch > 1:
            jac = jac.expand(batch, -1, -1, -1, -1)
        if jac.shape[0] != batch or jac.shape[1] < num_frames or jac.shape[2:5] != (len(IMU_BIAS_JACOBIAN_NAMES), 3, 3):
            raise ValueError(
                "imu_bias_jacobian must have shape [N, 5, 3, 3] or [B, N, 5, 3, 3], "
                f"got {tuple(imu_bias_jacobian.shape)} for batch={batch}, frames={num_frames}"
            )
        selected = jac[:, dst]

    else:
        raise ValueError(
            "imu_bias_jacobian must have shape [N, 5, 3, 3] or [B, N, 5, 3, 3], "
            f"got {tuple(imu_bias_jacobian.shape)}"
        )

    fallback = default_imu_bias_jacobian(dt)
    has_values = selected.abs().sum(dim=(-1, -2, -3), keepdim=True) > 1e-12
    return torch.where(has_values, selected, fallback)


def rotvec_to_quat(rotvec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert axis-angle vectors to xyzw quaternions."""

    theta = rotvec.norm(dim=-1, keepdim=True)
    half = 0.5 * theta
    small = theta < eps
    axis = rotvec / theta.clamp_min(eps)

    xyz = axis * torch.sin(half)
    w = torch.cos(half)

    small_xyz = 0.5 * rotvec
    small_w = torch.ones_like(w)
    q = torch.cat(
        [
            torch.where(small, small_xyz, xyz),
            torch.where(small, small_w, w),
        ],
        dim=-1,
    )
    return quat_normalize(q)


def predict_imu_rotation(
    q_i: torch.Tensor,
    q_imu: torch.Tensor,
    compose_order: str = "prev_dq",
    inverse_imu: bool = False,
) -> torch.Tensor:
    """Predict q_j from q_i and a preintegrated IMU delta quaternion."""

    if inverse_imu:
        q_imu = quat_inverse(q_imu)

    if compose_order == "prev_dq":
        return quat_multiply(q_i, q_imu)

    if compose_order == "dq_prev":
        return quat_multiply(q_imu, q_i)

    raise ValueError(f"Unknown compose_order: {compose_order}")


def camera_center_from_pose_data(pose_data: torch.Tensor) -> torch.Tensor:
    """
    Convert DROID pose data `[t_cw, q_cw]` into camera centers in world units.

    DROID/LieTorch stores poses as a world-to-camera transform. The translation
    part is therefore not the camera position itself. The camera center is:

        c_w = -R_cw^T t_cw

    Full IMU position/velocity residuals must use this center, not raw `t_cw`.
    """

    t_cw = pose_data[..., 0:3]
    q_cw = pose_data[..., 3:7]
    return quat_rotate(quat_inverse(q_cw), -t_cw)


def imu_center_from_pose_data(
    pose_data: torch.Tensor,
    imu_to_camera_translation: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return IMU origins in world coordinates from DROID camera poses.

    ``imu_to_camera_translation`` is ``t_CI`` in metres (or the current pose
    length unit), defined by the rigid transform::

        p_C = R_CI p_I + t_CI

    Therefore the IMU origin is ``t_CI`` expressed in the camera frame.  DROID
    stores ``T_CW``; after inversion its rotation maps camera vectors into the
    world, so the corresponding IMU center is::

        p_WI = p_WC + R_WC t_CI

    A zero or omitted lever arm preserves the historical camera-center model.
    """

    camera_centers = camera_center_from_pose_data(pose_data)
    if imu_to_camera_translation is None:
        return camera_centers

    translation = torch.as_tensor(
        imu_to_camera_translation,
        device=pose_data.device,
        dtype=pose_data.dtype,
    )
    if translation.numel() == 0:
        return camera_centers
    if translation.shape[-1] != 3:
        raise ValueError(
            "imu_to_camera_translation must end in three coordinates, "
            f"got {tuple(translation.shape)}"
        )

    target_shape = camera_centers.shape
    if translation.ndim == 1:
        translation = translation.view(*([1] * (camera_centers.ndim - 1)), 3)
    elif translation.ndim == 2 and camera_centers.ndim == 3:
        # Allow either one translation per batch or one per video frame.
        if translation.shape[0] == camera_centers.shape[0]:
            translation = translation[:, None, :]
        elif translation.shape[0] == camera_centers.shape[1]:
            translation = translation[None, :, :]

    try:
        translation = torch.broadcast_to(translation, target_shape)
    except RuntimeError as exc:
        raise ValueError(
            "imu_to_camera_translation must be [3], [B, 3], [N, 3], or "
            f"match pose centers {tuple(target_shape)}; got {tuple(translation.shape)}"
        ) from exc

    R_wc = quat_to_rotmat(pose_data[..., 3:7]).transpose(-1, -2)
    return camera_centers + torch.matmul(R_wc, translation.unsqueeze(-1)).squeeze(-1)


def temporal_imu_confidence_from_edges(
    edge_confidence: Optional[torch.Tensor],
    ii: torch.Tensor,
    jj: torch.Tensor,
    num_frames: int,
    fallback: Union[float, torch.Tensor] = 1.0,
    return_coverage: bool = False,
):
    """Convert GRU edge confidence maps into confidence for temporal IMU factors.

    The IMU factor for keyframe ``j`` is specifically the interval
    ``(j - 1, j)``.  Averaging confidence over every graph edge that merely
    touches frame ``j`` loses that meaning and makes one bad or good visual
    match affect unrelated IMU intervals.  This helper keeps only direct
    forward/reverse graph edges for each temporal pair and averages the two
    directions when both exist.

    Missing temporal visual edges receive ``fallback``.  This is normally a
    neutral constant for legacy callers.  The scalar DVI quality-fusion path
    instead supplies its IMU-quality-only confidence for every temporal row,
    so a factor never silently becomes full-strength merely because the
    current visual graph lacks the direct ``(j-1, j)`` edge.
    """

    if edge_confidence is None:
        return (None, None) if return_coverage else None

    if int(num_frames) < 0:
        raise ValueError(f"num_frames must be non-negative, got {num_frames}")

    confidence = edge_confidence
    if confidence.ndim == 1:
        confidence = confidence.unsqueeze(0).unsqueeze(-1)
    elif confidence.ndim == 2:
        confidence = confidence.unsqueeze(-1)
    elif confidence.ndim > 3:
        # UpdateModule emits [B, E, H, W, C].  Spatially pool every axis
        # between edge and component while preserving the final C channels.
        confidence = confidence.mean(dim=tuple(range(2, confidence.ndim - 1)))
    if confidence.ndim != 3:
        raise ValueError(
            "edge_confidence must be [E], [B, E], or [B, E, ..., C], "
            f"got {tuple(edge_confidence.shape)}"
        )

    ii = ii.to(device=confidence.device, dtype=torch.long).reshape(-1)
    jj = jj.to(device=confidence.device, dtype=torch.long).reshape(-1)
    if ii.numel() != confidence.shape[1] or jj.numel() != confidence.shape[1]:
        raise ValueError(
            "edge confidence and graph edges must have matching length, got "
            f"confidence={confidence.shape[1]}, ii={ii.numel()}, jj={jj.numel()}"
        )

    num_frames = int(num_frames)
    if torch.is_tensor(fallback):
        fallback_tensor = fallback.to(device=confidence.device, dtype=confidence.dtype)
        if fallback_tensor.ndim == 1:
            fallback_tensor = fallback_tensor.unsqueeze(0).unsqueeze(-1)
        elif fallback_tensor.ndim == 2:
            fallback_tensor = fallback_tensor.unsqueeze(-1)
        if fallback_tensor.ndim != 3:
            raise ValueError(
                "tensor fallback must be [N], [B, N], or [B, N, C], got "
                f"{tuple(fallback.shape)}"
            )
        if fallback_tensor.shape[0] == 1 and confidence.shape[0] > 1:
            fallback_tensor = fallback_tensor.expand(confidence.shape[0], -1, -1)
        if (
            fallback_tensor.shape[0] != confidence.shape[0]
            or fallback_tensor.shape[1] < num_frames
            or fallback_tensor.shape[-1] not in (1, confidence.shape[-1])
        ):
            raise ValueError(
                "tensor fallback must cover the temporal confidence layout: "
                f"fallback={tuple(fallback_tensor.shape)}, confidence="
                f"{tuple(confidence.shape)}, num_frames={num_frames}"
            )
        fallback_tensor = fallback_tensor[:, :num_frames]
        if fallback_tensor.shape[-1] == 1 and confidence.shape[-1] > 1:
            fallback_tensor = fallback_tensor.expand(-1, -1, confidence.shape[-1])
        temporal = fallback_tensor.clone()
    else:
        temporal = confidence.new_full(
            (confidence.shape[0], num_frames, confidence.shape[-1]), float(fallback)
        )
    coverage = torch.zeros(num_frames, dtype=torch.bool, device=confidence.device)
    for dst in range(1, num_frames):
        src = dst - 1
        direct = ((ii == src) & (jj == dst)) | ((ii == dst) & (jj == src))
        if bool(direct.any().item()):
            temporal[:, dst] = confidence[:, direct].mean(dim=1)
            coverage[dst] = True

    temporal = temporal.clamp(0.0, 1.0)
    if temporal.shape[-1] == 1:
        temporal = temporal[..., 0]
    if return_coverage:
        return temporal, coverage
    return temporal


def temporal_imu_quality_features(
    imu_delta: Optional[torch.Tensor],
    imu_info: Optional[torch.Tensor],
    num_frames: int,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    """Build normalized per-interval quality features from preintegration.

    The descriptor intentionally contains only quantities available for both
    offline training and streaming inference.  It is not a hand-written gate:
    a zero-initialized MLP learns how each quantity should alter the GRU's
    visual confidence during fine-tuning.

    Output layout is ``[B, N, 7]``:
    ``log(1 + dt/0.05)``, angular-rate proxy, acceleration proxy,
    position-change proxy, and covariance-information terms for rotation,
    velocity, and position.  Frame zero and invalid/missing rows are neutral
    zeros.
    """

    if imu_delta is None:
        return None

    delta = imu_delta
    if delta.ndim == 2:
        delta = delta.unsqueeze(0)
    if delta.ndim != 3 or delta.shape[-1] < 10:
        raise ValueError(
            "imu_delta must have shape [N, >=10] or [B, N, >=10], "
            f"got {tuple(imu_delta.shape)}"
        )

    count = min(int(num_frames), int(delta.shape[1]))
    output = delta.new_zeros((delta.shape[0], int(num_frames), IMU_QUALITY_FEATURE_DIM))
    if count <= 1:
        return output

    dt = delta[:, :count, 0:1].clamp_min(0.0)
    safe_dt = dt.clamp_min(float(eps))
    dr = delta[:, :count, 1:4].norm(dim=-1, keepdim=True)
    dv = delta[:, :count, 4:7].norm(dim=-1, keepdim=True)
    dp = delta[:, :count, 7:10].norm(dim=-1, keepdim=True)
    motion = torch.cat(
        [
            torch.log1p(dt / 0.05),
            torch.log1p(dr / safe_dt),
            torch.log1p(dv / safe_dt),
            torch.log1p(dp / (safe_dt * safe_dt)),
        ],
        dim=-1,
    )

    info_features = delta.new_zeros((delta.shape[0], count, 3))
    if imu_info is not None:
        info = imu_info.to(device=delta.device, dtype=delta.dtype)
        if info.ndim == 2:
            info = info.unsqueeze(0)
        if info.ndim != 3 or info.shape[-1] < 3:
            raise ValueError(
                "imu_info must have shape [N, 3] or [B, N, 3], "
                f"got {tuple(imu_info.shape)}"
            )
        if info.shape[0] == 1 and delta.shape[0] > 1:
            info = info.expand(delta.shape[0], -1, -1)
        if info.shape[0] != delta.shape[0]:
            raise ValueError(
                "imu_info batch must match imu_delta batch, "
                f"got info={tuple(info.shape)}, delta={tuple(delta.shape)}"
            )
        values = info[:, :count, :3].clamp_min(float(eps))
        valid = (dt > float(eps)).expand_as(values)
        masked = torch.where(valid, values, torch.full_like(values, float("nan")))
        reference = torch.nanmedian(masked, dim=1, keepdim=True).values
        reference = torch.where(
            torch.isfinite(reference), reference, torch.ones_like(reference)
        ).clamp_min(float(eps))
        info_features = torch.log(values / reference).clamp(-6.0, 6.0)
        info_features = torch.where(valid, info_features, torch.zeros_like(info_features))

    output[:, :count] = torch.cat([motion, info_features], dim=-1)
    output[:, 0] = 0.0
    return output


def edge_imu_quality_features(
    imu_delta: Optional[torch.Tensor],
    imu_info: Optional[torch.Tensor],
    ii: torch.Tensor,
    jj: torch.Tensor,
    num_frames: int,
) -> Optional[torch.Tensor]:
    """Select a temporal IMU descriptor for direct visual graph edges.

    A non-temporal/proximity edge has no one-to-one preintegrated interval, so
    it receives neutral zeros instead of incorrectly borrowing an adjacent
    interval's sensor quality.
    """

    temporal = temporal_imu_quality_features(imu_delta, imu_info, num_frames)
    if temporal is None:
        return None

    ii = ii.to(device=temporal.device, dtype=torch.long).reshape(-1)
    jj = jj.to(device=temporal.device, dtype=torch.long).reshape(-1)
    if ii.numel() != jj.numel():
        raise ValueError("ii and jj must have matching edge counts")

    features = temporal.new_zeros((temporal.shape[0], ii.numel(), IMU_QUALITY_FEATURE_DIM))
    direct = (ii - jj).abs() == 1
    if bool(direct.any().item()):
        dst = torch.maximum(ii[direct], jj[direct])
        in_range = dst < temporal.shape[1]
        if bool(in_range.any().item()):
            edge_indices = torch.nonzero(direct, as_tuple=False).reshape(-1)[in_range]
            features[:, edge_indices] = temporal[:, dst[in_range]]
    return features


def quat_to_rotvec(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Log-map xyzw quaternions to 3D rotation vectors."""

    q = quat_normalize(q)
    q = torch.where(q[..., 3:4] < 0.0, -q, q)

    xyz = q[..., :3]
    w = q[..., 3:4].clamp(-1.0, 1.0)
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    scale = angle / sin_half.clamp_min(eps)

    small = sin_half < eps
    return torch.where(small, 2.0 * xyz, scale * xyz)


def temporal_pairs(num_poses: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ii = torch.arange(0, max(num_poses - 1, 0), device=device, dtype=torch.long)
    jj = ii + 1
    return ii, jj


def _expand_global_vector(
    value: Optional[torch.Tensor],
    src: torch.Tensor,
    batch: int,
    num_poses: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Expand a shared or per-frame 3-vector onto temporal IMU edges."""

    edges = src.numel()
    if value is None:
        return torch.zeros((batch, edges, 3), device=device, dtype=dtype)

    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.ndim == 1:
        if value.shape[0] != 3:
            raise ValueError(f"{name} must have three entries, got {tuple(value.shape)}")
        return value.view(1, 1, 3).expand(batch, edges, 3)

    if value.ndim == 2:
        if value.shape[-1] != 3:
            raise ValueError(f"{name} last dimension must be three, got {tuple(value.shape)}")
        if value.shape[0] == batch:
            return value[:, None, :].expand(batch, edges, 3)
        if value.shape[0] >= num_poses:
            return value[src][None].expand(batch, edges, 3)

    if value.ndim == 3:
        if value.shape[0] == batch and value.shape[1] >= num_poses and value.shape[2] == 3:
            return value[:, src, :]

    raise ValueError(
        f"{name} must be [3], [B, 3], [N, 3], or [B, N, 3], got {tuple(value.shape)}"
    )


def _as_batched_motion(
    imu_motion: Optional[torch.Tensor],
    batch: int,
    num_poses: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return `[B, N, >=3]` motion values, using zero velocity as a fallback."""

    if imu_motion is None:
        return torch.zeros((batch, num_poses, 3), device=device, dtype=dtype)

    motion = imu_motion.to(device=device, dtype=dtype)
    if motion.ndim == 2:
        motion = motion.unsqueeze(0)
    if motion.ndim != 3 or motion.shape[-1] < 3:
        raise ValueError(f"imu_motion must be [N, >=3] or [B, N, >=3], got {tuple(motion.shape)}")
    if motion.shape[0] == 1 and batch > 1:
        motion = motion.expand(batch, -1, -1)
    if motion.shape[0] != batch or motion.shape[1] < num_poses:
        raise ValueError(
            "imu_motion must match pose batch/frame dimensions, "
            f"got {tuple(motion.shape)} for batch={batch}, frames={num_poses}"
        )
    return motion[:, :num_poses]


def _as_batched_temporal_values(
    name: str,
    values: Optional[torch.Tensor],
    batch: int,
    num_poses: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a `[B, N]` temporal scalar tensor, defaulting to one."""

    if values is None:
        return torch.ones((batch, num_poses), device=device, dtype=dtype)

    values = values.to(device=device, dtype=dtype)
    if values.ndim == 1:
        if values.shape[0] < num_poses:
            raise ValueError(f"{name} needs at least {num_poses} entries, got {tuple(values.shape)}")
        return values[:num_poses][None].expand(batch, -1)
    if values.ndim == 2:
        if values.shape[0] == 1 and batch > 1:
            values = values.expand(batch, -1)
        if values.shape[0] != batch or values.shape[1] < num_poses:
            raise ValueError(
                f"{name} must be [N] or [B, N], got {tuple(values.shape)} "
                f"for batch={batch}, frames={num_poses}"
            )
        return values[:, :num_poses]
    raise ValueError(f"{name} must be [N] or [B, N], got {tuple(values.shape)}")


@torch.no_grad()
def compute_imu_adaptive_gate(
    pose_data: torch.Tensor,
    imu_delta: Optional[torch.Tensor],
    visual_confidence: Optional[torch.Tensor] = None,
    imu_motion: Optional[torch.Tensor] = None,
    imu_bias_jacobian: Optional[torch.Tensor] = None,
    gyro_bias: Optional[torch.Tensor] = None,
    accel_bias: Optional[torch.Tensor] = None,
    gravity: Optional[torch.Tensor] = None,
    imu_optimize_bias: bool = False,
    compose_order: str = "dq_prev",
    inverse_imu: bool = True,
    gate_min: float = 0.2,
    gate_max: float = 1.25,
    visual_threshold: float = 0.2,
    visual_boost: float = 0.25,
    motion_threshold_deg: float = 5.0,
    motion_boost: float = 0.15,
    residual_soft_deg: float = 2.5,
    residual_hard_deg: float = 8.0,
    residual_pos_soft: float = 0.0,
    residual_vel_soft: float = 0.0,
) -> Optional[torch.Tensor]:
    """Compute the bounded deterministic IMU gate used during training.

    The gate follows the runtime policy: it boosts a temporal IMU factor only
    for weak visual evidence or large measured rotation, and attenuates it when
    the current visual and inertial states disagree. It intentionally runs
    without gradients: the gate is a safety rule, while the GRU confidence
    continues to receive gradients through the DBA solve itself.

    Returns:
        `[B, N]` multipliers, where row `j` applies to the IMU interval
        `(j - 1, j)`. The first row is one because it has no predecessor.
    """

    if imu_delta is None:
        return None
    if pose_data.ndim != 3 or pose_data.shape[-1] < 7:
        raise ValueError(f"pose_data must be [B, N, >=7], got {tuple(pose_data.shape)}")

    batch, num_poses = pose_data.shape[:2]
    if num_poses == 0:
        return pose_data.new_ones((batch, 0))

    gate_min = max(0.0, float(gate_min))
    gate_max = max(gate_min, float(gate_max))
    gates = pose_data.new_ones((batch, num_poses))
    if num_poses < 2:
        return gates

    imu, _ = _as_batched("imu_delta", imu_delta, 10)
    imu = imu.to(device=pose_data.device, dtype=pose_data.dtype)
    if imu.shape[0] == 1 and batch > 1:
        imu = imu.expand(batch, -1, -1)
    if imu.shape[0] != batch or imu.shape[1] < num_poses:
        raise ValueError(
            "imu_delta must match pose batch/frame dimensions, "
            f"got imu={tuple(imu.shape)}, poses={tuple(pose_data.shape)}"
        )

    src, dst = temporal_pairs(num_poses, pose_data.device)
    dt = imu[:, dst, 0:1].clamp_min(0.0)
    motion = _as_batched_motion(
        imu_motion, batch, num_poses, pose_data.device, pose_data.dtype
    )
    q_i = pose_data[:, src, 3:7]
    q_j = pose_data[:, dst, 3:7]
    centers = camera_center_from_pose_data(pose_data)
    p_i = centers[:, src]
    p_j = centers[:, dst]

    ba_global = _expand_global_vector(
        accel_bias, src, batch, num_poses, pose_data.device, pose_data.dtype, "accel_bias"
    )
    bg_global = _expand_global_vector(
        gyro_bias, src, batch, num_poses, pose_data.device, pose_data.dtype, "gyro_bias"
    )
    g_world = _expand_global_vector(
        gravity, src, batch, num_poses, pose_data.device, pose_data.dtype, "gravity"
    )

    v_i = motion[:, src, 0:3]
    v_j = motion[:, dst, 0:3]
    if bool(imu_optimize_bias) and motion.shape[-1] >= 9:
        ba_local = motion[:, src, 3:6]
        bg_local = motion[:, src, 6:9]
    else:
        ba_local = torch.zeros_like(v_i)
        bg_local = torch.zeros_like(v_i)
    ba_total = ba_global + ba_local
    bg_total = bg_global + bg_local

    bias_jac = select_imu_bias_jacobian(
        imu_bias_jacobian,
        dst,
        batch,
        num_poses,
        pose_data.device,
        pose_data.dtype,
        dt,
    )
    J_R_bg = bias_jac[:, :, 0]
    J_v_ba = bias_jac[:, :, 1]
    J_v_bg = bias_jac[:, :, 2]
    J_p_ba = bias_jac[:, :, 3]
    J_p_bg = bias_jac[:, :, 4]

    rotvec = imu[:, dst, 1:4] + torch.matmul(J_R_bg, bg_total[..., None]).squeeze(-1)
    dv_imu = (
        imu[:, dst, 4:7]
        + torch.matmul(J_v_ba, ba_total[..., None]).squeeze(-1)
        + torch.matmul(J_v_bg, bg_total[..., None]).squeeze(-1)
    )
    dp_imu = (
        imu[:, dst, 7:10]
        + torch.matmul(J_p_ba, ba_total[..., None]).squeeze(-1)
        + torch.matmul(J_p_bg, bg_total[..., None]).squeeze(-1)
    )

    q_pred = predict_imu_rotation(
        q_i,
        rotvec_to_quat(rotvec),
        compose_order=compose_order,
        inverse_imu=inverse_imu,
    )
    r_R = quat_to_rotvec(quat_multiply(quat_inverse(q_pred), q_j))
    p_world_delta = p_j - p_i - v_i * dt - 0.5 * g_world * dt * dt
    v_world_delta = v_j - v_i - g_world * dt
    r_p = quat_rotate(q_i, p_world_delta) - dp_imu
    r_v = quat_rotate(q_i, v_world_delta) - dv_imu

    visual = _as_batched_temporal_values(
        "visual_confidence",
        visual_confidence,
        batch,
        num_poses,
        pose_data.device,
        pose_data.dtype,
    ).clamp(0.0, 1.0)[:, dst]

    visual_gate = torch.ones_like(visual)
    if visual_threshold > 0.0:
        deficit = ((float(visual_threshold) - visual) / max(float(visual_threshold), 1e-6)).clamp(0.0, 1.0)
        visual_gate = 1.0 + max(0.0, float(visual_boost)) * deficit

    motion_deg = imu[:, dst, 1:4].norm(dim=-1) * (180.0 / torch.pi)
    motion_gate = torch.ones_like(visual)
    if motion_threshold_deg > 0.0:
        excess = (
            (motion_deg - float(motion_threshold_deg))
            / max(float(motion_threshold_deg), 1e-6)
        ).clamp(0.0, 1.0)
        motion_gate = 1.0 + max(0.0, float(motion_boost)) * excess

    def attenuation(value: torch.Tensor, soft: float, hard: float) -> torch.Tensor:
        if soft <= 0.0:
            return torch.ones_like(value)
        hard = max(float(hard), float(soft) + 1e-6)
        alpha = ((value - float(soft)) / (hard - float(soft))).clamp(0.0, 1.0)
        return 1.0 + alpha * (gate_min - 1.0)

    residual_deg = r_R.norm(dim=-1) * (180.0 / torch.pi)
    consistency = attenuation(residual_deg, residual_soft_deg, residual_hard_deg)
    if residual_pos_soft > 0.0:
        consistency = consistency * attenuation(
            r_p.norm(dim=-1), residual_pos_soft, 2.0 * float(residual_pos_soft)
        )
    if residual_vel_soft > 0.0:
        consistency = consistency * attenuation(
            r_v.norm(dim=-1), residual_vel_soft, 2.0 * float(residual_vel_soft)
        )

    gates[:, dst] = (visual_gate * motion_gate * consistency).clamp(gate_min, gate_max)
    return gates


def _as_batched(name: str, tensor: torch.Tensor, min_last_dim: int) -> Tuple[torch.Tensor, bool]:
    if tensor.ndim == 2 and tensor.shape[-1] >= min_last_dim:
        return tensor.unsqueeze(0), True

    if tensor.ndim == 3 and tensor.shape[-1] >= min_last_dim:
        return tensor, False

    raise ValueError(
        f"{name} must have shape [N, >={min_last_dim}] or [B, N, >={min_last_dim}], "
        f"got {tuple(tensor.shape)}"
    )


def _select_gyro_bias(
    gyro_bias: Optional[torch.Tensor],
    jj: torch.Tensor,
    batch: int,
    num_poses: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if gyro_bias is None:
        return None

    bias = gyro_bias.to(device=device, dtype=dtype)
    edges = jj.numel()

    if bias.ndim == 1:
        if bias.shape[0] != 3:
            raise ValueError(f"gyro_bias [3] expected, got {tuple(bias.shape)}")
        return bias.view(1, 1, 3).expand(batch, edges, 3)

    if bias.ndim == 2:
        if bias.shape[-1] != 3:
            raise ValueError(f"gyro_bias last dimension must be 3, got {tuple(bias.shape)}")

        if bias.shape[0] == batch:
            return bias[:, None, :].expand(batch, edges, 3)

        if bias.shape[0] == num_poses:
            return bias[jj][None].expand(batch, edges, 3)

    if bias.ndim == 3:
        if bias.shape[0] != batch or bias.shape[1] != num_poses or bias.shape[2] != 3:
            raise ValueError(
                "gyro_bias [B, N, 3] must match poses batch/frame dimensions, "
                f"got {tuple(bias.shape)} for batch={batch}, num_poses={num_poses}"
            )
        return bias[:, jj, :]

    raise ValueError(
        "gyro_bias must be [3], [B, 3], [N, 3], or [B, N, 3], "
        f"got {tuple(bias.shape)}"
    )


def rotation_only_residual(
    poses: torch.Tensor,
    imu_delta: torch.Tensor,
    imu_valid: Optional[torch.Tensor] = None,
    gyro_bias: Optional[torch.Tensor] = None,
    pairs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    compose_order: str = "prev_dq",
    inverse_imu: bool = False,
    return_mask: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute rotation-only inertial residuals for temporal IMU edges.

    Args:
        poses: `[N, 7]` DROID pose tensor, quaternion layout `[qx, qy, qz, qw]`.
            Batched `[B, N, 7]` is also accepted for training.
        imu_delta: `[N, >=4]` or `[B, N, >=4]`, with columns `[dt, dr_x, dr_y, dr_z, ...]`.
            Row `j` is assumed to describe the preintegrated IMU delta from the
            previous selected keyframe to keyframe `j`.
        imu_valid: optional `[N]` mask. Invalid target rows are skipped.
        gyro_bias: optional gyroscope bias in camera frame. During this first
            rotation+bias stage, correction uses the small-angle approximation
            `dr_corrected = dr - bg * dt`.
        pairs: optional `(ii, jj)` edge tensors. Defaults to consecutive pairs.
        compose_order: `prev_dq` predicts `q_j = q_i * dq`; `dq_prev` predicts
            `q_j = dq * q_i`. The former is the default used by the old
            post-BA regularizer and is the first convention to test.
        inverse_imu: invert `dq` before residual calculation.
        return_mask: if true, return `(residual, valid_edge_mask)`.
    """

    poses_b, _ = _as_batched("poses", poses, 7)
    imu_b, _ = _as_batched("imu_delta", imu_delta, 4)

    if imu_b.shape[0] == 1 and poses_b.shape[0] > 1:
        imu_b = imu_b.expand(poses_b.shape[0], -1, -1)

    if imu_b.shape[0] != poses_b.shape[0] or imu_b.shape[1] != poses_b.shape[1]:
        raise ValueError(
            "imu_delta must match poses batch/frame dimensions, "
            f"got poses={tuple(poses.shape)}, imu_delta={tuple(imu_delta.shape)}"
        )

    batch = poses_b.shape[0]
    num_poses = poses_b.shape[1]

    if pairs is None:
        ii, jj = temporal_pairs(num_poses, poses_b.device)
    else:
        ii, jj = pairs
        ii = ii.to(device=poses_b.device, dtype=torch.long)
        jj = jj.to(device=poses_b.device, dtype=torch.long)

    if ii.numel() == 0:
        empty = poses_b.new_zeros((0, 3))
        if return_mask:
            return empty, torch.zeros(0, device=poses_b.device, dtype=torch.bool)
        return empty

    q_i = poses_b[:, ii, 3:7]
    q_j = poses_b[:, jj, 3:7]
    rotvec = imu_b[:, jj, 1:4].to(dtype=poses_b.dtype, device=poses_b.device)

    selected_bias = _select_gyro_bias(
        gyro_bias,
        jj,
        batch=batch,
        num_poses=num_poses,
        device=poses_b.device,
        dtype=poses_b.dtype,
    )
    if selected_bias is not None:
        dt = imu_b[:, jj, 0:1].to(dtype=poses_b.dtype, device=poses_b.device).clamp_min(0.0)
        rotvec = rotvec - selected_bias * dt

    q_imu = rotvec_to_quat(rotvec)

    q_pred = predict_imu_rotation(
        q_i,
        q_imu,
        compose_order=compose_order,
        inverse_imu=inverse_imu,
    )

    q_err = quat_multiply(quat_inverse(q_pred), q_j)
    residual = quat_to_rotvec(q_err)

    if imu_valid is None:
        mask = torch.ones((batch, ii.shape[0]), dtype=torch.bool, device=poses_b.device)
    else:
        valid = imu_valid.to(device=poses_b.device, dtype=torch.bool)
        if valid.ndim == 1:
            mask = valid[jj][None].expand(batch, -1)
        elif valid.ndim == 2:
            if valid.shape[0] == 1 and batch > 1:
                valid = valid.expand(batch, -1)
            if valid.shape[0] != batch or valid.shape[1] != num_poses:
                raise ValueError(
                    "imu_valid must match poses batch/frame dimensions, "
                    f"got {tuple(imu_valid.shape)} for poses={tuple(poses.shape)}"
                )
            mask = valid[:, jj]
        else:
            raise ValueError(f"imu_valid must be [N] or [B, N], got {tuple(imu_valid.shape)}")

    residual = residual[mask]
    if return_mask:
        return residual, mask.reshape(-1)
    return residual


def rotation_only_cost(
    poses: torch.Tensor,
    imu_delta: torch.Tensor,
    imu_valid: Optional[torch.Tensor] = None,
    gyro_bias: Optional[torch.Tensor] = None,
    sqrt_info: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    residual, mask = rotation_only_residual(
        poses,
        imu_delta,
        imu_valid=imu_valid,
        gyro_bias=gyro_bias,
        return_mask=True,
        **kwargs,
    )

    if sqrt_info is not None and residual.numel() > 0:
        pairs = kwargs.get("pairs")
        poses_b, _ = _as_batched("poses", poses, 7)
        batch, num_poses = poses_b.shape[:2]
        if pairs is None:
            _, jj = temporal_pairs(num_poses, poses_b.device)
        else:
            _, jj = pairs
            jj = jj.to(device=poses_b.device, dtype=torch.long)

        info = sqrt_info.to(device=poses_b.device, dtype=poses_b.dtype)
        if info.ndim == 1:
            edge_info = info[jj][None].expand(batch, -1)
        elif info.ndim == 2:
            if info.shape[0] == 1 and batch > 1:
                info = info.expand(batch, -1)
            edge_info = info[:, jj]
        else:
            raise ValueError(f"sqrt_info must be [N] or [B, N], got {tuple(sqrt_info.shape)}")

        edge_info = edge_info.reshape(-1)[mask]
        residual = residual * edge_info.reshape(-1, 1)

    return 0.5 * torch.sum(residual * residual)


def finite_difference_jacobian(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Central-difference Jacobian helper for small residual tests."""

    x0 = x.detach().clone()
    y0 = fn(x0).detach().reshape(-1)
    J = x0.new_zeros((y0.numel(), x0.numel()))
    flat = x0.reshape(-1)

    for k in range(flat.numel()):
        xp = x0.detach().clone().reshape(-1)
        xm = x0.detach().clone().reshape(-1)
        xp[k] += eps
        xm[k] -= eps
        yp = fn(xp.reshape_as(x0)).detach().reshape(-1)
        ym = fn(xm.reshape_as(x0)).detach().reshape(-1)
        J[:, k] = (yp - ym) / (2.0 * eps)

    return J

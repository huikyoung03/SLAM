"""
Rotation-only IMU factor prototype for DROID-SLAM.

This module is intentionally independent from the CUDA DBA backend. It provides
the first testable form of the inertial residual:

    r_R = Log((q_i * Deltaq_imu)^-1 * q_j)

where DROID poses store quaternions in xyzw order. The next integration step is
to use this residual to assemble an IMU Hessian/gradient block and add that
block to the visual BA system before Schur complement.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch


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

    if inverse_imu:
        q_imu = quat_inverse(q_imu)

    if compose_order == "prev_dq":
        q_pred = quat_multiply(q_i, q_imu)
    elif compose_order == "dq_prev":
        q_pred = quat_multiply(q_imu, q_i)
    else:
        raise ValueError(f"Unknown compose_order: {compose_order}")

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

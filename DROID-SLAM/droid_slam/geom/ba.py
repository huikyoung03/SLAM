import lietorch
import torch
import torch.nn.functional as F

from .chol import block_solve, schur_solve
from .imu_factor import (
    quat_inverse,
    quat_multiply,
    camera_center_from_pose_data,
    imu_center_from_pose_data,
    predict_imu_rotation,
    quat_rotate,
    quat_to_rotmat,
    quat_to_rotvec,
    rotvec_to_quat,
    select_imu_bias_jacobian,
    skew_symmetric,
    so3_left_jacobian,
    so3_left_jacobian_inv,
)
import geom.projective_ops as pops

from torch_scatter import scatter_sum


# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))


def motion_retr(motion, dx, ii):
    ii = ii.to(device=dx.device)
    update = scatter_sum(dx, ii, dim=1, dim_size=motion.shape[1])
    if update.shape[-1] == motion.shape[-1]:
        return motion + update

    out = motion.clone()
    out[..., :update.shape[-1]] = out[..., :update.shape[-1]] + update
    return out


def _expand_vector_param(param, src, batch, num_frames, device, dtype, name):
    if param is None:
        return torch.zeros((batch, src.numel(), 3), device=device, dtype=dtype)

    if torch.is_tensor(param):
        value = param.to(device=device, dtype=dtype)
    else:
        value = torch.as_tensor(param, device=device, dtype=dtype)
    if value.ndim == 1:
        if value.shape[0] != 3:
            raise ValueError(f"{name} [3] expected, got {tuple(value.shape)}")
        return value.view(1, 1, 3).expand(batch, src.numel(), 3)

    if value.ndim == 2:
        if value.shape[-1] != 3:
            raise ValueError(f"{name} last dimension must be 3, got {tuple(value.shape)}")
        if value.shape[0] == batch:
            return value[:, None, :].expand(batch, src.numel(), 3)
        if value.shape[0] == num_frames:
            return value[src][None].expand(batch, src.numel(), 3)

    if value.ndim == 3:
        if value.shape[0] != batch or value.shape[1] < num_frames or value.shape[2] != 3:
            raise ValueError(
                f"{name} [B, N, 3] must match BA frame dimensions, "
                f"got {tuple(value.shape)} for batch={batch}, frames={num_frames}"
            )
        return value[:, src, :]

    raise ValueError(
        f"{name} must be [3], [B, 3], [N, 3], or [B, N, 3], got {tuple(value.shape)}"
    )


def _select_temporal_mask(imu_valid, dst, batch, num_frames, device):
    if imu_valid is None:
        return torch.ones((batch, dst.numel()), device=device, dtype=torch.bool)

    valid = imu_valid.to(device=device, dtype=torch.bool)
    if valid.ndim == 1:
        return valid[dst][None].expand(batch, -1)

    if valid.ndim == 2:
        if valid.shape[0] == 1 and batch > 1:
            valid = valid.expand(batch, -1)
        if valid.shape[0] != batch or valid.shape[1] < num_frames:
            raise ValueError(
                "imu_valid must have shape [N] or [B, N] and match BA poses, "
                f"got {tuple(imu_valid.shape)} for frames={num_frames}"
            )
        return valid[:, dst]

    raise ValueError(f"imu_valid must be [N] or [B, N], got {tuple(imu_valid.shape)}")


def _select_temporal_confidence(imu_confidence, dst, batch, num_frames, device, dtype):
    """Select scalar or component IMU confidence for temporal factors.

    The returned layout is always ``[B, E, C]`` where ``C`` is one for legacy
    checkpoints or three in the fixed order ``[rotation, velocity, position]``.
    Keeping the component axis lets the full inertial system weight each
    residual block without changing the surrounding Schur solve.
    """

    if imu_confidence is None:
        return torch.ones((batch, dst.numel(), 1), device=device, dtype=dtype)

    confidence = imu_confidence.to(device=device, dtype=dtype)
    if confidence.ndim == 1:
        return confidence[dst][None, :, None].expand(batch, -1, -1)

    if confidence.ndim == 2:
        # Unbatched component confidence stored by the runtime cache: [N, 3].
        if confidence.shape[-1] == 3 and confidence.shape[0] >= num_frames:
            return confidence[dst][None].expand(batch, -1, -1)
        if confidence.shape[0] == 1 and batch > 1:
            confidence = confidence.expand(batch, -1)
        if confidence.shape[0] != batch or confidence.shape[1] < num_frames:
            raise ValueError(
                "imu_confidence must have shape [N] or [B, N] and match BA poses, "
                f"got {tuple(imu_confidence.shape)} for frames={num_frames}"
            )
        return confidence[:, dst, None]

    if confidence.ndim == 3:
        if confidence.shape[0] == 1 and batch > 1:
            confidence = confidence.expand(batch, -1, -1)
        if (
            confidence.shape[0] != batch
            or confidence.shape[1] < num_frames
            or confidence.shape[-1] not in (1, 3)
        ):
            raise ValueError(
                "imu_confidence must have shape [B, N, 1|3] and match BA poses, "
                f"got {tuple(imu_confidence.shape)} for frames={num_frames}"
            )
        return confidence[:, dst]

    raise ValueError(
        "imu_confidence must be [N], [B, N], [N, 3], or [B, N, 1|3], "
        f"got {tuple(imu_confidence.shape)}"
    )


def _select_temporal_gate(imu_adaptive_gate, dst, batch, num_frames, device, dtype):
    """Select an optional per-frame deterministic IMU gate for temporal edges."""

    if imu_adaptive_gate is None:
        return torch.ones((batch, dst.numel()), device=device, dtype=dtype)

    gate = imu_adaptive_gate.to(device=device, dtype=dtype)
    if gate.ndim == 1:
        if gate.shape[0] < num_frames:
            raise ValueError(
                "imu_adaptive_gate must cover every BA frame, "
                f"got {tuple(gate.shape)} for frames={num_frames}"
            )
        return gate[dst][None].expand(batch, -1)

    if gate.ndim == 2:
        if gate.shape[0] == 1 and batch > 1:
            gate = gate.expand(batch, -1)
        if gate.shape[0] != batch or gate.shape[1] < num_frames:
            raise ValueError(
                "imu_adaptive_gate must have shape [N] or [B, N] and match BA poses, "
                f"got {tuple(gate.shape)} for frames={num_frames}"
            )
        return gate[:, dst]

    raise ValueError(
        f"imu_adaptive_gate must be [N] or [B, N], got {tuple(gate.shape)}"
    )


def _select_temporal_info_scale(
    imu_info,
    imu_info_reference,
    dst,
    batch,
    num_frames,
    device,
    dtype,
    clip=4.0,
    eps=1e-12,
):
    """
    Select per-edge IMU information and convert it to residual scale factors.

    `imu_info` is stored as `[rot_info, vel_info, pos_info]`, where each value
    is proportional to inverse variance. `imu_info_reference`, when supplied,
    is the median information of valid *raw camera-rate* intervals in the same
    session. That preserves uncertainty growth when a training sample composes
    several raw intervals. Older callers without a reference retain the local
    median fallback.
    """

    ones = torch.ones((batch, dst.numel(), 3), device=device, dtype=dtype)
    if imu_info is None:
        return ones

    info = imu_info.to(device=device, dtype=dtype)
    if info.ndim == 2:
        if info.shape[0] < num_frames or info.shape[-1] < 3:
            raise ValueError(
                "imu_info must have shape [N, >=3] or [B, N, >=3], "
                f"got {tuple(imu_info.shape)} for frames={num_frames}"
            )
        selected = info[dst, :3][None].expand(batch, -1, -1)

    elif info.ndim == 3:
        if info.shape[0] == 1 and batch > 1:
            info = info.expand(batch, -1, -1)
        if info.shape[0] != batch or info.shape[1] < num_frames or info.shape[-1] < 3:
            raise ValueError(
                "imu_info must have shape [N, >=3] or [B, N, >=3], "
                f"got {tuple(imu_info.shape)} for batch={batch}, frames={num_frames}"
            )
        selected = info[:, dst, :3]

    else:
        raise ValueError(f"imu_info must be [N, >=3] or [B, N, >=3], got {tuple(imu_info.shape)}")

    valid = torch.isfinite(selected) & (selected > float(eps))
    if not bool(valid.any().detach().cpu().item()):
        return ones

    ref = torch.ones((batch, 1, 3), device=device, dtype=dtype)
    reference_valid = torch.zeros((batch, 3), device=device, dtype=torch.bool)
    if imu_info_reference is not None:
        if torch.is_tensor(imu_info_reference):
            reference = imu_info_reference.to(device=device, dtype=dtype)
        else:
            reference = torch.as_tensor(imu_info_reference, device=device, dtype=dtype)

        if reference.ndim == 1:
            if reference.numel() < 3:
                raise ValueError(
                    "imu_info_reference must contain [rot, vel, pos] information"
                )
            reference = reference[:3].view(1, 3).expand(batch, -1)
        elif reference.ndim == 2:
            if reference.shape[-1] < 3:
                raise ValueError(
                    "imu_info_reference must have shape [B, >=3]"
                )
            if reference.shape[0] == 1 and batch > 1:
                reference = reference.expand(batch, -1)
            if reference.shape[0] != batch:
                raise ValueError(
                    "imu_info_reference batch dimension must match IMU batch, "
                    f"got {tuple(reference.shape)} for batch={batch}"
                )
            reference = reference[:, :3]
        else:
            raise ValueError(
                "imu_info_reference must have shape [3] or [B, 3], "
                f"got {tuple(reference.shape)}"
            )

        reference_valid = torch.isfinite(reference) & (reference > float(eps))
        ref[:, 0, :] = torch.where(reference_valid, reference, ref[:, 0, :])

    for b in range(batch):
        for c in range(3):
            if not bool(reference_valid[b, c].item()):
                values = selected[b, :, c][valid[b, :, c]]
                if values.numel() > 0:
                    ref[b, 0, c] = values.median().clamp_min(float(eps))

    ratio = torch.where(valid, selected / ref.clamp_min(float(eps)), ones)
    if clip is not None and float(clip) > 0.0:
        clip_value = float(clip)
        ratio = ratio.clamp(min=1.0 / clip_value, max=clip_value)
    else:
        ratio = ratio.clamp_min(0.0)

    return ratio.sqrt()


def _global_factor_weight(factor_weight, batch, device, dtype):
    if torch.is_tensor(factor_weight):
        global_weight = factor_weight.to(device=device, dtype=dtype)
    else:
        global_weight = torch.tensor(float(factor_weight), device=device, dtype=dtype)

    if global_weight.ndim == 0:
        return global_weight.view(1, 1)

    if global_weight.ndim == 1:
        if global_weight.shape[0] == batch:
            return global_weight.view(batch, 1)
        raise ValueError(
            f"imu_factor_weight [B] must match batch={batch}, got {tuple(global_weight.shape)}"
        )

    if global_weight.ndim == 2:
        return global_weight

    raise ValueError(
        "imu_factor_weight must be scalar, [B], or [B, 1], "
        f"got {tuple(global_weight.shape)}"
    )


def _build_rotation_imu_system(
    poses,
    imu_delta,
    imu_valid=None,
    gyro_bias=None,
    fixedp=1,
    rig=1,
    num_opt=0,
    dim=6,
    factor_weight=0.0,
    imu_confidence=None,
    imu_adaptive_gate=None,
    max_residual=0.5,
):
    """Build rotation-only IMU normal-equation blocks for pose variables."""

    if imu_delta is None or dim < 6:
        return None, None

    if not torch.is_tensor(factor_weight) and factor_weight <= 0.0:
        return None, None

    pose_data = poses.data
    B = pose_data.shape[0]
    num_frames = pose_data.shape[1] // rig

    if num_frames < 2 or num_opt <= 0:
        H = pose_data.new_zeros((B, num_opt * num_opt, dim, dim))
        v = pose_data.new_zeros((B, num_opt, dim))
        return H, v

    imu = imu_delta.to(device=pose_data.device, dtype=pose_data.dtype)
    if imu.ndim == 2:
        imu = imu.unsqueeze(0)
    if imu.shape[0] == 1 and B > 1:
        imu = imu.expand(B, -1, -1)

    if imu.shape[0] != B or imu.shape[1] < num_frames or imu.shape[-1] < 4:
        raise ValueError(
            "imu_delta must have shape [B, N, >=4] and match BA poses, "
            f"got imu_delta={tuple(imu_delta.shape)}, poses={tuple(pose_data.shape)}"
        )

    src = torch.arange(0, num_frames - 1, device=pose_data.device, dtype=torch.long)
    dst = src + 1

    q_i = pose_data[:, src, 3:7]
    q_j = pose_data[:, dst, 3:7]
    rotvec = imu[:, dst, 1:4]

    if gyro_bias is not None:
        bias = gyro_bias.to(device=pose_data.device, dtype=pose_data.dtype)
        if bias.ndim == 1:
            bias = bias.view(1, 1, 3).expand(B, dst.numel(), 3)
        elif bias.ndim == 2 and bias.shape[0] == B:
            bias = bias[:, None, :].expand(B, dst.numel(), 3)
        elif bias.ndim == 2 and bias.shape[0] == num_frames:
            bias = bias[dst][None].expand(B, dst.numel(), 3)
        elif bias.ndim == 3:
            bias = bias[:, dst, :]
        else:
            raise ValueError(f"unsupported gyro_bias shape: {tuple(gyro_bias.shape)}")

        dt = imu[:, dst, 0:1].clamp_min(0.0)
        rotvec = rotvec - bias * dt

    q_imu = rotvec_to_quat(rotvec)
    q_pred = quat_multiply(q_i, q_imu)
    q_err = quat_multiply(quat_inverse(q_pred), q_j)
    residual = quat_to_rotvec(q_err)

    if imu_valid is None:
        edge_mask = torch.ones((B, dst.numel()), device=pose_data.device, dtype=torch.bool)
    else:
        valid = imu_valid.to(device=pose_data.device, dtype=torch.bool)
        if valid.ndim == 1:
            edge_mask = valid[dst][None].expand(B, -1)
        elif valid.ndim == 2:
            if valid.shape[0] == 1 and B > 1:
                valid = valid.expand(B, -1)
            edge_mask = valid[:, dst]
        else:
            raise ValueError(f"imu_valid must be [N] or [B, N], got {tuple(imu_valid.shape)}")

    if max_residual is not None and max_residual > 0.0:
        edge_mask = edge_mask & (residual.norm(dim=-1) <= float(max_residual))

    if torch.is_tensor(factor_weight):
        global_weight = factor_weight.to(device=pose_data.device, dtype=pose_data.dtype)
    else:
        global_weight = pose_data.new_tensor(float(factor_weight))

    if global_weight.ndim == 0:
        global_weight = global_weight.view(1, 1)
    elif global_weight.ndim == 1:
        if global_weight.shape[0] == B:
            global_weight = global_weight.view(B, 1)
        else:
            raise ValueError(
                f"imu_factor_weight [B] must match batch={B}, got {tuple(global_weight.shape)}"
            )
    elif global_weight.ndim != 2:
        raise ValueError(
            "imu_factor_weight must be scalar, [B], or [B, 1], "
            f"got {tuple(global_weight.shape)}"
        )

    confidence = _select_temporal_confidence(
        imu_confidence, dst, B, num_frames, pose_data.device, pose_data.dtype
    )[:, :, 0]

    adaptive_gate = _select_temporal_gate(
        imu_adaptive_gate, dst, B, num_frames, pose_data.device, pose_data.dtype
    )
    edge_weight = global_weight * confidence * adaptive_gate * edge_mask.to(dtype=pose_data.dtype)

    Ji = pose_data.new_zeros((B, dst.numel(), 3, dim))
    Jj = pose_data.new_zeros((B, dst.numel(), 3, dim))
    R_pred = quat_to_rotmat(q_pred)
    rot_map = torch.matmul(
        so3_left_jacobian_inv(residual),
        R_pred.transpose(-1, -2),
    )

    # BA solves residuals as r - J dx. DROID/LieTorch uses left SE(3)
    # perturbations, so rotation-only IMU also needs the current pose frame.
    Ji[:, :, :, 3:6] = rot_map
    Jj[:, :, :, 3:6] = -rot_map

    Hii = edge_weight[:, :, None, None] * torch.einsum("bead,beaf->bedf", Ji, Ji)
    Hij = edge_weight[:, :, None, None] * torch.einsum("bead,beaf->bedf", Ji, Jj)
    Hji = edge_weight[:, :, None, None] * torch.einsum("bead,beaf->bedf", Jj, Ji)
    Hjj = edge_weight[:, :, None, None] * torch.einsum("bead,beaf->bedf", Jj, Jj)

    vi = edge_weight[:, :, None] * torch.einsum("bead,bea->bed", Ji, residual)
    vj = edge_weight[:, :, None] * torch.einsum("bead,bea->bed", Jj, residual)

    src_opt = src // rig - fixedp
    dst_opt = dst // rig - fixedp

    H = safe_scatter_add_mat(Hii, src_opt, src_opt, num_opt, num_opt) + \
        safe_scatter_add_mat(Hij, src_opt, dst_opt, num_opt, num_opt) + \
        safe_scatter_add_mat(Hji, dst_opt, src_opt, num_opt, num_opt) + \
        safe_scatter_add_mat(Hjj, dst_opt, dst_opt, num_opt, num_opt)

    v = safe_scatter_add_vec(vi, src_opt, num_opt) + \
        safe_scatter_add_vec(vj, dst_opt, num_opt)

    return H, v


def _build_full_imu_system(
    poses,
    imu_motion,
    imu_delta,
    imu_valid=None,
    imu_info=None,
    imu_info_reference=None,
    imu_bias_jacobian=None,
    gyro_bias=None,
    accel_bias=None,
    imu_camera_translation=None,
    fixedp=1,
    rig=1,
    num_opt=0,
    state_dim=15,
    pose_dim=6,
    factor_weight=0.0,
    imu_confidence=None,
    imu_adaptive_gate=None,
    max_residual=0.5,
    pos_weight=0.05,
    vel_weight=0.05,
    rot_weight=1.0,
    bias_weight=0.001,
    gravity=None,
    imu_compose_order="dq_prev",
    imu_inverse=True,
    use_imu_info_weighting=False,
    imu_info_weight_clip=4.0,
    imu_info_weight_eps=1e-12,
    imu_optimize_bias=True,
):
    """Build first-order full IMU normal-equation blocks.

    State layout is `[pose(6), velocity(3), accel_bias(3), gyro_bias(3)]`.
    The residual layout is `[r_p, r_v, r_R, r_ba, r_bg]`. The Jacobians match
    DROID/LieTorch's left SE(3) pose perturbation. When available, propagated
    preintegration bias Jacobians are used for the bias correction terms; older
    priors fall back to the first-order `-dt I` approximation.
    """

    if imu_delta is None or imu_motion is None:
        return None, None

    if not torch.is_tensor(factor_weight) and factor_weight <= 0.0:
        return None, None

    pose_data = poses.data
    B = pose_data.shape[0]
    num_frames = pose_data.shape[1] // rig

    if num_frames < 2 or num_opt <= 0:
        H = pose_data.new_zeros((B, num_opt * num_opt, state_dim, state_dim))
        v = pose_data.new_zeros((B, num_opt, state_dim))
        return H, v

    imu = imu_delta.to(device=pose_data.device, dtype=pose_data.dtype)
    if imu.ndim == 2:
        imu = imu.unsqueeze(0)
    if imu.shape[0] == 1 and B > 1:
        imu = imu.expand(B, -1, -1)

    if imu.shape[0] != B or imu.shape[1] < num_frames or imu.shape[-1] < 10:
        raise ValueError(
            "full IMU BA requires imu_delta with shape [B, N, >=10], "
            f"got imu_delta={tuple(imu_delta.shape)}, poses={tuple(pose_data.shape)}"
        )

    imu_optimize_bias = bool(imu_optimize_bias)
    motion = imu_motion.to(device=pose_data.device, dtype=pose_data.dtype)
    if motion.ndim == 2:
        motion = motion.unsqueeze(0)
    if motion.shape[0] == 1 and B > 1:
        motion = motion.expand(B, -1, -1)
    required_motion_dim = 9 if imu_optimize_bias else 3
    if motion.shape[0] != B or motion.shape[1] < num_frames or motion.shape[-1] < required_motion_dim:
        raise ValueError(
            "imu_motion must contain [v] or [v, ba, bg] depending on IMU BA mode, "
            f"got {tuple(imu_motion.shape)} for poses={tuple(pose_data.shape)}"
        )

    src = torch.arange(0, num_frames - 1, device=pose_data.device, dtype=torch.long)
    dst = src + 1
    edge_count = dst.numel()

    dt = imu[:, dst, 0:1].clamp_min(0.0)
    q_i = pose_data[:, src, 3:7]
    q_j = pose_data[:, dst, 3:7]
    centers = imu_center_from_pose_data(pose_data, imu_camera_translation)
    p_i = centers[:, src]
    p_j = centers[:, dst]
    lever_i = _expand_vector_param(
        imu_camera_translation,
        src,
        B,
        num_frames,
        pose_data.device,
        pose_data.dtype,
        "imu_camera_translation",
    )
    lever_j = _expand_vector_param(
        imu_camera_translation,
        dst,
        B,
        num_frames,
        pose_data.device,
        pose_data.dtype,
        "imu_camera_translation",
    )

    v_i = motion[:, src, 0:3]
    v_j = motion[:, dst, 0:3]
    if imu_optimize_bias:
        ba_i = motion[:, src, 3:6]
        ba_j = motion[:, dst, 3:6]
        bg_i = motion[:, src, 6:9]
        bg_j = motion[:, dst, 6:9]
    else:
        ba_i = torch.zeros_like(v_i)
        ba_j = torch.zeros_like(v_j)
        bg_i = torch.zeros_like(v_i)
        bg_j = torch.zeros_like(v_j)

    ba_global = _expand_vector_param(
        accel_bias, src, B, num_frames, pose_data.device, pose_data.dtype, "accel_bias"
    )
    bg_global = _expand_vector_param(
        gyro_bias, src, B, num_frames, pose_data.device, pose_data.dtype, "gyro_bias"
    )
    g_world = _expand_vector_param(
        gravity, src, B, num_frames, pose_data.device, pose_data.dtype, "gravity"
    )

    ba_total = ba_global + (ba_i if imu_optimize_bias else 0.0)
    bg_total = bg_global + (bg_i if imu_optimize_bias else 0.0)

    bias_jac = select_imu_bias_jacobian(
        imu_bias_jacobian,
        dst,
        B,
        num_frames,
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

    q_imu = rotvec_to_quat(rotvec)
    q_pred = predict_imu_rotation(
        q_i,
        q_imu,
        compose_order=imu_compose_order,
        inverse_imu=imu_inverse,
    )
    r_R = quat_to_rotvec(quat_multiply(quat_inverse(q_pred), q_j))

    p_world_delta = p_j - p_i - v_i * dt - 0.5 * g_world * dt * dt
    v_world_delta = v_j - v_i - g_world * dt
    p_cam_delta = quat_rotate(q_i, p_world_delta)
    v_cam_delta = quat_rotate(q_i, v_world_delta)

    r_p = p_cam_delta - dp_imu
    r_v = v_cam_delta - dv_imu
    r_ba = ba_j - ba_i
    r_bg = bg_j - bg_i

    residual = (
        torch.cat([r_p, r_v, r_R, r_ba, r_bg], dim=-1)
        if imu_optimize_bias
        else torch.cat([r_p, r_v, r_R], dim=-1)
    )

    edge_mask = _select_temporal_mask(imu_valid, dst, B, num_frames, pose_data.device)
    if max_residual is not None and max_residual > 0.0:
        edge_mask = edge_mask & (r_R.norm(dim=-1) <= float(max_residual))

    global_weight = _global_factor_weight(factor_weight, B, pose_data.device, pose_data.dtype)
    confidence = _select_temporal_confidence(
        imu_confidence, dst, B, num_frames, pose_data.device, pose_data.dtype
    )
    if confidence.shape[-1] == 1:
        confidence = confidence.expand(-1, -1, 3)
    elif confidence.shape[-1] != 3:
        raise ValueError(
            "full IMU confidence must have one legacy or three component channels, "
            f"got {tuple(confidence.shape)}"
        )
    adaptive_gate = _select_temporal_gate(
        imu_adaptive_gate, dst, B, num_frames, pose_data.device, pose_data.dtype
    )
    # Confidence is applied per residual block below.  The scalar edge term
    # carries only global factor strength, deterministic gating, and validity.
    edge_weight = global_weight * adaptive_gate * edge_mask.to(dtype=pose_data.dtype)

    residual_dim = 15 if imu_optimize_bias else 9
    Ji = pose_data.new_zeros((B, edge_count, residual_dim, state_dim))
    Jj = pose_data.new_zeros((B, edge_count, residual_dim, state_dim))
    eye = torch.eye(3, device=pose_data.device, dtype=pose_data.dtype)
    I = eye.view(1, 1, 3, 3)

    R_i = quat_to_rotmat(q_i)
    R_j = quat_to_rotmat(q_j)
    R_pred = quat_to_rotmat(q_pred)
    Ri_RjT = torch.matmul(R_i, R_j.transpose(-1, -2))
    Jl_inv = so3_left_jacobian_inv(r_R)

    # DROID/LieTorch left-multiplies SE(3) perturbations. BA solves residuals as
    # r - J dx, so these blocks are `-dr/dx` for that perturbation model.
    Ji[:, :, 0:3, 0:3] = -I
    Jj[:, :, 0:3, 0:3] = Ri_RjT
    # Position is measured at the IMU origin, not the camera centre.  The
    # camera-IMU lever arm contributes to both endpoint rotation Jacobians
    # under DROID/LieTorch's left SE(3) perturbation.
    Ji[:, :, 0:3, 3:6] = skew_symmetric(p_cam_delta + lever_i)
    Jj[:, :, 0:3, 3:6] = -torch.matmul(Ri_RjT, skew_symmetric(lever_j))
    Ji[:, :, 0:3, 6:9] = dt[:, :, None] * R_i
    if imu_optimize_bias:
        Ji[:, :, 0:3, 9:12] = J_p_ba
        Ji[:, :, 0:3, 12:15] = J_p_bg

    Ji[:, :, 3:6, 3:6] = skew_symmetric(v_cam_delta)
    Ji[:, :, 3:6, 6:9] = R_i
    Jj[:, :, 3:6, 6:9] = -R_i
    if imu_optimize_bias:
        Ji[:, :, 3:6, 9:12] = J_v_ba
        Ji[:, :, 3:6, 12:15] = J_v_bg

    if imu_compose_order == "dq_prev":
        rot_i_map = R_i.transpose(-1, -2)
        gyro_map = R_pred.transpose(-1, -2)
    elif imu_compose_order == "prev_dq":
        rot_i_map = R_pred.transpose(-1, -2)
        gyro_map = torch.matmul(R_pred.transpose(-1, -2), R_i)
    else:
        raise ValueError(f"Unknown imu_compose_order: {imu_compose_order}")

    Ji[:, :, 6:9, 3:6] = torch.matmul(Jl_inv, rot_i_map)
    Jj[:, :, 6:9, 3:6] = -torch.matmul(Jl_inv, R_pred.transpose(-1, -2))
    if imu_optimize_bias:
        gyro_sign = -1.0 if imu_inverse else 1.0
        preintegration_map = gyro_sign * so3_left_jacobian(
            gyro_sign * rotvec
        )
        Ji[:, :, 6:9, 12:15] = (
            torch.matmul(
                torch.matmul(torch.matmul(Jl_inv, gyro_map), preintegration_map),
                J_R_bg,
            )
        )

        Ji[:, :, 9:12, 9:12] = I
        Jj[:, :, 9:12, 9:12] = -I

        Ji[:, :, 12:15, 12:15] = I
        Jj[:, :, 12:15, 12:15] = -I

    def scalar_weight(value, name):
        if torch.is_tensor(value):
            value = value.to(device=pose_data.device, dtype=pose_data.dtype)
            if value.numel() != 1:
                raise ValueError(f"{name} must be scalar, got {tuple(value.shape)}")
            return value.reshape(())
        return pose_data.new_tensor(float(value))

    pos_weight = scalar_weight(pos_weight, "pos_weight")
    vel_weight = scalar_weight(vel_weight, "vel_weight")
    rot_weight = scalar_weight(rot_weight, "rot_weight")
    bias_weight = scalar_weight(bias_weight, "bias_weight")
    base_weight_terms = [
        pos_weight.expand(3),
        vel_weight.expand(3),
        rot_weight.expand(3),
    ]
    if imu_optimize_bias:
        base_weight_terms.extend([bias_weight.expand(3), bias_weight.expand(3)])
    base_scale = torch.cat(base_weight_terms).clamp_min(0.0).sqrt()

    if use_imu_info_weighting:
        info_scale = _select_temporal_info_scale(
            imu_info,
            imu_info_reference,
            dst,
            B,
            num_frames,
            pose_data.device,
            pose_data.dtype,
            clip=imu_info_weight_clip,
            eps=imu_info_weight_eps,
        )
        rot_scale = info_scale[:, :, 0:1]
        vel_scale = info_scale[:, :, 1:2]
        pos_scale = info_scale[:, :, 2:3]
        residual_info_scale = torch.cat(
            [
                pos_scale.expand(-1, -1, 3),
                vel_scale.expand(-1, -1, 3),
                rot_scale.expand(-1, -1, 3),
                torch.ones(
                    (B, edge_count, 6 if imu_optimize_bias else 0),
                    device=pose_data.device,
                    dtype=pose_data.dtype,
                ),
            ],
            dim=-1,
        )
    else:
        residual_info_scale = pose_data.new_ones((B, edge_count, residual_dim))

    # Channel order is [rotation, velocity, position], while the factor rows
    # are [position, velocity, rotation, accel-bias, gyro-bias].  Multiplying
    # residual and Jacobian by sqrt(confidence) yields the standard weighted
    # least-squares / Mahalanobis normal equations without ad-hoc H scaling.
    rot_confidence = confidence[:, :, 0:1]
    vel_confidence = confidence[:, :, 1:2]
    pos_confidence = confidence[:, :, 2:3]
    bias_confidence = confidence.mean(dim=-1, keepdim=True)
    confidence_scale = torch.cat(
        [
            pos_confidence.expand(-1, -1, 3),
            vel_confidence.expand(-1, -1, 3),
            rot_confidence.expand(-1, -1, 3),
            bias_confidence.expand(-1, -1, 6 if imu_optimize_bias else 0),
        ],
        dim=-1,
    ).clamp(0.0, 1.0).sqrt()

    scale = base_scale.view(1, 1, residual_dim) * residual_info_scale * confidence_scale
    residual = residual * scale
    Ji = Ji * scale.unsqueeze(-1)
    Jj = Jj * scale.unsqueeze(-1)

    Hii = edge_weight[:, :, None, None] * torch.einsum("bead,beaf->bedf", Ji, Ji)
    Hij = edge_weight[:, :, None, None] * torch.einsum("bead,beaf->bedf", Ji, Jj)
    Hji = edge_weight[:, :, None, None] * torch.einsum("bead,beaf->bedf", Jj, Ji)
    Hjj = edge_weight[:, :, None, None] * torch.einsum("bead,beaf->bedf", Jj, Jj)

    vi = edge_weight[:, :, None] * torch.einsum("bead,bea->bed", Ji, residual)
    vj = edge_weight[:, :, None] * torch.einsum("bead,bea->bed", Jj, residual)

    src_opt = src // rig - fixedp
    dst_opt = dst // rig - fixedp

    H = safe_scatter_add_mat(Hii, src_opt, src_opt, num_opt, num_opt) + \
        safe_scatter_add_mat(Hij, src_opt, dst_opt, num_opt, num_opt) + \
        safe_scatter_add_mat(Hji, dst_opt, src_opt, num_opt, num_opt) + \
        safe_scatter_add_mat(Hjj, dst_opt, dst_opt, num_opt, num_opt)

    v = safe_scatter_add_vec(vi, src_opt, num_opt) + \
        safe_scatter_add_vec(vj, dst_opt, num_opt)

    return H, v


def _build_motion_prior_system(
    imu_motion,
    imu_motion_prior=None,
    fixedp=1,
    rig=1,
    num_opt=0,
    state_dim=15,
    velocity_weight=0.0,
    local_bias_weight=0.0,
):
    """Keep full-IMU auxiliary motion states near their initialized values."""

    if imu_motion is None:
        return None, None

    if velocity_weight <= 0.0 and local_bias_weight <= 0.0:
        return None, None

    motion = imu_motion
    B, num_frames, _ = motion.shape
    if num_opt <= 0:
        H = motion.new_zeros((B, num_opt * num_opt, state_dim, state_dim))
        v = motion.new_zeros((B, num_opt, state_dim))
        return H, v

    if imu_motion_prior is None:
        prior = torch.zeros_like(motion)
    else:
        prior = imu_motion_prior.to(device=motion.device, dtype=motion.dtype)
        if prior.ndim == 2:
            prior = prior.unsqueeze(0)
        if prior.shape[0] == 1 and B > 1:
            prior = prior.expand(B, -1, -1)
        if prior.shape != motion.shape:
            raise ValueError(
                "imu_motion_prior must match imu_motion shape, "
                f"got prior={tuple(prior.shape)}, motion={tuple(motion.shape)}"
            )

    residual = motion - prior
    J = motion.new_zeros((B, num_frames, 9, state_dim))
    eye = torch.eye(3, device=motion.device, dtype=motion.dtype).view(1, 1, 3, 3)

    scales = motion.new_zeros(9)
    if velocity_weight > 0.0:
        scales[0:3] = float(velocity_weight) ** 0.5
        J[:, :, 0:3, 6:9] = -eye
    if local_bias_weight > 0.0 and state_dim >= 15:
        scales[3:9] = float(local_bias_weight) ** 0.5
        J[:, :, 3:6, 9:12] = -eye
        J[:, :, 6:9, 12:15] = -eye

    residual = residual * scales.view(1, 1, 9)
    J = J * scales.view(1, 1, 9, 1)

    Hii = torch.einsum("bnad,bnaf->bndf", J, J)
    vi = torch.einsum("bnad,bna->bnd", J, residual)

    frame_idx = torch.arange(0, num_frames, device=motion.device, dtype=torch.long)
    opt_idx = frame_idx // rig - fixedp
    H = safe_scatter_add_mat(Hii, opt_idx, opt_idx, num_opt, num_opt)
    v = safe_scatter_add_vec(vi, opt_idx, num_opt)

    return H, v


def BA(
    target,
    weight,
    eta,
    poses,
    disps,
    intrinsics,
    ii,
    jj,
    fixedp=1,
    rig=1,
    imu_delta=None,
    imu_valid=None,
    imu_info=None,
    imu_info_reference=None,
    imu_bias_jacobian=None,
    gyro_bias=None,
    imu_factor_weight=0.0,
    imu_confidence=None,
    imu_adaptive_gate=None,
    imu_max_residual=0.5,
    imu_motion=None,
    imu_motion_prior=None,
    accel_bias=None,
    imu_camera_translation=None,
    use_full_imu=False,
    imu_full_pos_weight=0.05,
    imu_full_vel_weight=0.05,
    imu_full_bias_weight=0.001,
    imu_motion_prior_weight=0.0,
    imu_local_bias_prior_weight=0.0,
    imu_gravity=None,
    imu_full_compose_order="dq_prev",
    imu_full_inverse=True,
    use_imu_info_weighting=False,
    imu_info_weight_clip=4.0,
    imu_info_weight_eps=1e-12,
    imu_optimize_bias=True,
):
    """ Full Bundle Adjustment """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1)
    w = .001 * (valid * weight).view(B, N, -1, 1)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2,3)
    wJjT = (w * Jj).transpose(2,3)

    Jz = Jz.reshape(B, N, ht*wd, -1)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    Ei = (wJiT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)
    Ej = (wJjT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)

    w = w.view(B, N, ht*wd, -1)
    r = r.view(B, N, ht*wd, -1)
    wk = torch.sum(w*r*Jz, dim=-1)
    Ck = torch.sum(w*Jz*Jz, dim=-1)

    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    # only optimize keyframe poses
    P = P // rig - fixedp
    ii = ii // rig - fixedp
    jj = jj // rig - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    E = safe_scatter_add_mat(Ei, ii, kk, P, M) + \
        safe_scatter_add_mat(Ej, jj, kk, P, M)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)

    solve_dim = D
    if use_full_imu and imu_delta is not None:
        solve_dim = D + (9 if imu_optimize_bias else 3)
        H_pose = H
        E_pose = E
        v_pose = v
        H = H_pose.new_zeros((B, P * P, solve_dim, solve_dim))
        E = E_pose.new_zeros((B, P * M, solve_dim, ht * wd))
        v = v_pose.new_zeros((B, P, solve_dim))
        H[:, :, :D, :D] = H_pose
        E[:, :, :D, :] = E_pose
        v[:, :, :D] = v_pose

        H_imu, v_imu = _build_full_imu_system(
            poses,
            imu_motion,
            imu_delta,
            imu_valid=imu_valid,
            imu_info=imu_info,
            imu_info_reference=imu_info_reference,
            imu_bias_jacobian=imu_bias_jacobian,
            gyro_bias=gyro_bias,
            accel_bias=accel_bias,
            imu_camera_translation=imu_camera_translation,
            fixedp=fixedp,
            rig=rig,
            num_opt=P,
            state_dim=solve_dim,
            pose_dim=D,
            factor_weight=imu_factor_weight,
            imu_confidence=imu_confidence,
            imu_adaptive_gate=imu_adaptive_gate,
            max_residual=imu_max_residual,
            pos_weight=imu_full_pos_weight,
            vel_weight=imu_full_vel_weight,
            rot_weight=1.0,
            bias_weight=imu_full_bias_weight,
            gravity=imu_gravity,
            imu_compose_order=imu_full_compose_order,
            imu_inverse=imu_full_inverse,
            use_imu_info_weighting=use_imu_info_weighting,
            imu_info_weight_clip=imu_info_weight_clip,
            imu_info_weight_eps=imu_info_weight_eps,
            imu_optimize_bias=imu_optimize_bias,
        )
        if H_imu is not None and v_imu is not None:
            H = H + H_imu
            v = v + v_imu

        H_prior, v_prior = _build_motion_prior_system(
            imu_motion,
            imu_motion_prior=imu_motion_prior,
            fixedp=fixedp,
            rig=rig,
            num_opt=P,
            state_dim=solve_dim,
            velocity_weight=imu_motion_prior_weight,
            local_bias_weight=imu_local_bias_prior_weight if imu_optimize_bias else 0.0,
        )
        if H_prior is not None and v_prior is not None:
            H = H + H_prior
            v = v + v_prior

    else:
        H_imu, v_imu = _build_rotation_imu_system(
            poses,
            imu_delta,
            imu_valid=imu_valid,
            gyro_bias=gyro_bias,
            fixedp=fixedp,
            rig=rig,
            num_opt=P,
            dim=D,
            factor_weight=imu_factor_weight,
            imu_confidence=imu_confidence,
            imu_adaptive_gate=imu_adaptive_gate,
            max_residual=imu_max_residual,
        )
        if H_imu is not None and v_imu is not None:
            H = H + H_imu
            v = v + v_imu

    C = safe_scatter_add_vec(Ck, kk, M)
    w = safe_scatter_add_vec(wk, kk, M)

    C = C + eta.view(*C.shape) + 1e-7

    H = H.view(B, P, P, solve_dim, solve_dim)
    E = E.view(B, P, M, solve_dim, ht*wd)

    ### 3: solve the system ###
    dx, dz = schur_solve(H, E, C, v, w)
    
    ### 4: apply retraction ###
    opt_indices = torch.arange(P, device=dx.device) + fixedp
    if use_full_imu and imu_motion is not None:
        poses = pose_retr(poses, dx[..., :D], opt_indices)
        imu_motion = motion_retr(imu_motion, dx[..., D:], opt_indices)
    else:
        poses = pose_retr(poses, dx, opt_indices)
    disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)

    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)

    if use_full_imu and imu_motion is not None:
        return poses, disps, imu_motion

    return poses, disps


def MoBA(target, weight, eta, poses, disps, intrinsics, ii, jj, fixedp=1, rig=1):
    """ Motion only bundle adjustment """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1)
    w = .001 * (valid * weight).view(B, N, -1, 1)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2,3)
    wJjT = (w * Jj).transpose(2,3)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    # only optimize keyframe poses
    P = P // rig - fixedp
    ii = ii // rig - fixedp
    jj = jj // rig - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)
    
    H = H.view(B, P, P, D, D)

    ### 3: solve the system ###
    dx = block_solve(H, v)

    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    return poses

import lietorch
import torch
import torch.nn.functional as F

from .chol import block_solve, schur_solve
from .imu_factor import (
    quat_inverse,
    quat_multiply,
    quat_rotate,
    quat_to_rotvec,
    rotvec_to_quat,
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
    return motion + scatter_sum(dx, ii, dim=1, dim_size=motion.shape[1])


def _expand_vector_param(param, src, batch, num_frames, device, dtype, name):
    if param is None:
        return torch.zeros((batch, src.numel(), 3), device=device, dtype=dtype)

    value = param.to(device=device, dtype=dtype)
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
    if imu_confidence is None:
        return torch.ones((batch, dst.numel()), device=device, dtype=dtype)

    confidence = imu_confidence.to(device=device, dtype=dtype)
    if confidence.ndim == 1:
        return confidence[dst][None].expand(batch, -1)

    if confidence.ndim == 2:
        if confidence.shape[0] == 1 and batch > 1:
            confidence = confidence.expand(batch, -1)
        if confidence.shape[0] != batch or confidence.shape[1] < num_frames:
            raise ValueError(
                "imu_confidence must have shape [N] or [B, N] and match BA poses, "
                f"got {tuple(imu_confidence.shape)} for frames={num_frames}"
            )
        return confidence[:, dst]

    raise ValueError(
        f"imu_confidence must be [N] or [B, N], got {tuple(imu_confidence.shape)}"
    )


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

    if imu_confidence is None:
        confidence = pose_data.new_ones((B, dst.numel()))
    else:
        confidence = imu_confidence.to(device=pose_data.device, dtype=pose_data.dtype)
        if confidence.ndim == 1:
            confidence = confidence[dst][None].expand(B, -1)
        elif confidence.ndim == 2:
            if confidence.shape[0] == 1 and B > 1:
                confidence = confidence.expand(B, -1)
            if confidence.shape[0] != B or confidence.shape[1] < num_frames:
                raise ValueError(
                    "imu_confidence must have shape [N] or [B, N] and match BA poses, "
                    f"got {tuple(imu_confidence.shape)} for poses={tuple(pose_data.shape)}"
                )
            confidence = confidence[:, dst]
        else:
            raise ValueError(
                f"imu_confidence must be [N] or [B, N], got {tuple(imu_confidence.shape)}"
            )

    edge_weight = global_weight * confidence * edge_mask.to(dtype=pose_data.dtype)

    Ji = pose_data.new_zeros((B, dst.numel(), 3, dim))
    Jj = pose_data.new_zeros((B, dst.numel(), 3, dim))
    eye = torch.eye(3, device=pose_data.device, dtype=pose_data.dtype)

    # BA solves residuals in the form r - J dx. For
    # r = Log((q_i * dq)^-1 * q_j), the small-angle solver Jacobians are:
    #   J_i = +I, J_j = -I
    Ji[:, :, :, 3:6] = eye.view(1, 1, 3, 3)
    Jj[:, :, :, 3:6] = -eye.view(1, 1, 3, 3)

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
    gyro_bias=None,
    accel_bias=None,
    fixedp=1,
    rig=1,
    num_opt=0,
    state_dim=15,
    pose_dim=6,
    factor_weight=0.0,
    imu_confidence=None,
    max_residual=0.5,
    pos_weight=0.05,
    vel_weight=0.05,
    rot_weight=1.0,
    bias_weight=0.001,
):
    """Build approximate full IMU normal-equation blocks.

    State layout is `[pose(6), velocity(3), accel_bias(3), gyro_bias(3)]`.
    The residual layout is `[r_p, r_v, r_R, r_ba, r_bg]`. This is deliberately
    PyTorch-only so training can learn an IMU confidence head before the CUDA
    backend is extended.
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

    motion = imu_motion.to(device=pose_data.device, dtype=pose_data.dtype)
    if motion.ndim == 2:
        motion = motion.unsqueeze(0)
    if motion.shape[0] == 1 and B > 1:
        motion = motion.expand(B, -1, -1)
    if motion.shape[0] != B or motion.shape[1] < num_frames or motion.shape[-1] != 9:
        raise ValueError(
            "imu_motion must have shape [B, N, 9] with [v, ba, bg], "
            f"got {tuple(imu_motion.shape)} for poses={tuple(pose_data.shape)}"
        )

    src = torch.arange(0, num_frames - 1, device=pose_data.device, dtype=torch.long)
    dst = src + 1
    edge_count = dst.numel()

    dt = imu[:, dst, 0:1].clamp_min(0.0)
    p_i = pose_data[:, src, 0:3]
    p_j = pose_data[:, dst, 0:3]
    q_i = pose_data[:, src, 3:7]
    q_j = pose_data[:, dst, 3:7]

    v_i = motion[:, src, 0:3]
    v_j = motion[:, dst, 0:3]
    ba_i = motion[:, src, 3:6]
    ba_j = motion[:, dst, 3:6]
    bg_i = motion[:, src, 6:9]
    bg_j = motion[:, dst, 6:9]

    ba_global = _expand_vector_param(
        accel_bias, src, B, num_frames, pose_data.device, pose_data.dtype, "accel_bias"
    )
    bg_global = _expand_vector_param(
        gyro_bias, src, B, num_frames, pose_data.device, pose_data.dtype, "gyro_bias"
    )

    ba_total = ba_i + ba_global
    bg_total = bg_i + bg_global

    rotvec = imu[:, dst, 1:4] - bg_total * dt
    dv_imu = imu[:, dst, 4:7] - ba_total * dt
    dp_imu = imu[:, dst, 7:10] - 0.5 * ba_total * dt * dt

    q_imu = rotvec_to_quat(rotvec)
    q_pred = quat_multiply(q_i, q_imu)
    r_R = quat_to_rotvec(quat_multiply(quat_inverse(q_pred), q_j))

    q_i_inv = quat_inverse(q_i)
    r_p = quat_rotate(q_i_inv, p_j - p_i - v_i * dt) - dp_imu
    r_v = quat_rotate(q_i_inv, v_j - v_i) - dv_imu
    r_ba = ba_j - ba_i
    r_bg = bg_j - bg_i

    residual = torch.cat([r_p, r_v, r_R, r_ba, r_bg], dim=-1)

    edge_mask = _select_temporal_mask(imu_valid, dst, B, num_frames, pose_data.device)
    if max_residual is not None and max_residual > 0.0:
        edge_mask = edge_mask & (r_R.norm(dim=-1) <= float(max_residual))

    global_weight = _global_factor_weight(factor_weight, B, pose_data.device, pose_data.dtype)
    confidence = _select_temporal_confidence(
        imu_confidence, dst, B, num_frames, pose_data.device, pose_data.dtype
    )
    edge_weight = global_weight * confidence * edge_mask.to(dtype=pose_data.dtype)

    Ji = pose_data.new_zeros((B, edge_count, 15, state_dim))
    Jj = pose_data.new_zeros((B, edge_count, 15, state_dim))
    eye = torch.eye(3, device=pose_data.device, dtype=pose_data.dtype)
    I = eye.view(1, 1, 3, 3)

    # BA uses residuals as r - J dx. These are small-angle Jacobians for the
    # full preintegration factor; the stronger exact SO(3) terms are deferred to
    # the CUDA/Forster-style implementation stage.
    Ji[:, :, 0:3, 0:3] = I
    Jj[:, :, 0:3, 0:3] = -I
    Ji[:, :, 0:3, 6:9] = dt[:, :, None] * I
    Ji[:, :, 0:3, 9:12] = -0.5 * dt[:, :, None] * dt[:, :, None] * I

    Ji[:, :, 3:6, 6:9] = I
    Jj[:, :, 3:6, 6:9] = -I
    Ji[:, :, 3:6, 9:12] = -dt[:, :, None] * I

    Ji[:, :, 6:9, 3:6] = I
    Jj[:, :, 6:9, 3:6] = -I
    Ji[:, :, 6:9, 12:15] = -dt[:, :, None] * I

    Ji[:, :, 9:12, 9:12] = I
    Jj[:, :, 9:12, 9:12] = -I

    Ji[:, :, 12:15, 12:15] = I
    Jj[:, :, 12:15, 12:15] = -I

    scale = pose_data.new_tensor(
        [pos_weight] * 3 + [vel_weight] * 3 + [rot_weight] * 3 +
        [bias_weight] * 3 + [bias_weight] * 3
    ).clamp_min(0.0).sqrt()
    residual = residual * scale.view(1, 1, 15)
    Ji = Ji * scale.view(1, 1, 15, 1)
    Jj = Jj * scale.view(1, 1, 15, 1)

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
    if local_bias_weight > 0.0:
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
    gyro_bias=None,
    imu_factor_weight=0.0,
    imu_confidence=None,
    imu_max_residual=0.5,
    imu_motion=None,
    imu_motion_prior=None,
    accel_bias=None,
    use_full_imu=False,
    imu_full_pos_weight=0.05,
    imu_full_vel_weight=0.05,
    imu_full_bias_weight=0.001,
    imu_motion_prior_weight=0.0,
    imu_local_bias_prior_weight=0.0,
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
        solve_dim = D + 9
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
            gyro_bias=gyro_bias,
            accel_bias=accel_bias,
            fixedp=fixedp,
            rig=rig,
            num_opt=P,
            state_dim=solve_dim,
            pose_dim=D,
            factor_weight=imu_factor_weight,
            imu_confidence=imu_confidence,
            max_residual=imu_max_residual,
            pos_weight=imu_full_pos_weight,
            vel_weight=imu_full_vel_weight,
            rot_weight=1.0,
            bias_weight=imu_full_bias_weight,
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
            local_bias_weight=imu_local_bias_prior_weight,
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

from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from lietorch import SO3, SE3, Sim3
from .graph_utils import graph_to_edge_list
from .projective_ops import projective_transform
from .imu_factor import (
    quat_inverse,
    quat_multiply,
    quat_rotate,
    quat_to_rotvec,
    rotvec_to_quat,
    rotation_only_residual,
)


def pose_metrics(dE):
    """ Translation/Rotation/Scaling metrics from Sim3 """
    t, q, s = dE.data.split([3, 4, 1], -1)
    ang = SO3(q).log().norm(dim=-1)

    # convert radians to degrees
    r_err = (180 / np.pi) * ang
    t_err = t.norm(dim=-1)
    s_err = (s - 1.0).abs()
    return r_err, t_err, s_err


def fit_scale(Ps, Gs):
    b = Ps.shape[0]
    t1 = Ps.data[...,:3].detach().reshape(b, -1)
    t2 = Gs.data[...,:3].detach().reshape(b, -1)

    s = (t1*t2).sum(-1) / ((t2*t2).sum(-1) + 1e-8)
    return s


def geodesic_loss(Ps, Gs, graph, gamma=0.9, do_scale=True):
    """ Loss function for training network """

    # relative pose
    ii, jj, kk = graph_to_edge_list(graph)
    dP = Ps[:,jj] * Ps[:,ii].inv()

    n = len(Gs)
    geodesic_loss = 0.0

    for i in range(n):
        w = gamma ** (n - i - 1)
        dG = Gs[i][:,jj] * Gs[i][:,ii].inv()

        if do_scale:
            s = fit_scale(dP, dG)
            dG = dG.scale(s[:,None])
        
        # pose error
        d = (dG * dP.inv()).log()

        if isinstance(dG, SE3):
            tau, phi = d.split([3,3], dim=-1)
            geodesic_loss += w * (
                tau.norm(dim=-1).mean() + 
                phi.norm(dim=-1).mean())

        elif isinstance(dG, Sim3):
            tau, phi, sig = d.split([3,3,1], dim=-1)
            geodesic_loss += w * (
                tau.norm(dim=-1).mean() + 
                phi.norm(dim=-1).mean() + 
                0.05 * sig.norm(dim=-1).mean())
            
        dE = Sim3(dG * dP.inv()).detach()
        r_err, t_err, s_err = pose_metrics(dE)

    metrics = {
        'rot_error': r_err.mean().item(),
        'tr_error': t_err.mean().item(),
        'bad_rot': (r_err < .1).float().mean().item(),
        'bad_tr': (t_err < .01).float().mean().item(),
    }

    return geodesic_loss, metrics


def residual_loss(residuals, gamma=0.9):
    """ loss on system residuals """
    residual_loss = 0.0
    n = len(residuals)

    for i in range(n):
        w = gamma ** (n - i - 1)
        residual_loss += w * residuals[i].abs().mean()

    return residual_loss, {'residual': residual_loss.item()}


def flow_loss(Ps, disps, poses_est, disps_est, intrinsics, graph, gamma=0.9):
    """ optical flow loss """

    N = Ps.shape[1]
    graph = OrderedDict()
    for i in range(N):
        graph[i] = [j for j in range(N) if abs(i-j)==1]

    ii, jj, kk = graph_to_edge_list(graph)
    coords0, val0 = projective_transform(Ps, disps, intrinsics, ii, jj)
    val0 = val0 * (disps[:,ii] > 0).float().unsqueeze(dim=-1)

    n = len(poses_est)
    flow_loss = 0.0

    for i in range(n):
        w = gamma ** (n - i - 1)
        coords1, val1 = projective_transform(poses_est[i], disps_est[i], intrinsics, ii, jj)

        v = (val0 * val1).squeeze(dim=-1)
        epe = v * (coords1 - coords0).norm(dim=-1)
        flow_loss += w * epe.mean()

    epe = epe.reshape(-1)[v.reshape(-1) > 0.5]
    metrics = {
        'f_error': epe.mean().item(),
        '1px': (epe<1.0).float().mean().item(),
    }

    return flow_loss, metrics


def _as_batched_imu_tensor(name, tensor, min_last_dim):
    if tensor.ndim == 2 and tensor.shape[-1] >= min_last_dim:
        return tensor.unsqueeze(0)

    if tensor.ndim == 3 and tensor.shape[-1] >= min_last_dim:
        return tensor

    raise ValueError(
        f"{name} must have shape [N, >={min_last_dim}] or [B, N, >={min_last_dim}], "
        f"got {tuple(tensor.shape)}"
    )


def _pose_data(poses):
    return poses if torch.is_tensor(poses) else poses.data


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
                "imu_valid must have shape [N] or [B, N] and match poses, "
                f"got {tuple(imu_valid.shape)} for frames={num_frames}"
            )
        return valid[:, dst]

    raise ValueError(f"imu_valid must be [N] or [B, N], got {tuple(imu_valid.shape)}")


def _pose_velocity_motion(poses, imu_delta):
    pose_data = _pose_data(poses)
    imu = _as_batched_imu_tensor("imu_delta", imu_delta, 10).to(
        device=pose_data.device,
        dtype=pose_data.dtype,
    )
    if imu.shape[0] == 1 and pose_data.shape[0] > 1:
        imu = imu.expand(pose_data.shape[0], -1, -1)

    batch, num_frames = pose_data.shape[:2]
    motion = pose_data.new_zeros((batch, num_frames, 9))
    if num_frames < 2 or imu.shape[1] < num_frames:
        return motion

    dt = imu[:, 1:num_frames, 0:1].clamp_min(1e-4)
    velocity = (pose_data[:, 1:num_frames, 0:3] - pose_data[:, :num_frames - 1, 0:3]) / dt
    motion[:, :num_frames - 1, 0:3] = velocity
    motion[:, num_frames - 1, 0:3] = velocity[:, -1]
    return motion


def full_preintegration_residual(
    poses,
    imu_delta,
    imu_valid=None,
    imu_motion=None,
    gyro_bias=None,
    accel_bias=None,
    gravity=None,
    max_residual=0.5,
):
    """
    Compute full preintegration residuals for consecutive frame pairs.

    The residual layout is compatible with the PyTorch full-IMU BA prototype:
    `[r_p, r_v, r_R, r_ba, r_bg]`, each term with 3 channels.
    """

    pose_data = _pose_data(poses)
    if imu_delta is None or pose_data.shape[1] < 2:
        empty = pose_data.new_zeros((0, 3))
        return empty, empty, empty, empty, empty

    imu = _as_batched_imu_tensor("imu_delta", imu_delta, 10).to(
        device=pose_data.device,
        dtype=pose_data.dtype,
    )
    if imu.shape[0] == 1 and pose_data.shape[0] > 1:
        imu = imu.expand(pose_data.shape[0], -1, -1)

    batch, num_frames = pose_data.shape[:2]
    if imu.shape[0] != batch or imu.shape[1] < num_frames:
        raise ValueError(
            "imu_delta must match poses batch/frame dimensions, "
            f"got poses={tuple(pose_data.shape)}, imu_delta={tuple(imu_delta.shape)}"
        )

    if imu_motion is None:
        motion = _pose_velocity_motion(pose_data, imu)
    else:
        motion = imu_motion.to(device=pose_data.device, dtype=pose_data.dtype)
        if motion.ndim == 2:
            motion = motion.unsqueeze(0)
        if motion.shape[0] == 1 and batch > 1:
            motion = motion.expand(batch, -1, -1)
        if motion.shape[0] != batch or motion.shape[1] < num_frames or motion.shape[-1] != 9:
            raise ValueError(
                "imu_motion must have shape [B, N, 9] with [v, ba, bg], "
                f"got {tuple(imu_motion.shape)} for poses={tuple(pose_data.shape)}"
            )

    src = torch.arange(0, num_frames - 1, device=pose_data.device, dtype=torch.long)
    dst = src + 1

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
        accel_bias, src, batch, num_frames, pose_data.device, pose_data.dtype, "accel_bias"
    )
    bg_global = _expand_vector_param(
        gyro_bias, src, batch, num_frames, pose_data.device, pose_data.dtype, "gyro_bias"
    )
    g_world = _expand_vector_param(
        gravity, src, batch, num_frames, pose_data.device, pose_data.dtype, "gravity"
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
    r_p = quat_rotate(q_i_inv, p_j - p_i - v_i * dt - 0.5 * g_world * dt * dt) - dp_imu
    r_v = quat_rotate(q_i_inv, v_j - v_i - g_world * dt) - dv_imu
    r_ba = ba_j - ba_i
    r_bg = bg_j - bg_i

    mask = _select_temporal_mask(imu_valid, dst, batch, num_frames, pose_data.device)
    if max_residual is not None and max_residual > 0.0:
        mask = mask & (r_R.norm(dim=-1) <= float(max_residual))

    return r_p[mask], r_v[mask], r_R[mask], r_ba[mask], r_bg[mask]


def imu_full_preintegration_loss(
    poses_est,
    imu_delta,
    imu_valid=None,
    imu_motions=None,
    gyro_bias=None,
    accel_bias=None,
    gamma=0.9,
    max_residual=0.5,
    smooth_beta=0.05,
    pos_weight=0.05,
    vel_weight=0.05,
    rot_weight=1.0,
    bias_weight=0.001,
    gravity=None,
):
    """
    Full IMU preintegration loss over position, velocity, rotation, and bias.

    This is still a training-side PyTorch approximation, but unlike the old
    rotation-only loss it gives the accelerometer path a direct learning signal.
    """

    if imu_delta is None or len(poses_est) == 0:
        device = poses_est[-1].data.device if len(poses_est) else "cuda"
        zero = torch.zeros((), device=device)
        return zero, {
            "imu_full_loss": 0.0,
            "imu_pos_loss": 0.0,
            "imu_vel_loss": 0.0,
            "imu_rot_loss": 0.0,
            "imu_bias_loss": 0.0,
            "imu_pos_error": 0.0,
            "imu_vel_error": 0.0,
            "imu_rot_error": 0.0,
            "imu_bias_error": 0.0,
            "imu_edges": 0.0,
        }

    imu_loss = 0.0
    pos_loss_total = 0.0
    vel_loss_total = 0.0
    rot_loss_total = 0.0
    bias_loss_total = 0.0
    total_edges = 0

    last_pos_error = torch.zeros((), device=imu_delta.device)
    last_vel_error = torch.zeros((), device=imu_delta.device)
    last_rot_error = torch.zeros((), device=imu_delta.device)
    last_bias_error = torch.zeros((), device=imu_delta.device)
    n = len(poses_est)

    for i in range(n):
        w = gamma ** (n - i - 1)
        motion_i = None
        if imu_motions is not None and i < len(imu_motions):
            motion_i = imu_motions[i]

        r_p, r_v, r_R, r_ba, r_bg = full_preintegration_residual(
            poses_est[i],
            imu_delta,
            imu_valid=imu_valid,
            imu_motion=motion_i,
            gyro_bias=gyro_bias,
            accel_bias=accel_bias,
            gravity=gravity,
            max_residual=max_residual,
        )

        if r_R.numel() == 0:
            continue

        zero_p = torch.zeros_like(r_p)
        zero_v = torch.zeros_like(r_v)
        zero_R = torch.zeros_like(r_R)
        zero_ba = torch.zeros_like(r_ba)
        zero_bg = torch.zeros_like(r_bg)

        pos_loss = F.smooth_l1_loss(r_p, zero_p, beta=float(smooth_beta))
        vel_loss = F.smooth_l1_loss(r_v, zero_v, beta=float(smooth_beta))
        rot_loss = F.smooth_l1_loss(r_R, zero_R, beta=float(smooth_beta))
        ba_loss = F.smooth_l1_loss(r_ba, zero_ba, beta=float(smooth_beta))
        bg_loss = F.smooth_l1_loss(r_bg, zero_bg, beta=float(smooth_beta))
        bias_loss = ba_loss + bg_loss

        imu_loss = imu_loss + w * (
            float(pos_weight) * pos_loss
            + float(vel_weight) * vel_loss
            + float(rot_weight) * rot_loss
            + float(bias_weight) * bias_loss
        )

        pos_loss_total = pos_loss_total + w * pos_loss
        vel_loss_total = vel_loss_total + w * vel_loss
        rot_loss_total = rot_loss_total + w * rot_loss
        bias_loss_total = bias_loss_total + w * bias_loss

        total_edges += r_R.shape[0]
        last_pos_error = r_p.norm(dim=-1).mean()
        last_vel_error = r_v.norm(dim=-1).mean()
        last_rot_error = r_R.norm(dim=-1).mean()
        last_bias_error = torch.cat([r_ba, r_bg], dim=-1).norm(dim=-1).mean()

    if not torch.is_tensor(imu_loss):
        zero_anchor = poses_est[-1].data.sum() * 0.0
        if gyro_bias is not None:
            zero_anchor = zero_anchor + gyro_bias.sum() * 0.0
        if accel_bias is not None:
            zero_anchor = zero_anchor + accel_bias.sum() * 0.0
        if imu_motions:
            zero_anchor = zero_anchor + imu_motions[-1].sum() * 0.0
        imu_loss = zero_anchor
        pos_loss_total = zero_anchor
        vel_loss_total = zero_anchor
        rot_loss_total = zero_anchor
        bias_loss_total = zero_anchor

    metrics = {
        "imu_full_loss": float(imu_loss.detach().item()),
        "imu_pos_loss": float(pos_loss_total.detach().item()),
        "imu_vel_loss": float(vel_loss_total.detach().item()),
        "imu_rot_loss": float(rot_loss_total.detach().item()),
        "imu_bias_loss": float(bias_loss_total.detach().item()),
        "imu_pos_error": float(last_pos_error.detach().item()),
        "imu_vel_error": float(last_vel_error.detach().item()),
        "imu_rot_error": float((last_rot_error.detach() * 180.0 / np.pi).item()),
        "imu_bias_error": float(last_bias_error.detach().item()),
        "imu_edges": float(total_edges),
    }

    return imu_loss, metrics


def imu_rotation_bias_loss(
    poses_est,
    imu_delta,
    imu_valid=None,
    gyro_bias=None,
    gamma=0.9,
    max_residual=0.5,
    smooth_beta=0.05,
):
    """
    Rotation-only IMU loss with a learnable gyro bias.

    This is the executable first training stage for inertial coupling. It does
    not introduce velocity, accelerometer bias, gravity, or translation terms.
    The residual is computed from preintegrated gyro rotation:

        r_R = Log((q_i * Exp(dr_j - bg * dt_j))^-1 * q_j)

    where `bg` is the gyro bias in camera-frame rad/s.
    """

    if imu_delta is None or len(poses_est) == 0:
        device = poses_est[-1].data.device if len(poses_est) else "cuda"
        zero = torch.zeros((), device=device)
        return zero, {"imu_rot_loss": 0.0, "imu_rot_error": 0.0, "imu_edges": 0.0}

    imu_loss = 0.0
    total_edges = 0
    last_error = torch.zeros((), device=imu_delta.device)
    n = len(poses_est)

    for i in range(n):
        w = gamma ** (n - i - 1)
        pose_data = poses_est[i].data if hasattr(poses_est[i], "data") else poses_est[i]

        residual = rotation_only_residual(
            pose_data,
            imu_delta,
            imu_valid=imu_valid,
            gyro_bias=gyro_bias,
        )

        if residual.numel() == 0:
            continue

        if max_residual is not None and max_residual > 0.0:
            keep = residual.norm(dim=-1) <= float(max_residual)
            residual = residual[keep]

        if residual.numel() == 0:
            continue

        total_edges += residual.shape[0]
        last_error = residual.norm(dim=-1).mean()
        imu_loss = imu_loss + w * F.smooth_l1_loss(
            residual,
            torch.zeros_like(residual),
            beta=float(smooth_beta),
        )

    if not torch.is_tensor(imu_loss):
        zero_anchor = poses_est[-1].data.sum() * 0.0
        if gyro_bias is not None:
            zero_anchor = zero_anchor + gyro_bias.sum() * 0.0
        imu_loss = zero_anchor

    metrics = {
        "imu_rot_loss": float(imu_loss.detach().item()),
        "imu_rot_error": float((last_error.detach() * 180.0 / np.pi).item()),
        "imu_edges": float(total_edges),
    }

    return imu_loss, metrics

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
    camera_center_from_pose_data,
    imu_center_from_pose_data,
    predict_imu_rotation,
    quat_rotate,
    quat_to_rotvec,
    rotvec_to_quat,
    rotation_only_residual,
    select_imu_bias_jacobian,
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
    """Return raw-camera-reference sqrt information scales for IMU residuals.

    If a sample carries its source camera-rate reference, preserve the
    uncertainty accumulated by composed intervals. Legacy callers without that
    metadata retain the previous local-median behavior.
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
                raise ValueError("imu_info_reference must have shape [B, >=3]")
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


def _pose_velocity_motion(poses, imu_delta, imu_camera_translation=None):
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

    centers = imu_center_from_pose_data(pose_data, imu_camera_translation)
    dt = imu[:, 1:num_frames, 0:1].clamp_min(1e-4)
    velocity = (centers[:, 1:num_frames] - centers[:, :num_frames - 1]) / dt
    motion[:, :num_frames - 1, 0:3] = velocity
    motion[:, num_frames - 1, 0:3] = velocity[:, -1]
    return motion


def full_preintegration_residual(
    poses,
    imu_delta,
    imu_valid=None,
    imu_info=None,
    imu_info_reference=None,
    imu_bias_jacobian=None,
    imu_motion=None,
    gyro_bias=None,
    accel_bias=None,
    imu_camera_translation=None,
    gravity=None,
    imu_compose_order="dq_prev",
    imu_inverse=True,
    max_residual=0.5,
    use_imu_info_weighting=False,
    imu_info_weight_clip=4.0,
    imu_info_weight_eps=1e-12,
    imu_optimize_bias=True,
    return_mask=False,
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

    imu_optimize_bias = bool(imu_optimize_bias)

    if imu_motion is None:
        motion = _pose_velocity_motion(
            pose_data,
            imu,
            imu_camera_translation=imu_camera_translation,
        )
    else:
        motion = imu_motion.to(device=pose_data.device, dtype=pose_data.dtype)
        if motion.ndim == 2:
            motion = motion.unsqueeze(0)
        if motion.shape[0] == 1 and batch > 1:
            motion = motion.expand(batch, -1, -1)
        required_motion_dim = 9 if imu_optimize_bias else 3
        if motion.shape[0] != batch or motion.shape[1] < num_frames or motion.shape[-1] < required_motion_dim:
            raise ValueError(
                "imu_motion must contain [v] or [v, ba, bg] depending on IMU mode, "
                f"got {tuple(imu_motion.shape)} for poses={tuple(pose_data.shape)}"
            )

    src = torch.arange(0, num_frames - 1, device=pose_data.device, dtype=torch.long)
    dst = src + 1

    dt = imu[:, dst, 0:1].clamp_min(0.0)
    q_i = pose_data[:, src, 3:7]
    q_j = pose_data[:, dst, 3:7]
    centers = imu_center_from_pose_data(pose_data, imu_camera_translation)
    p_i = centers[:, src]
    p_j = centers[:, dst]

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
        accel_bias, src, batch, num_frames, pose_data.device, pose_data.dtype, "accel_bias"
    )
    bg_global = _expand_vector_param(
        gyro_bias, src, batch, num_frames, pose_data.device, pose_data.dtype, "gyro_bias"
    )
    g_world = _expand_vector_param(
        gravity, src, batch, num_frames, pose_data.device, pose_data.dtype, "gravity"
    )

    ba_total = ba_global + (ba_i if imu_optimize_bias else 0.0)
    bg_total = bg_global + (bg_i if imu_optimize_bias else 0.0)

    bias_jac = select_imu_bias_jacobian(
        imu_bias_jacobian,
        dst,
        batch,
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

    r_p = quat_rotate(q_i, p_j - p_i - v_i * dt - 0.5 * g_world * dt * dt) - dp_imu
    r_v = quat_rotate(q_i, v_j - v_i - g_world * dt) - dv_imu
    r_ba = ba_j - ba_i if imu_optimize_bias else torch.zeros_like(r_p)
    r_bg = bg_j - bg_i if imu_optimize_bias else torch.zeros_like(r_p)

    mask = _select_temporal_mask(imu_valid, dst, batch, num_frames, pose_data.device)
    if max_residual is not None and max_residual > 0.0:
        mask = mask & (r_R.norm(dim=-1) <= float(max_residual))

    if use_imu_info_weighting:
        info_scale = _select_temporal_info_scale(
            imu_info,
            imu_info_reference,
            dst,
            batch,
            num_frames,
            pose_data.device,
            pose_data.dtype,
            clip=imu_info_weight_clip,
            eps=imu_info_weight_eps,
        )
        r_R = r_R * info_scale[:, :, 0:1]
        r_v = r_v * info_scale[:, :, 1:2]
        r_p = r_p * info_scale[:, :, 2:3]

    if return_mask:
        return r_p, r_v, r_R, r_ba, r_bg, mask
    return r_p[mask], r_v[mask], r_R[mask], r_ba[mask], r_bg[mask]


def imu_full_preintegration_loss(
    poses_est,
    imu_delta,
    imu_valid=None,
    imu_info=None,
    imu_info_reference=None,
    imu_bias_jacobian=None,
    imu_motions=None,
    gyro_bias=None,
    accel_bias=None,
    imu_camera_translation=None,
    gamma=0.9,
    max_residual=0.5,
    smooth_beta=0.05,
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
            "imu_info_weighted": float(bool(use_imu_info_weighting)),
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
            imu_info=imu_info,
            imu_info_reference=imu_info_reference,
            imu_bias_jacobian=imu_bias_jacobian,
            imu_motion=motion_i,
            gyro_bias=gyro_bias,
            accel_bias=accel_bias,
            imu_camera_translation=imu_camera_translation,
            gravity=gravity,
            imu_compose_order=imu_compose_order,
            imu_inverse=imu_inverse,
            max_residual=max_residual,
            use_imu_info_weighting=use_imu_info_weighting,
            imu_info_weight_clip=imu_info_weight_clip,
            imu_info_weight_eps=imu_info_weight_eps,
            imu_optimize_bias=imu_optimize_bias,
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

        def scalar_weight(value):
            if torch.is_tensor(value):
                return value.to(device=pos_loss.device, dtype=pos_loss.dtype).reshape(())
            return pos_loss.new_tensor(float(value))

        imu_loss = imu_loss + w * (
            scalar_weight(pos_weight) * pos_loss
            + scalar_weight(vel_weight) * vel_loss
            + scalar_weight(rot_weight) * rot_loss
            + scalar_weight(bias_weight) * bias_loss
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
        "imu_info_weighted": float(bool(use_imu_info_weighting)),
    }

    return imu_loss, metrics


def imu_confidence_nll_loss(
    poses_est,
    imu_confidences,
    imu_delta,
    imu_valid=None,
    imu_info=None,
    imu_info_reference=None,
    imu_bias_jacobian=None,
    imu_motions=None,
    gyro_bias=None,
    accel_bias=None,
    gravity=None,
    imu_camera_translation=None,
    gamma=0.9,
    max_residual=0.5,
    pos_weight=0.05,
    vel_weight=0.05,
    rot_weight=1.0,
    imu_compose_order="dq_prev",
    imu_inverse=True,
    use_imu_info_weighting=False,
    imu_info_weight_clip=4.0,
    imu_info_weight_eps=1e-12,
    imu_optimize_bias=False,
):
    """Calibrate temporal GRU IMU confidence with a precision-style NLL.

    ``c_ij`` is a learned factor precision surrogate in ``(0, 1)``.  The
    confidence head should reduce a factor's influence when its *own*
    preintegration residual is unexpectedly large, rather than merely staying
    near one for every graph edge.  For each temporal IMU factor we use the
    normalized energy ``E_ij`` and minimize ``c_ij E_ij - log(c_ij)``.

    The residual energy is detached deliberately: normal BA and the IMU loss
    train the pose/motion estimate, while this auxiliary term calibrates the
    confidence head instead of rewarding it for changing the residual itself.
    Per-step median normalization makes position, velocity, and rotation units
    comparable across source datasets and sensor rates.
    """

    if (
        imu_delta is None
        or not poses_est
        or not imu_confidences
    ):
        if poses_est:
            zero = poses_est[-1].data.sum() * 0.0
        elif torch.is_tensor(imu_delta):
            zero = imu_delta.sum() * 0.0
        else:
            zero = torch.zeros(())
        return zero, {
            "imu_confidence_nll": 0.0,
            "imu_confidence_nll_edges": 0.0,
            "imu_confidence_nll_mean": 0.0,
        }

    total = None
    total_weight = 0.0
    total_edges = 0
    last_confidence_mean = None
    steps = min(len(poses_est), len(imu_confidences))

    def scalar_weight(value, anchor):
        if torch.is_tensor(value):
            return value.to(device=anchor.device, dtype=anchor.dtype).reshape(())
        return anchor.new_tensor(float(value))

    for step_index in range(steps):
        pose_data = _pose_data(poses_est[step_index])
        batch, num_frames = pose_data.shape[:2]
        if num_frames < 2:
            continue

        confidence = imu_confidences[step_index]
        if confidence is None:
            continue
        confidence = confidence.to(device=pose_data.device, dtype=pose_data.dtype)
        if confidence.ndim == 1:
            confidence = confidence.unsqueeze(0).unsqueeze(-1)
        elif confidence.ndim == 2:
            confidence = confidence.unsqueeze(-1)
        if confidence.ndim != 3:
            raise ValueError(
                "imu confidence must be [N], [B, N], or [B, N, 1|3] after temporal aggregation, "
                f"got {tuple(confidence.shape)}"
            )
        if confidence.shape[0] == 1 and batch > 1:
            confidence = confidence.expand(batch, -1, -1)
        if (
            confidence.shape[0] != batch
            or confidence.shape[1] < num_frames
            or confidence.shape[-1] not in (1, 3)
        ):
            raise ValueError(
                "imu confidence does not cover the full pose sequence: "
                f"confidence={tuple(confidence.shape)}, poses={tuple(pose_data.shape)}"
            )

        motion = None
        if imu_motions is not None and step_index < len(imu_motions):
            motion = imu_motions[step_index]
        r_p, r_v, r_R, _, _, mask = full_preintegration_residual(
            poses_est[step_index],
            imu_delta,
            imu_valid=imu_valid,
            imu_info=imu_info,
            imu_info_reference=imu_info_reference,
            imu_bias_jacobian=imu_bias_jacobian,
            imu_motion=motion,
            gyro_bias=gyro_bias,
            accel_bias=accel_bias,
            gravity=gravity,
            imu_camera_translation=imu_camera_translation,
            imu_compose_order=imu_compose_order,
            imu_inverse=imu_inverse,
            max_residual=max_residual,
            use_imu_info_weighting=use_imu_info_weighting,
            imu_info_weight_clip=imu_info_weight_clip,
            imu_info_weight_eps=imu_info_weight_eps,
            imu_optimize_bias=imu_optimize_bias,
            return_mask=True,
        )

        pos_energy = scalar_weight(pos_weight, r_p) * r_p.square().mean(dim=-1)
        vel_energy = scalar_weight(vel_weight, r_v) * r_v.square().mean(dim=-1)
        rot_energy = scalar_weight(rot_weight, r_R) * r_R.square().mean(dim=-1)

        if confidence.shape[-1] == 1:
            # Preserve the original scalar-confidence objective exactly for
            # old checkpoints and scalar-head ablations.
            energy = pos_energy + vel_energy + rot_energy
            valid = mask & torch.isfinite(energy)
            if not bool(valid.any().detach().cpu().item()):
                continue
            normalizer = energy.detach()[valid].median().clamp_min(1e-6)
            normalized_energy = (energy.detach() / normalizer).clamp(max=100.0)
            factor_confidence = confidence[:, 1:num_frames, 0].clamp(1e-4, 1.0 - 1e-4)
            nll = factor_confidence * normalized_energy - torch.log(factor_confidence)
            step_loss = nll[valid].mean()
            confidence_for_metric = factor_confidence[valid]
        else:
            # Component channel order is [rotation, velocity, position].
            # Each category receives an independent normalized energy so a
            # large translation unit cannot dominate gyro confidence learning.
            component_energy = torch.stack(
                [rot_energy, vel_energy, pos_energy], dim=-1
            )
            valid = mask & torch.isfinite(component_energy).all(dim=-1)
            if not bool(valid.any().detach().cpu().item()):
                continue
            masked_energy = torch.where(
                valid[..., None],
                component_energy.detach(),
                torch.full_like(component_energy, float("nan")),
            )
            normalizer = torch.nanmedian(masked_energy, dim=1, keepdim=True).values
            normalizer = torch.where(
                torch.isfinite(normalizer), normalizer, torch.ones_like(normalizer)
            ).clamp_min(1e-6)
            normalized_energy = (component_energy.detach() / normalizer).clamp(max=100.0)
            factor_confidence = confidence[:, 1:num_frames].clamp(1e-4, 1.0 - 1e-4)
            nll = factor_confidence * normalized_energy - torch.log(factor_confidence)
            step_loss = nll[valid].mean()
            confidence_for_metric = factor_confidence[valid]
        step_weight = float(gamma) ** (steps - step_index - 1)
        total = step_loss * step_weight if total is None else total + step_loss * step_weight
        total_weight += step_weight
        total_edges += int(valid.sum().detach().cpu().item())
        last_confidence_mean = confidence_for_metric.mean()

    if total is None:
        zero = poses_est[-1].data.sum() * 0.0
        return zero, {
            "imu_confidence_nll": 0.0,
            "imu_confidence_nll_edges": 0.0,
            "imu_confidence_nll_mean": 0.0,
        }

    total = total / max(total_weight, 1e-12)
    return total, {
        "imu_confidence_nll": float(total.detach().item()),
        "imu_confidence_nll_edges": float(total_edges),
        "imu_confidence_nll_mean": float(
            last_confidence_mean.detach().item()
            if last_confidence_mean is not None
            else 0.0
        ),
    }


def imu_confidence_pose_prior_ranking_loss(
    pose_only_poses,
    pose_only_confidence,
    imu_delta,
    imu_valid=None,
    imu_info=None,
    imu_info_reference=None,
    imu_bias_jacobian=None,
    gyro_bias=None,
    accel_bias=None,
    gravity=None,
    imu_camera_translation=None,
    max_residual=None,
    pos_weight=0.05,
    vel_weight=0.05,
    rot_weight=1.0,
    imu_compose_order="dq_prev",
    imu_inverse=True,
    imu_optimize_bias=True,
    augmented_mask=None,
    absolute_weight=0.0,
    energy_log_reference=None,
    energy_reference_momentum=0.98,
    energy_temperature=0.75,
    absolute_target_mode="median_residual",
    quality_only_confidence=None,
):
    """Supervise confidence from the *visual-only* pose/IMU disagreement.

    This intentionally does not inspect a pose that has already gone through
    full IMU DBA.  ``pose_only_poses`` is the final estimate of the visual
    prefix, so it is an independent reference for each IMU interval.

    The ranking term teaches interval ordering: if visual pose ``i``
    disagrees with its IMU prior more than visual pose ``j``, confidence ``i``
    must be lower.  When ``absolute_weight`` is positive, the same *pre-BA*
    disagreement additionally produces an absolute confidence target.  In
    ``median_residual`` mode that target is fully data-derived from a robust
    running log-energy reference. In ``nominal_clean`` mode, the supplied IMU
    covariance defines a clean, uncorrupted interval as the nominal factor
    (confidence one). ``split_nominal_quality`` keeps this nominal convention
    only for the raw-IMU quality branch; the final visual-GRU+quality factor
    confidence keeps its residual-derived target. In
    ``one_sided_nominal_quality`` mode, normal clean residuals retain full
    confidence and only their high-residual tail is attenuated. Runtime still
    consumes one scalar per factor: these modes alter supervision, not the BA
    weighting topology.

    When synthetic degradation is supplied, every bad-vs-clean pair is also
    ordered as ``c_bad < c_clean``.  This makes raw-noise, drift, and short
    dropout augmentation observable even for a nearly static camera, where a
    pose residual alone can be ambiguous.
    """

    if (
        pose_only_poses is None
        or pose_only_confidence is None
        or imu_delta is None
    ):
        if pose_only_poses is not None:
            zero = _pose_data(pose_only_poses).sum() * 0.0
        elif torch.is_tensor(imu_delta):
            zero = imu_delta.sum() * 0.0
        else:
            zero = torch.zeros(())
        return zero, {
            "imu_confidence_pose_prior": 0.0,
            "imu_confidence_pose_prior_ranking": 0.0,
            "imu_confidence_pose_prior_absolute": 0.0,
            "imu_confidence_pose_prior_quality_only_absolute": 0.0,
            "imu_confidence_pose_prior_target_mean": 0.0,
            "imu_confidence_pose_prior_energy_log_reference": float("nan"),
            "imu_confidence_pose_prior_pairs": 0.0,
            "imu_confidence_pose_prior_aug_pairs": 0.0,
            "imu_confidence_pose_prior_clean_mean": 0.0,
            "imu_confidence_pose_prior_aug_mean": 0.0,
        }

    pose_data = _pose_data(pose_only_poses)
    batch, num_frames = pose_data.shape[:2]
    if num_frames < 3:
        zero = pose_data.sum() * 0.0
        return zero, {
            "imu_confidence_pose_prior": 0.0,
            "imu_confidence_pose_prior_ranking": 0.0,
            "imu_confidence_pose_prior_absolute": 0.0,
            "imu_confidence_pose_prior_quality_only_absolute": 0.0,
            "imu_confidence_pose_prior_target_mean": 0.0,
            "imu_confidence_pose_prior_energy_log_reference": float("nan"),
            "imu_confidence_pose_prior_pairs": 0.0,
            "imu_confidence_pose_prior_aug_pairs": 0.0,
            "imu_confidence_pose_prior_clean_mean": 0.0,
            "imu_confidence_pose_prior_aug_mean": 0.0,
        }

    confidence = pose_only_confidence.to(device=pose_data.device, dtype=pose_data.dtype)
    if confidence.ndim == 1:
        confidence = confidence.unsqueeze(0).unsqueeze(-1)
    elif confidence.ndim == 2:
        confidence = confidence.unsqueeze(-1)
    if confidence.ndim != 3 or confidence.shape[-1] != 1:
        raise ValueError(
            "pose-only scalar IMU confidence must be [B, N] or [B, N, 1], got "
            f"{tuple(confidence.shape)}"
        )
    if confidence.shape[0] == 1 and batch > 1:
        confidence = confidence.expand(batch, -1, -1)
    if confidence.shape[0] != batch or confidence.shape[1] < num_frames:
        raise ValueError(
            "pose-only confidence does not cover pose sequence: "
            f"confidence={tuple(confidence.shape)}, poses={tuple(pose_data.shape)}"
        )

    quality_confidence = None
    if quality_only_confidence is not None:
        quality_confidence = quality_only_confidence.to(
            device=pose_data.device, dtype=pose_data.dtype
        )
        if quality_confidence.ndim == 1:
            quality_confidence = quality_confidence.unsqueeze(0)
        if quality_confidence.ndim == 3 and quality_confidence.shape[-1] == 1:
            quality_confidence = quality_confidence[..., 0]
        if quality_confidence.ndim != 2:
            raise ValueError(
                "quality_only_confidence must be [B, N] or [B, N, 1], got "
                f"{tuple(quality_only_confidence.shape)}"
            )
        if quality_confidence.shape[0] == 1 and batch > 1:
            quality_confidence = quality_confidence.expand(batch, -1)
        if (
            quality_confidence.shape[0] != batch
            or quality_confidence.shape[1] < num_frames
        ):
            raise ValueError(
                "quality_only_confidence does not cover pose sequence: "
                f"confidence={tuple(quality_confidence.shape)}, "
                f"poses={tuple(pose_data.shape)}"
            )
        quality_confidence = quality_confidence[:, :num_frames]

    # No local velocity/bias from a later full-IMU solve is accepted here.
    # `_pose_velocity_motion` inside the residual derives velocity exclusively
    # from this visual pose prefix; local biases are zero in this supervision.
    r_p, r_v, r_R, _, _, valid = full_preintegration_residual(
        pose_only_poses,
        imu_delta,
        imu_valid=imu_valid,
        imu_info=imu_info,
        imu_info_reference=imu_info_reference,
        imu_bias_jacobian=imu_bias_jacobian,
        imu_motion=None,
        gyro_bias=gyro_bias,
        accel_bias=accel_bias,
        gravity=gravity,
        imu_camera_translation=imu_camera_translation,
        imu_compose_order=imu_compose_order,
        imu_inverse=imu_inverse,
        max_residual=max_residual,
        # The quality ordering must be driven by the pose/prior mismatch, not
        # cancelled by the covariance reduction we intentionally inject for a
        # dropout.  IMU information remains an input to the quality head.
        use_imu_info_weighting=False,
        imu_optimize_bias=imu_optimize_bias,
        return_mask=True,
    )

    def scalar_weight(value, anchor):
        if torch.is_tensor(value):
            return value.to(device=anchor.device, dtype=anchor.dtype).reshape(())
        return anchor.new_tensor(float(value))

    energy = (
        scalar_weight(pos_weight, r_p) * r_p.square().mean(dim=-1)
        + scalar_weight(vel_weight, r_v) * r_v.square().mean(dim=-1)
        + scalar_weight(rot_weight, r_R) * r_R.square().mean(dim=-1)
    ).detach()
    valid = valid & torch.isfinite(energy)
    logits = torch.logit(
        confidence[:, 1:num_frames, 0].clamp(1e-5, 1.0 - 1e-5)
    )

    augmented = torch.zeros_like(valid)
    if augmented_mask is not None:
        augmented = augmented_mask.to(device=pose_data.device, dtype=torch.bool)
        if augmented.ndim == 1:
            augmented = augmented.unsqueeze(0)
        if augmented.shape[0] == 1 and batch > 1:
            augmented = augmented.expand(batch, -1)
        if augmented.shape[0] != batch or augmented.shape[1] < num_frames:
            raise ValueError(
                "augmented_mask must cover the pose sequence: "
                f"mask={tuple(augmented.shape)}, poses={tuple(pose_data.shape)}"
            )
        augmented = augmented[:, 1:num_frames]

    absolute_weight = float(absolute_weight)
    if absolute_weight < 0.0:
        raise ValueError(f"absolute_weight must be non-negative, got {absolute_weight}")
    energy_temperature = float(energy_temperature)
    if energy_temperature <= 0.0:
        raise ValueError(
            f"energy_temperature must be positive, got {energy_temperature}"
        )
    energy_reference_momentum = float(energy_reference_momentum)
    if not 0.0 <= energy_reference_momentum < 1.0:
        raise ValueError(
            "energy_reference_momentum must be in [0, 1), got "
            f"{energy_reference_momentum}"
        )

    # Absolute calibration is intentionally based on the same visual-only
    # pose/IMU residual as ranking.  The reference is updated from *clean*
    # intervals only, so train-time corruption is correctly treated as an
    # out-of-distribution quality drop rather than redefining "normal" IMU.
    absolute_loss = confidence.sum() * 0.0
    quality_only_absolute_loss = confidence.sum() * 0.0
    absolute_target_mean = 0.0
    absolute_target_clean_mean = 0.0
    absolute_target_augmented_mean = 0.0
    quality_target_mean = 0.0
    quality_target_clean_mean = 0.0
    quality_target_augmented_mean = 0.0
    energy_reference_value = float("nan")
    if absolute_weight > 0.0 and bool(valid.any().detach().cpu().item()):
        log_energy = torch.log1p(energy.clamp_min(0.0))
        reference_mask = valid & ~augmented
        if not bool(reference_mask.any().detach().cpu().item()):
            reference_mask = valid
        # Residual-centred modes use the clean median, which maps a normal
        # interval to a target of 0.5. For confidence-only IMU weighting the
        # supplied covariance already represents nominal information, so the
        # one-sided mode instead learns the clean 90th-percentile residual and
        # attenuates only above that data-derived tail threshold.
        if absolute_target_mode == "one_sided_nominal_quality":
            batch_reference = torch.quantile(
                log_energy[reference_mask], 0.90
            ).detach()
        else:
            batch_reference = torch.median(log_energy[reference_mask]).detach()

        if energy_log_reference is not None:
            if not torch.is_tensor(energy_log_reference) or energy_log_reference.numel() != 1:
                raise ValueError(
                    "energy_log_reference must be a scalar tensor when provided"
                )
            reference_tensor = energy_log_reference.to(
                device=pose_data.device, dtype=pose_data.dtype
            )
            previous_reference = reference_tensor.detach()
            if torch.isfinite(previous_reference):
                active_reference = previous_reference
                updated_reference = (
                    energy_reference_momentum * previous_reference
                    + (1.0 - energy_reference_momentum) * batch_reference
                )
            else:
                active_reference = batch_reference
                updated_reference = batch_reference
            with torch.no_grad():
                energy_log_reference.copy_(updated_reference.to(energy_log_reference))
        else:
            active_reference = batch_reference

        residual_targets = torch.sigmoid(
            (active_reference - log_energy) / energy_temperature
        ).detach()
        if absolute_target_mode == "median_residual":
            direct_targets = residual_targets
            quality_targets = residual_targets
        elif absolute_target_mode == "nominal_clean":
            # `imu_info` already represents the nominal raw-IMU covariance.
            # A confidence value is therefore an *extra*, relative information
            # correction: it must not attenuate intervals that were neither
            # corrupted nor identified as an abnormal raw-IMU sample.  The
            # synthetic corruption keeps its residual-derived target and the
            # ranking loss still lets a visual/IMU disagreement lower it.
            direct_targets = torch.where(
                augmented,
                residual_targets,
                torch.ones_like(residual_targets),
            )
            quality_targets = direct_targets
        elif absolute_target_mode == "split_nominal_quality":
            # The raw-quality MLP has no visual context. Its calibration is
            # therefore relative to the supplied IMU covariance: untouched
            # intervals retain nominal information, while corrupted intervals
            # learn a residual-derived attenuation. The fused DVI confidence
            # does have visual context, so it remains free to lower confidence
            # wherever the visual pose and IMU prior disagree.
            direct_targets = residual_targets
            quality_targets = torch.where(
                augmented,
                residual_targets,
                torch.ones_like(residual_targets),
            )
        elif absolute_target_mode == "one_sided_nominal_quality":
            # A residual in the normal clean range is not evidence that this
            # IMU factor is bad. Leave it at full covariance-relative
            # strength; smoothly lower only the high-residual tail. Synthetic
            # corruption remains residual-supervised and is also constrained
            # by the bad-vs-clean ranking term below.
            direct_targets = torch.where(
                augmented,
                residual_targets,
                torch.exp(
                    -((log_energy - active_reference).clamp_min(0.0))
                    / energy_temperature
                ),
            )
            # Raw quality has no visual context. Untouched IMU is nominal;
            # corrupted IMU receives the same residual-derived correction.
            quality_targets = torch.where(
                augmented,
                residual_targets,
                torch.ones_like(residual_targets),
            )
        else:
            raise ValueError(
                "absolute_target_mode must be 'median_residual', 'nominal_clean', "
                "'split_nominal_quality', or 'one_sided_nominal_quality', got "
                f"{absolute_target_mode!r}"
            )
        direct_confidence = confidence[:, 1:num_frames, 0].clamp(1e-5, 1.0 - 1e-5)
        absolute_loss = torch.nn.functional.binary_cross_entropy(
            direct_confidence[valid], direct_targets[valid]
        )
        if quality_confidence is not None:
            quality_factor = quality_confidence[:, 1:num_frames].clamp(1e-5, 1.0 - 1e-5)
            quality_only_absolute_loss = torch.nn.functional.binary_cross_entropy(
                quality_factor[valid], quality_targets[valid]
            )
            absolute_loss = 0.5 * (absolute_loss + quality_only_absolute_loss)

        absolute_target_mean = float(direct_targets[valid].mean().detach().item())
        quality_target_mean = float(quality_targets[valid].mean().detach().item())
        clean_target_mask = valid & ~augmented
        augmented_target_mask = valid & augmented
        if bool(clean_target_mask.any().detach().cpu().item()):
            absolute_target_clean_mean = float(
                direct_targets[clean_target_mask].mean().detach().item()
            )
            quality_target_clean_mean = float(
                quality_targets[clean_target_mask].mean().detach().item()
            )
        if bool(augmented_target_mask.any().detach().cpu().item()):
            absolute_target_augmented_mean = float(
                direct_targets[augmented_target_mask].mean().detach().item()
            )
            quality_target_augmented_mean = float(
                quality_targets[augmented_target_mask].mean().detach().item()
            )
        energy_reference_value = float(active_reference.detach().item())

    pair_losses = []
    aug_pair_losses = []
    pair_count = 0
    aug_pair_count = 0
    clean_confidences = []
    augmented_confidences = []
    for batch_index in range(batch):
        row_valid = valid[batch_index]
        if int(row_valid.sum().detach().item()) >= 2:
            row_energy = energy[batch_index, row_valid]
            row_logits = logits[batch_index, row_valid]
            # Strict upper triangle prevents each ordering constraint from
            # being counted twice.  Log energy keeps a rare extreme interval
            # from setting a numerical scale for all other pairs.
            energy_order = torch.log1p(row_energy.clamp_min(0.0))
            upper = torch.triu(
                torch.ones(
                    (row_energy.numel(), row_energy.numel()),
                    device=pose_data.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            higher_energy = energy_order[:, None] > energy_order[None, :]
            lower_energy = energy_order[:, None] < energy_order[None, :]
            ordered = upper & (higher_energy | lower_energy)
            if bool(ordered.any().detach().cpu().item()):
                sign = torch.where(higher_energy, 1.0, -1.0)
                # Higher visual/IMU disagreement => lower confidence logit.
                ranking = torch.nn.functional.softplus(
                    sign * (row_logits[:, None] - row_logits[None, :])
                )
                pair_losses.append(ranking[ordered].mean())
                pair_count += int(ordered.sum().detach().item())

        bad = row_valid & augmented[batch_index]
        clean = row_valid & ~augmented[batch_index]
        if bool(bad.any().detach().cpu().item()) and bool(clean.any().detach().cpu().item()):
            # Synthetic corruption has an unambiguous ordering label.  It is
            # still only a relative constraint, so it cannot become a second
            # global attenuation parameter.
            bad_logits = logits[batch_index, bad]
            clean_logits = logits[batch_index, clean]
            aug_ranking = torch.nn.functional.softplus(
                bad_logits[:, None] - clean_logits[None, :]
            )
            aug_pair_losses.append(aug_ranking.mean())
            aug_pair_count += int(aug_ranking.numel())

        row_confidence = confidence[batch_index, 1:num_frames, 0]
        if bool((row_valid & ~augmented[batch_index]).any().detach().cpu().item()):
            clean_confidences.append(row_confidence[row_valid & ~augmented[batch_index]])
        if bool((row_valid & augmented[batch_index]).any().detach().cpu().item()):
            augmented_confidences.append(row_confidence[row_valid & augmented[batch_index]])

    terms = []
    if pair_losses:
        terms.append(torch.stack(pair_losses).mean())
    if aug_pair_losses:
        terms.append(torch.stack(aug_pair_losses).mean())
    ranking_total = (
        torch.stack(terms).mean() if terms else confidence.sum() * 0.0
    )
    total = ranking_total + absolute_weight * absolute_loss

    clean_mean = (
        torch.cat(clean_confidences).mean()
        if clean_confidences
        else total.detach() * 0.0
    )
    augmented_mean = (
        torch.cat(augmented_confidences).mean()
        if augmented_confidences
        else total.detach() * 0.0
    )
    return total, {
        "imu_confidence_pose_prior": float(total.detach().item()),
        "imu_confidence_pose_prior_ranking": float(ranking_total.detach().item()),
        "imu_confidence_pose_prior_absolute": float(absolute_loss.detach().item()),
        "imu_confidence_pose_prior_quality_only_absolute": float(
            quality_only_absolute_loss.detach().item()
        ),
        "imu_confidence_pose_prior_target_mean": absolute_target_mean,
        "imu_confidence_pose_prior_target_clean_mean": absolute_target_clean_mean,
        "imu_confidence_pose_prior_target_augmented_mean": (
            absolute_target_augmented_mean
        ),
        "imu_confidence_pose_prior_quality_target_mean": quality_target_mean,
        "imu_confidence_pose_prior_quality_target_clean_mean": (
            quality_target_clean_mean
        ),
        "imu_confidence_pose_prior_quality_target_augmented_mean": (
            quality_target_augmented_mean
        ),
        "imu_confidence_pose_prior_energy_log_reference": energy_reference_value,
        "imu_confidence_pose_prior_pairs": float(pair_count),
        "imu_confidence_pose_prior_aug_pairs": float(aug_pair_count),
        "imu_confidence_pose_prior_clean_mean": float(clean_mean.detach().item()),
        "imu_confidence_pose_prior_aug_mean": float(augmented_mean.detach().item()),
    }


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

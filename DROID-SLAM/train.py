import sys

# droid_slam 내부 모듈을 import하기 위해 경로 추가
sys.path.append('droid_slam')

import cv2
import numpy as np
from collections import OrderedDict
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# dataset loader 생성 함수
from data_readers.factory import dataset_factory
from data_readers.sampling import BalancedSequenceDistributedSampler

# Lie group pose 표현
# SO3: rotation group
# SE3: rigid transformation, rotation + translation
# Sim3: similarity transformation, SE3 + scale
from lietorch import SO3, SE3, Sim3

# loss 함수 모듈
from geom import losses
from geom.losses import geodesic_loss, residual_loss, flow_loss
from geom.imu_factor import temporal_imu_quality_features

# 학습용 frame graph 생성 함수
from geom.graph_utils import build_frame_graph

# DROID neural network
from droid_net import DroidNet
from imu_runtime import configure_imu_noise_model

# 학습 log 저장/출력용 logger
from logger import Logger

# DDP training 관련 모듈
# 여러 GPU를 사용해 분산 학습하기 위한 PyTorch 기능
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(gpu, args):
    """
    Distributed Data Parallel 학습 환경 초기화 함수.

    gpu:
        현재 process가 사용할 GPU 번호.

    args.world_size:
        전체 GPU/process 수.

    역할:
        1. NCCL backend로 distributed process group 초기화
        2. random seed 설정
        3. 현재 process가 사용할 CUDA device 지정
    """

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=gpu
    )

    # 재현성을 위한 torch seed 고정
    torch.manual_seed(0)

    # 현재 process가 사용할 GPU 설정
    torch.cuda.set_device(gpu)


def show_image(image):
    """
    디버깅용 이미지 출력 함수.

    image:
        [3, H, W] 형태의 torch tensor.

    처리:
        [3, H, W] -> [H, W, 3]으로 바꾼 뒤 OpenCV로 표시.
    """

    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()


def build_temporal_graph(num_frames, radius=2):
    """
    temporal graph 생성 함수.

    기존 build_frame_graph는 pose/depth 기반으로 frame 간 covisibility나 flow distance를 보고
    edge를 만들 수 있다.

    하지만 dense depth가 없는 dataset 또는 IMU 중심 학습에서는
    단순히 시간적으로 가까운 frame끼리 edge를 연결하는 graph가 필요할 수 있다.

    num_frames:
        한 training sample 안의 frame 수.

    radius:
        frame index 차이가 radius 이하인 frame끼리 edge 연결.

    예:
        num_frames=7, radius=2이면
        0 -> 1,2
        1 -> 0,2,3
        2 -> 0,1,3,4
        ...
    """

    graph = OrderedDict()

    for i in range(num_frames):
        graph[i] = [
            j for j in range(num_frames)
            if i != j and abs(i - j) <= radius
        ]

    return graph


def freeze_non_imu_parameters(
    model,
    use_imu_data,
    use_imu_ba,
    use_acc_bias,
    use_learned_imu_residual_weights=False,
    use_component_imu_confidence=False,
    use_dvi_imu_confidence=False,
    use_dvi_imu_quality_fusion=False,
    freeze_imu_ba_weight=False,
    freeze_imu_confidence_head=False,
    imu_confidence_only_weighting=False,
):
    """
    기존 DROID visual/update network 파라미터를 freeze하고,
    IMU 관련 파라미터만 학습 가능하게 만드는 함수.

    이 함수는 DVI-SLAM의 3단계 학습 방식과 유사한 의도로 볼 수 있다.

    DVI-SLAM:
        기존 visual branch를 먼저 학습한 뒤,
        후반 단계에서 IMU confidence branch를 따로 학습.

    이 코드:
        pretrained DROID weight를 불러온 뒤,
        DROID 본체는 freeze하고 IMU bias / IMU confidence / IMU BA weight만 학습 가능하게 함.

    use_imu_data:
        gyro bias 학습 여부와 관련.

    use_imu_ba:
        IMU BA weight 및 update.imu_confidence branch 학습 여부.

    use_acc_bias:
        accelerometer bias 학습 여부.
    """

    # 일단 전체 파라미터 freeze
    for p in model.parameters():
        p.requires_grad_(False)

    # gyro bias는 IMU data를 사용할 때만 학습
    if use_imu_data:
        model.imu_gyro_bias.requires_grad_(True)

    # accelerometer bias는 full IMU BA/loss를 사용할 때만 학습
    if use_acc_bias:
        model.imu_acc_bias.requires_grad_(True)

    # IMU BA를 사용하는 경우:
    #   1. global IMU BA weight
    #   2. update module 내부 IMU confidence branch
    # 를 학습 가능하게 설정
    if use_imu_ba:
        # In confidence-only mode the factor's global strength is exactly
        # one.  Keep the legacy checkpoint parameter for compatibility, but
        # never optimize it: the confidence head is the sole learned
        # attenuation mechanism.
        if not freeze_imu_ba_weight and not imu_confidence_only_weighting:
            model.imu_ba_log_weight.requires_grad_(True)

        # The global IMU factor scale and per-interval confidence multiply in
        # DBA.  A scale-only calibration must be able to hold the confidence
        # head at its neutral value so the two mechanisms are identifiable.
        if not freeze_imu_confidence_head:
            if use_component_imu_confidence:
                for p in model.update.imu_component_confidence.parameters():
                    p.requires_grad_(True)
                for p in model.update.imu_quality_fusion.parameters():
                    p.requires_grad_(True)
            elif use_dvi_imu_confidence:
                for p in model.update.dvi_imu_confidence.parameters():
                    p.requires_grad_(True)
                if use_dvi_imu_quality_fusion:
                    for p in model.update.dvi_imu_quality_fusion.parameters():
                        p.requires_grad_(True)
            else:
                for p in model.update.imu_confidence.parameters():
                    p.requires_grad_(True)

    if use_learned_imu_residual_weights:
        model.imu_pos_log_weight.requires_grad_(True)
        model.imu_vel_log_weight.requires_grad_(True)


def count_trainable_parameters(model):
    """
    현재 학습 가능한 parameter 수를 세는 함수.

    freeze_non_imu 옵션을 썼을 때 실제로 IMU 관련 파라미터만
    학습되는지 확인하는 용도.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, path):
    """
    checkpoint 저장 함수.

    DDP로 감싸진 model은 실제 원본 model이 model.module 안에 있으므로,
    DDP 여부를 확인한 뒤 state_dict를 저장한다.

    이렇게 저장하면 나중에 DDP 없이도 load하기 쉬움.
    """

    module = model.module if hasattr(model, "module") else model
    torch.save(module.state_dict(), path)


def save_training_state(model, optimizer, scheduler, path, total_steps, sampler_epoch, rng):
    """Save an optimizer/scheduler continuation state beside a model ckpt.

    The ordinary ``*.pth`` checkpoint deliberately remains a plain DROID
    ``state_dict`` so demo/evaluation loaders stay backward-compatible.  The
    separate training-state file is used only when an experiment must release
    GPU memory for an intermediate validation run and then resume the same
    8k learning-rate schedule.
    """

    module = model.module if hasattr(model, "module") else model
    payload = {
        "format": "droid-training-state-v1",
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "total_steps": int(total_steps),
        "sampler_epoch": int(sampler_epoch),
        "numpy_generator_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    torch.save(payload, path)


def restore_training_state(model, optimizer, scheduler, path, rng):
    """Restore a state written by :func:`save_training_state`."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "droid-training-state-v1":
        raise ValueError(
            "resume_training_state is not a DROID training-state checkpoint: "
            f"{path}"
        )
    for key in ("model", "optimizer", "scheduler", "total_steps", "sampler_epoch"):
        if key not in payload:
            raise ValueError(f"resume_training_state is missing '{key}': {path}")

    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    rng.bit_generator.state = payload.get("numpy_generator_state", rng.bit_generator.state)
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"])
    if "cuda_rng_state" in payload:
        torch.cuda.set_rng_state(payload["cuda_rng_state"])

    return int(payload["total_steps"]), int(payload["sampler_epoch"])


def clamp_parameter_norm(param, max_norm):
    """
    bias parameter의 norm을 제한하는 함수.

    IMU bias는 학습 중 값이 과도하게 커질 수 있다.
    특히 gyro/acc bias를 end-to-end로 학습하면 비정상적으로 큰 bias로 loss를 줄이려는
    방향이 생길 수 있으므로 norm clamp를 제공한다.

    max_norm <= 0이면 clamp를 하지 않는다.
    """

    max_norm = float(max_norm)

    if max_norm <= 0.0:
        return

    with torch.no_grad():
        norm = param.norm()

        if norm > max_norm:
            param.mul_(max_norm / norm.clamp_min(1e-12))


def imu_confidence_saturation_loss(confidences, imu_valid=None, ceiling=0.95):
    """Penalize only the saturated upper tail of the learned IMU confidence.

    The branch is a sigmoid probability/precision surrogate. Without a small
    calibration term, both the global IMU weight and the confidence branch can
    minimize the training objective by making every factor maximally trusted.
    The penalty is normalized by the available range above ``ceiling``. This
    keeps its gradient bounded near one, where a logit-space penalty can
    dominate the actual IMU objective. The first confidence entry represents
    frame zero, which has no preceding IMU interval and is deliberately a
    constant neutral value. It must not be regularized as if it were a learned
    IMU factor confidence.
    """

    if not confidences:
        return None

    ceiling = float(ceiling)
    if not 0.0 < ceiling < 1.0:
        raise ValueError(f"imu confidence ceiling must be in (0, 1), got {ceiling}")

    losses = []
    for confidence in confidences:
        if confidence is None:
            continue
        if confidence.ndim == 1:
            confidence = confidence.unsqueeze(0)
        if confidence.ndim not in (2, 3) or confidence.shape[1] < 2:
            continue

        # c[:, j] belongs to the temporal IMU factor (j - 1, j). Frame zero
        # has no such factor, so it is a sentinel rather than a prediction.
        factor_confidence = confidence[:, 1:]

        if imu_valid is not None:
            valid = imu_valid.to(device=confidence.device, dtype=torch.bool)
            if valid.ndim == 1:
                valid = valid.unsqueeze(0)
            if valid.shape[0] == 1 and factor_confidence.shape[0] > 1:
                valid = valid.expand(factor_confidence.shape[0], -1)
            if valid.shape[0] != factor_confidence.shape[0] or valid.shape[1] < 2:
                raise ValueError(
                    "imu_valid must cover the temporal confidence sequence: "
                    f"confidence={tuple(confidence.shape)}, valid={tuple(valid.shape)}"
                )
            factor_confidence = factor_confidence[valid[:, 1:factor_confidence.shape[1] + 1]]
        else:
            factor_confidence = factor_confidence.reshape(-1)

        if factor_confidence.numel() == 0:
            continue
        excess = torch.relu(factor_confidence - ceiling) / (1.0 - ceiling)
        losses.append(excess.square().mean())

    if not losses:
        # Keep the returned zero attached to the graph so DDP sees the same
        # parameter usage pattern even for an all-invalid IMU batch.
        return confidences[-1].sum() * 0.0
    return torch.stack(losses).mean()


def imu_residual_weight_prior_loss(model, reference=0.05):
    """Anchor learned position/velocity weights relative to rotation.

    The global IMU BA weight already controls the overall visual-versus-IMU
    scale. These values only model category-relative corrections, so a small
    log-space prior keeps that decomposition identifiable.
    """

    weights = model.get_imu_full_residual_weights()
    target = weights.new_full((2,), float(reference)).clamp_min(1e-8)
    return torch.log(weights.clamp_min(1e-8) / target).square().mean()


def reset_imu_residual_weights(model, reference=0.05):
    """Start a relative p/v calibration run from the fixed 15DoF setting."""

    if not hasattr(model, "set_imu_full_residual_weights"):
        raise ValueError(
            "--reset_imu_residual_weights requires a DroidNet with "
            "set_imu_full_residual_weights()."
        )
    model.set_imu_full_residual_weights(reference, reference)


def reset_imu_confidence_head(
    model,
    value=0.5,
    use_dvi_imu_confidence=False,
    use_dvi_imu_quality_fusion=False,
):
    """Recalibrate the active scalar IMU-confidence logit layer.

    This is useful when a previous clean-sensor training run drove the sigmoid
    to one for every frame. The preceding GRU and confidence features remain
    intact; zeroing the final projection makes the initial confidence a known
    neutral value and lets the fine-tuning run learn a calibrated output.
    """

    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"IMU confidence reset value must be in (0, 1), got {value}")

    # The legacy scalar head is a small Sequential module, whereas the
    # DVI head is itself the documented 1x1 projection.  Resetting the
    # legacy head during a DVI run has no effect on the active branch and can
    # leave an inherited DVI checkpoint saturated near one.
    head = (
        model.update.dvi_imu_confidence
        if use_dvi_imu_confidence
        else model.update.imu_confidence
    )
    if isinstance(head, torch.nn.Conv2d):
        final_conv = head
    elif isinstance(head, torch.nn.Sequential):
        final_conv = next(
            (layer for layer in reversed(head) if isinstance(layer, torch.nn.Conv2d)),
            None,
        )
    else:
        final_conv = None
    if final_conv is None:
        raise TypeError("expected the active IMU confidence projection to be Conv2d")

    logit = float(np.log(value / (1.0 - value)))
    with torch.no_grad():
        final_conv.weight.zero_()
        final_conv.bias.fill_(logit)

    if use_dvi_imu_confidence and use_dvi_imu_quality_fusion:
        model.initialize_dvi_imu_quality_fusion()


def imu_confidence_schedule_loss(confidences, targets, imu_valid=None):
    """Teach a weak, manual confidence curriculum on full-IMU unrolls.

    ``targets[k]`` supervises the mean confidence of full-IMU unroll ``k``.
    The last target is reused if a caller supplies more unrolls than targets.
    This is intentionally a *mean* calibration loss: individual temporal
    factors remain free to deviate from the manual ramp based on the GRU
    state, so the learned branch can still represent data-dependent quality.
    """

    if not confidences:
        return None
    if not targets:
        raise ValueError("imu confidence schedule targets must not be empty")
    targets = tuple(float(value) for value in targets)
    if any(value < 0.0 or value > 1.0 for value in targets):
        raise ValueError(
            "imu confidence schedule targets must lie in [0, 1], got "
            f"{targets}"
        )

    losses = []
    for step, confidence in enumerate(confidences):
        if confidence is None:
            continue
        if confidence.ndim == 1:
            confidence = confidence.unsqueeze(0)
        if confidence.ndim not in (2, 3) or confidence.shape[1] < 2:
            continue

        # Frame zero has no preceding IMU interval, so it must not influence
        # the schedule mean.  This mirrors the saturation/NLL masking above.
        factor_confidence = confidence[:, 1:]
        if imu_valid is not None:
            valid = imu_valid.to(device=confidence.device, dtype=torch.bool)
            if valid.ndim == 1:
                valid = valid.unsqueeze(0)
            if valid.shape[0] == 1 and factor_confidence.shape[0] > 1:
                valid = valid.expand(factor_confidence.shape[0], -1)
            if (
                valid.shape[0] != factor_confidence.shape[0]
                or valid.shape[1] < factor_confidence.shape[1] + 1
            ):
                raise ValueError(
                    "imu_valid must cover the temporal confidence sequence: "
                    f"confidence={tuple(confidence.shape)}, valid={tuple(valid.shape)}"
                )
            factor_confidence = factor_confidence[
                valid[:, 1:factor_confidence.shape[1] + 1]
            ]

        if factor_confidence.numel() == 0:
            continue
        target = factor_confidence.new_tensor(targets[min(step, len(targets) - 1)])
        losses.append((factor_confidence.mean() - target).square())

    if not losses:
        # Keep DDP parameter usage identical for an all-invalid IMU batch.
        return confidences[-1].sum() * 0.0
    return torch.stack(losses).mean()


def augment_imu_confidence_inputs(
    imu_delta,
    imu_valid,
    imu_info,
    *,
    probability=0.0,
    gyro_noise_std=0.0,
    accel_noise_std=0.0,
    gyro_bias_drift_std=0.0,
    accel_bias_drift_std=0.0,
    dropout_probability=0.0,
    dropout_max_length=1,
    dropout_info_scale=0.05,
):
    """Inject train-only, preintegration-equivalent IMU degradation.

    The loader deliberately exposes composed camera-to-camera IMU priors, not
    the individual 200 Hz samples.  Perturbing ``[dR, dv, dp]`` with the
    corresponding integration powers of ``dt`` is therefore the equivalent
    of perturbing the raw gyro/accelerometer stream before preintegration:

    * white gyro/accelerometer noise changes ``dR``/``dv``/``dp``;
    * random-walk bias is cumulative over the selected clip;
    * a short sample dropout is represented as a held/incomplete interval
      with greatly reduced information, but remains a valid factor so the
      confidence head can learn to reject it.

    ``imu_info_reference`` is intentionally *not* changed by this routine.
    It remains the nominal sensor reference, while the degraded row carries
    lower relative information for the IMU-quality feature branch.

    Returns augmented ``(delta, info, degraded_mask, dropout_mask)``.  The
    two masks refer to temporal factors and have the same ``[B, N]`` layout
    as ``imu_valid``; frame zero is always false.
    """

    if imu_delta is None:
        return imu_delta, imu_info, None, None

    probability = float(probability)
    dropout_probability = float(dropout_probability)
    if probability <= 0.0 and dropout_probability <= 0.0:
        return imu_delta, imu_info, None, None

    if imu_delta.ndim != 3 or imu_delta.shape[-1] < 10:
        raise ValueError(
            "IMU confidence augmentation expects imu_delta [B, N, >=10], got "
            f"{tuple(imu_delta.shape)}"
        )

    batch, num_frames = imu_delta.shape[:2]
    delta = imu_delta.clone()
    info = imu_info.clone() if imu_info is not None else None

    if imu_valid is None:
        valid = torch.ones(
            (batch, num_frames), device=delta.device, dtype=torch.bool
        )
    else:
        valid = imu_valid.to(device=delta.device, dtype=torch.bool)
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.shape[0] == 1 and batch > 1:
            valid = valid.expand(batch, -1)
        if valid.shape != (batch, num_frames):
            raise ValueError(
                "imu_valid must match imu_delta during confidence augmentation: "
                f"valid={tuple(valid.shape)}, delta={tuple(imu_delta.shape)}"
            )

    temporal_valid = valid.clone()
    temporal_valid[:, 0] = False
    random_bad = (
        torch.rand((batch, num_frames), device=delta.device) < probability
    ) & temporal_valid

    # Draw contiguous short dropout runs per clip.  An actual invalid runtime
    # interval is omitted by BA; this train-only partial-dropout model instead
    # leaves a degraded factor in place so learned confidence has a gradient.
    dropout = torch.zeros_like(temporal_valid)
    max_length = max(1, int(dropout_max_length))
    for batch_index in range(batch):
        if num_frames <= 1 or torch.rand((), device=delta.device) >= dropout_probability:
            continue
        length = int(
            torch.randint(1, max_length + 1, (), device=delta.device).item()
        )
        length = min(length, num_frames - 1)
        start = int(
            torch.randint(1, num_frames - length + 1, (), device=delta.device).item()
        )
        dropout[batch_index, start:start + length] = temporal_valid[
            batch_index, start:start + length
        ]

    degraded = random_bad | dropout
    if not bool(degraded.any().detach().cpu().item()):
        return delta, info, degraded, dropout

    dt = delta[..., 0:1].clamp_min(1e-4)
    factor_mask = random_bad.unsqueeze(-1).to(dtype=delta.dtype)
    sqrt_dt = dt.sqrt()

    gyro_noise_std = float(gyro_noise_std)
    accel_noise_std = float(accel_noise_std)
    if gyro_noise_std > 0.0:
        delta[..., 1:4] = delta[..., 1:4] + factor_mask * (
            torch.randn_like(delta[..., 1:4]) * gyro_noise_std * sqrt_dt
        )
    if accel_noise_std > 0.0:
        acceleration_noise = (
            torch.randn_like(delta[..., 4:7]) * accel_noise_std * sqrt_dt
        )
        delta[..., 4:7] = delta[..., 4:7] + factor_mask * acceleration_noise
        delta[..., 7:10] = delta[..., 7:10] + factor_mask * (
            0.5 * acceleration_noise * dt
        )

    # The drift is a per-clip random walk in physical bias units.  It is only
    # active for clips that contain a synthetic bad interval, which keeps a
    # clean clip genuinely clean and provides clean-vs-degraded ranking pairs.
    clip_has_bad = random_bad.any(dim=1, keepdim=True).unsqueeze(-1)
    gyro_bias_drift_std = float(gyro_bias_drift_std)
    accel_bias_drift_std = float(accel_bias_drift_std)
    if gyro_bias_drift_std > 0.0:
        gyro_drift = torch.cumsum(
            torch.randn_like(delta[..., 1:4]) * gyro_bias_drift_std * sqrt_dt,
            dim=1,
        )
        delta[..., 1:4] = delta[..., 1:4] + gyro_drift * dt * clip_has_bad
    if accel_bias_drift_std > 0.0:
        accel_drift = torch.cumsum(
            torch.randn_like(delta[..., 4:7]) * accel_bias_drift_std * sqrt_dt,
            dim=1,
        )
        delta[..., 4:7] = delta[..., 4:7] + accel_drift * dt * clip_has_bad
        delta[..., 7:10] = delta[..., 7:10] + 0.5 * accel_drift * dt.square() * clip_has_bad

    if bool(dropout.any().detach().cpu().item()):
        # A zero-order-held interval is a conservative proxy for a short raw
        # sample gap.  Keep dt intact because the camera timestamps did not
        # disappear; only the measured increment became untrustworthy.
        dropout_mask = dropout.unsqueeze(-1)
        delta[..., 1:10] = torch.where(
            dropout_mask,
            torch.zeros_like(delta[..., 1:10]),
            delta[..., 1:10],
        )

    if info is not None:
        if info.ndim != 3 or info.shape[:2] != (batch, num_frames) or info.shape[-1] < 3:
            raise ValueError(
                "imu_info must be [B, N, >=3] and match imu_delta during "
                f"confidence augmentation, got {tuple(info.shape)}"
            )
        # The raw-quality MLP can see the lower propagated information.  The
        # factor is still valid: confidence, rather than a hard validity gate,
        # learns the remaining attenuation.
        ordinary_scale = 0.5
        info[..., :3] = torch.where(
            random_bad.unsqueeze(-1),
            info[..., :3] * ordinary_scale,
            info[..., :3],
        )
        info[..., :3] = torch.where(
            dropout.unsqueeze(-1),
            info[..., :3] * float(dropout_info_scale),
            info[..., :3],
        )

    return delta, info, degraded, dropout


def select_full_imu_stage_outputs(
    poses_est,
    imu_confidences,
    imu_motions,
    pose_only_steps,
):
    """Select only the full-IMU suffix of an unrolled two-stage solve.

    ``DroidNet`` keeps pose and confidence outputs aligned with every unroll
    step, but initializes and returns motion states only after the visual-only
    prefix.  Full preintegration supervision must use the matching suffix;
    otherwise the auxiliary loss would train on states that never saw an IMU
    factor in DBA.
    """

    start = int(pose_only_steps)
    if start < 0 or start >= len(poses_est):
        raise ValueError(
            "imu_train_pose_only_steps must leave at least one full IMU "
            f"unroll step, got start={start}, outputs={len(poses_est)}"
        )
    if imu_confidences and len(imu_confidences) != len(poses_est):
        raise ValueError(
            "DroidNet IMU confidence outputs must align with unrolled poses: "
            f"confidence={len(imu_confidences)}, poses={len(poses_est)}"
        )

    stage_poses = poses_est[start:]
    stage_confidences = imu_confidences[start:] if imu_confidences else []
    if imu_motions and len(imu_motions) != len(stage_poses):
        raise ValueError(
            "DroidNet IMU motion outputs must cover exactly the full-IMU "
            f"suffix: motion={len(imu_motions)}, full_steps={len(stage_poses)}"
        )
    return stage_poses, stage_confidences, imu_motions


def train(gpu, args):
    """
    DROID-SLAM 학습 함수.

    기존 DROID train.py와 비교했을 때 추가된 핵심:
        1. IMU prior를 dataset에서 함께 로드할 수 있음.
        2. DroidNet에 IMU BA 관련 인자를 전달할 수 있음.
        3. IMU rotation loss 또는 full IMU preintegration loss를 추가할 수 있음.
        4. gyro bias / accelerometer bias를 학습할 수 있음.
        5. pretrained DROID를 freeze하고 IMU branch만 학습할 수 있음.
    """

    ############################################################
    # 1. DDP 초기화
    ############################################################

    setup_ddp(gpu, args)

    # random restart를 위한 numpy RNG
    rng = np.random.default_rng(12345)

    # 한 sample에서 사용할 frame 수
    N = args.n_frames

    ############################################################
    # 2. 모델 생성
    ############################################################

    model = DroidNet()
    model.cuda()
    model.train()

    ############################################################
    # 3. checkpoint 로드
    ############################################################

    checkpoint_has_imu_ba_weight = False
    checkpoint_has_component_imu_confidence = False
    checkpoint_has_dvi_imu_quality_fusion = False
    if args.ckpt is not None:
        # DDP checkpoint에서 "module." prefix가 붙어 있을 수 있으므로 제거
        state_dict = OrderedDict([
            (k.replace("module.", ""), v)
            for (k, v) in torch.load(args.ckpt).items()
        ])
        checkpoint_has_imu_ba_weight = "imu_ba_log_weight" in state_dict
        checkpoint_has_component_imu_confidence = any(
            key.startswith("update.imu_component_confidence.")
            or key.startswith("update.imu_quality_fusion.")
            for key in state_dict
        )
        checkpoint_has_dvi_imu_quality_fusion = any(
            key.startswith("update.dvi_imu_quality_fusion.")
            for key in state_dict
        )

        # strict=False:
        #   기존 DROID checkpoint에는 새로 추가한 IMU parameter가 없을 수 있음.
        #   예: imu_gyro_bias, imu_acc_bias, imu_ba_log_weight, update.imu_confidence 등.
        #
        #   그래서 strict=False로 로드해야 기존 weight는 가져오고,
        #   새 IMU parameter는 새로 초기화된 상태로 둘 수 있다.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if gpu == 0 and (missing or unexpected):
            print(f"[CKPT] missing={missing}, unexpected={unexpected}")

    if args.use_component_imu_confidence and not checkpoint_has_component_imu_confidence:
        model.initialize_component_imu_confidence_from_scalar()
        if gpu == 0:
            print(
                "[IMU] initialized component confidence from the scalar GRU head; "
                "IMU-quality fusion starts neutral"
            )

    if args.use_dvi_imu_quality_fusion and not checkpoint_has_dvi_imu_quality_fusion:
        model.initialize_dvi_imu_quality_fusion()
        if gpu == 0:
            print(
                "[IMU] initialized scalar DVI IMU-quality fusion from an "
                "older checkpoint; quality contribution starts neutral"
            )

    ############################################################
    # 4. IMU bias 초기값 설정
    ############################################################

    # gyro bias 초기값을 command line에서 직접 주는 경우
    if args.imu_bias_init is not None:
        with torch.no_grad():
            model.imu_gyro_bias.copy_(
                torch.as_tensor(
                    args.imu_bias_init,
                    device="cuda",
                    dtype=model.imu_gyro_bias.dtype
                )
            )

    # accelerometer bias 초기값을 command line에서 직접 주는 경우
    if args.imu_acc_bias_init is not None:
        with torch.no_grad():
            model.imu_acc_bias.copy_(
                torch.as_tensor(
                    args.imu_acc_bias_init,
                    device="cuda",
                    dtype=model.imu_acc_bias.dtype
                )
            )

    ############################################################
    # 5. IMU 사용 모드 결정
    ############################################################

    # full IMU loss 사용 여부
    #
    # use_full_imu_loss:
    #   loss 함수에서 rotation뿐 아니라 position, velocity, bias까지 포함한
    #   full preintegration residual을 사용.
    #
    # use_full_imu_ba and use_imu_loss:
    #   BA 내부에도 full IMU residual을 쓰고, loss도 IMU를 쓰는 경우.
    use_full_imu_loss = args.use_full_imu_loss or (
        args.use_full_imu_ba and args.use_imu_loss
    )

    # The two-stage unroll has visual-only outputs first, followed by the
    # full T/v/ba/bg optimizer. IMU supervision is intentionally restricted
    # to the latter so training and runtime share the same state contract.
    imu_train_pose_only_steps = (
        args.imu_train_pose_only_steps if args.use_full_imu_ba else 0
    )
    if gpu == 0 and args.use_full_imu_ba:
        print(
            "[TRAIN] full IMU unroll schedule: "
            f"pose_only={imu_train_pose_only_steps}, "
            f"full_15dof={args.iters - imu_train_pose_only_steps}"
        )

    # IMU 데이터를 dataset에서 로드해야 하는지 여부
    #
    # IMU loss를 쓰거나, IMU BA를 쓰거나, full IMU loss를 쓰면
    # dataset에서 imu_delta, imu_valid가 필요하다.
    use_imu_data = args.use_imu_loss or args.use_imu_ba or use_full_imu_loss

    # 9DoF pose-velocity mode treats global gravity and both biases as fixed
    # calibration parameters. It may still *use* them in the residual, but it
    # must not optimize them during the confidence/weight fine-tuning stage.
    train_global_imu_bias = use_imu_data and not args.use_pose_velocity_imu_ba
    if args.freeze_global_imu_bias:
        train_global_imu_bias = False

    # Rotation-only IMU loss can train gyro bias alone. Full 15DoF training can
    # additionally train accel bias, whereas the fixed-bias 9DoF mode cannot.
    use_acc_bias = (
        (args.use_full_imu_ba or use_full_imu_loss)
        and train_global_imu_bias
    )

    # IMU BA global weight는 새 학습에서는 0.05로 초기화한다. 반면 IMU
    # checkpoint를 이어 학습할 때는 이미 학습된 값을 보존하고, 사용자가
    # --imu_ba_weight를 명시했을 때만 의도적으로 재설정한다.
    if (
        args.use_imu_ba
        and not args.eval_only
        and not args.imu_confidence_only_weighting
    ):
        if args.imu_ba_weight is not None:
            model.set_imu_ba_weight(args.imu_ba_weight)
            if gpu == 0:
                print(f"[IMU] reset global BA weight to {args.imu_ba_weight:g}")
        elif not checkpoint_has_imu_ba_weight:
            model.set_imu_ba_weight(0.05)
            if gpu == 0:
                print("[IMU] initialized new global BA weight to 0.05")
        elif gpu == 0:
            print(
                "[IMU] preserving checkpoint global BA weight="
                f"{model.get_imu_ba_weight().detach().item():.7g}"
            )
    elif args.use_imu_ba and args.imu_confidence_only_weighting and gpu == 0:
        print(
            "[IMU] confidence-only weighting: global IMU BA weight is fixed "
            "to 1.0; imu_ba_log_weight is ignored and frozen"
        )

    if args.reset_imu_confidence_head:
        reset_imu_confidence_head(
            model,
            args.imu_confidence_reset_value,
            use_dvi_imu_confidence=args.use_dvi_imu_confidence,
            use_dvi_imu_quality_fusion=args.use_dvi_imu_quality_fusion,
        )
        if args.use_component_imu_confidence:
            model.initialize_component_imu_confidence_from_scalar()
        model.reset_imu_confidence_calibration()
        if gpu == 0:
            print(
                "[IMU] reset active "
                f"{'DVI' if args.use_dvi_imu_confidence else 'scalar'} "
                "confidence head to "
                f"{args.imu_confidence_reset_value:g}"
            )
            print("[IMU] reset pre-BA confidence calibration reference")

    if args.reset_imu_residual_weights:
        reset_imu_residual_weights(model)
        if gpu == 0:
            print(
                "[IMU] reset learned relative position/velocity residual "
                "weights to (0.05, 0.05)"
            )

    ############################################################
    # 6. 사용하지 않는 IMU parameter freeze
    ############################################################

    # A fixed-bias 9DoF run deliberately keeps calibration bias out of the
    # optimizer. Full-state experiments may opt in through the old path.
    if not train_global_imu_bias:
        model.imu_gyro_bias.requires_grad_(False)
        model.imu_acc_bias.requires_grad_(False)

        if gpu == 0 and args.freeze_global_imu_bias:
            print(
                "[TRAIN] global IMU biases are frozen; full 15DoF DBA may "
                "still optimize its per-window local bias state."
            )

    # acc bias가 필요 없는 모드이면 acc bias freeze
    if not use_acc_bias:
        model.imu_acc_bias.requires_grad_(False)

    # IMU BA를 사용하지 않으면:
    #   - IMU BA weight freeze
    #   - update network 안의 IMU confidence branch freeze
    if not args.use_imu_ba:
        model.imu_ba_log_weight.requires_grad_(False)
        model.imu_pos_log_weight.requires_grad_(False)
        model.imu_vel_log_weight.requires_grad_(False)

        for p in model.update.imu_confidence.parameters():
            p.requires_grad_(False)
        for p in model.update.imu_component_confidence.parameters():
            p.requires_grad_(False)
        for p in model.update.imu_quality_fusion.parameters():
            p.requires_grad_(False)
        for p in model.update.dvi_imu_confidence.parameters():
            p.requires_grad_(False)
        for p in model.update.dvi_imu_quality_fusion.parameters():
            p.requires_grad_(False)

    if args.freeze_imu_ba_weight or args.imu_confidence_only_weighting:
        model.imu_ba_log_weight.requires_grad_(False)

    if not args.use_learned_imu_residual_weights:
        model.imu_pos_log_weight.requires_grad_(False)
        model.imu_vel_log_weight.requires_grad_(False)

    # Only one confidence branch participates in a forward pass.  Explicitly
    # freezing the inactive branch avoids DDP's unused-parameter failure while
    # keeping old scalar-checkpoint training behavior unchanged.
    if args.use_component_imu_confidence:
        for p in model.update.imu_confidence.parameters():
            p.requires_grad_(False)
        for p in model.update.dvi_imu_confidence.parameters():
            p.requires_grad_(False)
        for p in model.update.dvi_imu_quality_fusion.parameters():
            p.requires_grad_(False)
    elif args.use_dvi_imu_confidence:
        for p in model.update.imu_confidence.parameters():
            p.requires_grad_(False)
        for p in model.update.imu_component_confidence.parameters():
            p.requires_grad_(False)
        for p in model.update.imu_quality_fusion.parameters():
            p.requires_grad_(False)
        if not args.use_dvi_imu_quality_fusion:
            for p in model.update.dvi_imu_quality_fusion.parameters():
                p.requires_grad_(False)
    else:
        for p in model.update.imu_component_confidence.parameters():
            p.requires_grad_(False)
        for p in model.update.imu_quality_fusion.parameters():
            p.requires_grad_(False)
        for p in model.update.dvi_imu_confidence.parameters():
            p.requires_grad_(False)
        for p in model.update.dvi_imu_quality_fusion.parameters():
            p.requires_grad_(False)

    if gpu == 0 and args.use_pose_velocity_imu_ba:
        print(
            "[TRAIN] pose-velocity 9DoF: local DBA optimizes pose+velocity; "
            "gravity and global IMU biases are fixed calibration parameters. "
            "Trainable network parameters are the IMU confidence head and "
            "enabled IMU scalar weights."
        )

    ############################################################
    # 7. 기존 DROID 파라미터 freeze 옵션
    ############################################################

    if args.freeze_non_imu:
        freeze_non_imu_parameters(
            model,
            use_imu_data=train_global_imu_bias,
            use_imu_ba=args.use_imu_ba,
            use_acc_bias=use_acc_bias,
            use_learned_imu_residual_weights=args.use_learned_imu_residual_weights,
            use_component_imu_confidence=args.use_component_imu_confidence,
            use_dvi_imu_confidence=args.use_dvi_imu_confidence,
            use_dvi_imu_quality_fusion=args.use_dvi_imu_quality_fusion,
            freeze_imu_ba_weight=args.freeze_imu_ba_weight,
            freeze_imu_confidence_head=args.freeze_imu_confidence_head,
            imu_confidence_only_weighting=args.imu_confidence_only_weighting,
        )

        if gpu == 0:
            if args.ckpt is None:
                print(
                    "[WARN] --freeze_non_imu is set without --ckpt; "
                    "IMU heads train on random DROID features."
                )

            print(
                f"[TRAIN] freeze_non_imu=True, "
                f"trainable_parameters={count_trainable_parameters(model)}"
            )

    ############################################################
    # 8. eval only 모드
    ############################################################

    if args.eval_only:
        model.eval()

    ############################################################
    # 9. Dataset 생성
    ############################################################

    # args.datasets가 있으면 해당 dataset 사용,
    # 없으면 기본 tartan 사용
    dataset_names = args.datasets if args.datasets is not None else ["tartan"]

    db = dataset_factory(
        dataset_names,
        datapath=args.datapath,
        n_frames=args.n_frames,
        fmin=args.fmin,
        fmax=args.fmax,

        # 추가된 IMU 관련 dataset 옵션
        use_imu=use_imu_data,
        imu_prior_name=args.imu_prior_name,
        imu_require=args.imu_require,
        imu_train_frame_strides=args.imu_train_frame_strides,
        imu_train_gyro_noise_density=args.imu_train_gyro_noise_density,
        imu_train_accel_noise_density=args.imu_train_accel_noise_density,
        euroc_camera_frame_stride=args.euroc_camera_frame_stride,
        euroc_keyframe_manifest_root=args.euroc_keyframe_manifest_root,
        euroc_sequence_include=args.euroc_sequence_include,
        euroc_sequence_exclude=args.euroc_sequence_exclude,
    )

    # dataset에 dense depth가 있는지 확인
    #
    # TartanAir처럼 dense depth/disparity가 있으면 flow_loss 사용 가능.
    # 실제 수집 데이터처럼 dense depth가 없으면 flow_loss 계산이 어려움.
    has_dense_depth = getattr(db, "has_dense_depth", True)

    # flow loss를 학습할지 여부
    train_flow_loss = (
        has_dense_depth
        and not args.disable_flow_loss
        and args.w3 != 0.0
    )

    if gpu == 0 and not has_dense_depth:
        print("[DATA] dense depth is unavailable; using temporal graph and disabling flow_loss")

    # flow loss를 학습하지 않는 경우 upmask branch를 freeze
    #
    # upmask는 disparity upsampling에 사용되며 flow_loss/disp supervision과 관련이 깊다.
    # dense depth가 없으면 이 branch를 굳이 학습시키지 않는다.
    if not train_flow_loss:
        for p in model.update.agg.upmask.parameters():
            p.requires_grad_(False)

    ############################################################
    # 10. DDP wrapping
    ############################################################

    model = DDP(
        model,
        device_ids=[gpu],
        find_unused_parameters=False
    )

    ############################################################
    # 11. Distributed DataLoader 생성
    ############################################################

    if args.balance_sequence_sampling:
        train_sampler = BalancedSequenceDistributedSampler(
            db,
            num_replicas=args.world_size,
            rank=gpu,
            seed=0,
        )
        if gpu == 0:
            print(
                "[DATA] balanced sequence sampling enabled; each discovered "
                "sequence receives equal expected clip mass"
            )
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            db,
            shuffle=True,
            num_replicas=args.world_size,
            rank=gpu
        )

    train_loader = DataLoader(
        db,
        batch_size=args.batch,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )

    ############################################################
    # 12. Optimizer / Scheduler 생성
    ############################################################

    if args.eval_only:
        optimizer = None
        scheduler = None

    else:
        # requires_grad=True인 parameter만 optimizer에 넣는다.
        #
        # freeze_non_imu=True이면 여기에는 IMU 관련 parameter만 들어간다.
        trainable_params = [
            p for p in model.parameters()
            if p.requires_grad
        ]

        if len(trainable_params) == 0:
            raise ValueError(
                "no trainable parameters; check --freeze_non_imu / IMU flags"
            )

        optimizer = torch.optim.Adam(
            trainable_params,
            lr=args.lr,
            weight_decay=1e-5
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            args.lr,
            args.steps,
            pct_start=0.01,
            cycle_momentum=False
        )

    ############################################################
    # 13. Logger 및 training loop 준비
    ############################################################

    logger = Logger(args.name, scheduler, sum_freq=args.log_freq)

    should_keep_training = True
    total_steps = 0
    sampler_epoch = 0
    if args.resume_training_state is not None:
        total_steps, sampler_epoch = restore_training_state(
            model,
            optimizer,
            scheduler,
            args.resume_training_state,
            rng,
        )
        if total_steps >= args.steps:
            raise ValueError(
                "resume_training_state has already reached or exceeded "
                f"--steps: restored={total_steps}, steps={args.steps}"
            )
        if args.stop_after_steps <= total_steps:
            raise ValueError(
                "--stop_after_steps must be greater than the restored step: "
                f"restored={total_steps}, stop_after={args.stop_after_steps}"
            )
        if gpu == 0:
            print(
                "[RESUME] restored optimizer/scheduler/training state "
                f"from {args.resume_training_state}; step={total_steps}, "
                f"next_stop={args.stop_after_steps}, total_schedule={args.steps}"
            )

    ############################################################
    # 14. Main training loop
    ############################################################

    while should_keep_training:
        # DistributedSampler otherwise repeats its initial shuffle order on
        # every pass through the finite dataset. The balanced sampler follows
        # the same PyTorch epoch contract.
        train_sampler.set_epoch(sampler_epoch)
        sampler_epoch += 1
        for i_batch, item in enumerate(train_loader):

            ####################################################
            # 14-1. gradient 초기화
            ####################################################

            if optimizer is not None:
                optimizer.zero_grad()

            ####################################################
            # 14-2. batch item CUDA 이동
            ####################################################

            # item은 dataset에 따라 길이가 달라질 수 있다.
            #
            # 기본:
            #   images, poses, disps, intrinsics
            #
            # IMU 사용 시:
            #   images, poses, disps, intrinsics, imu_delta, imu_valid
            item = [
                x.to("cuda") if torch.is_tensor(x) else x
                for x in item
            ]

            images, poses, disps, intrinsics = item[:4]

            # IMU preintegration delta
            # 예: frame 간 ΔR, Δp, Δv, dt 등으로 구성될 수 있음.
            imu_delta = item[4] if len(item) > 4 else None

            # IMU edge가 유효한지 나타내는 mask
            imu_valid = item[5] if len(item) > 5 else None

            # IMU preintegration covariance/information
            # column order: [rot_info, vel_info, pos_info]
            imu_info = item[6] if len(item) > 6 else None

            # IMU preintegration bias Jacobians.
            # column order: [J_R_bg, J_v_ba, J_v_bg, J_p_ba, J_p_bg]
            imu_bias_jacobian = item[7] if len(item) > 7 else None

            # Nominal raw-camera information for each sequence. This is kept
            # separate from the selected frame intervals so composed IMU
            # covariance remains meaningful in the training BA/loss.
            imu_info_reference = item[8] if len(item) > 8 else None

            # Confidence training sees train-only IMU degradation *before*
            # both its direct raw-quality features and the full-IMU DBA.  The
            # visual-only prefix below deliberately receives no IMU factor,
            # making it a valid independent teacher for this augmented prior.
            imu_confidence_augmented = None
            imu_confidence_dropout = None
            if args.use_imu_confidence_augmentation:
                (
                    imu_delta,
                    imu_info,
                    imu_confidence_augmented,
                    imu_confidence_dropout,
                ) = augment_imu_confidence_inputs(
                    imu_delta,
                    imu_valid,
                    imu_info,
                    probability=args.imu_confidence_aug_probability,
                    gyro_noise_std=args.imu_confidence_aug_gyro_noise_std,
                    accel_noise_std=args.imu_confidence_aug_accel_noise_std,
                    gyro_bias_drift_std=args.imu_confidence_aug_gyro_bias_drift_std,
                    accel_bias_drift_std=args.imu_confidence_aug_accel_bias_drift_std,
                    dropout_probability=args.imu_confidence_aug_dropout_probability,
                    dropout_max_length=args.imu_confidence_aug_dropout_max_length,
                    dropout_info_scale=args.imu_confidence_aug_dropout_info_scale,
                )

            ####################################################
            # 14-3. pose convention 변환
            ####################################################

            # dataset pose가 world-to-camera라면,
            # DROID 내부 학습에서는 camera-to-world 형태로 쓰기 위해 inverse.
            Ps = SE3(poses).inv()

            # 초기 pose estimate는 identity 형태로 생성
            Gs = SE3.IdentityLike(Ps)

            ####################################################
            # 14-4. frame graph 생성
            ####################################################

            if not has_dense_depth:
                # dense depth가 없으면 build_frame_graph를 쓰기 어렵기 때문에
                # 시간적으로 가까운 frame끼리만 연결
                graph = build_temporal_graph(N)

            elif np.random.rand() < 0.5:
                # pose/depth/intrinsics 기반으로 학습용 frame graph 생성
                graph = build_frame_graph(
                    poses,
                    disps,
                    intrinsics,
                    num=args.edges
                )

            else:
                # 절반 확률로 단순 temporal graph 사용
                graph = build_temporal_graph(N)

            ####################################################
            # 14-5. 초기 pose / disparity 설정
            ####################################################

            # 첫 번째 pose는 GT pose로 고정
            Gs.data[:, 0] = Ps.data[:, 0].clone()

            # 나머지 frame pose는 두 번째 GT pose로 초기화
            # 기존 DROID 학습 방식과 동일한 구조
            Gs.data[:, 1:] = Ps.data[:, [1]].clone()

            # 초기 disparity는 1로 설정
            # DROID는 1/8 해상도 disparity를 최적화하므로
            # disps[:,:,3::8,3::8]와 같은 shape의 tensor를 사용
            disp0 = torch.ones_like(disps[:, :, 3::8, 3::8])

            ####################################################
            # 14-6. Random restart loop
            ####################################################

            # restart_prob 확률 구조:
            #   최소 1번은 실행되고,
            #   r < restart_prob이면 추가 restart가 이어질 수 있음.
            r = 0

            while r < args.restart_prob:
                r = rng.random()

                # projection은 1/8 feature resolution에서 수행되므로 intrinsics도 /8
                intrinsics0 = intrinsics / 8.0

                # full IMU BA와 full IMU loss를 모두 사용하는 경우
                # model forward에서 imu_motions를 반환받음
                return_imu_motion = args.use_full_imu_ba and use_full_imu_loss

                ################################################
                # 14-6-1. DroidNet forward
                ################################################

                with torch.set_grad_enabled(not args.eval_only):
                    model_output = model(
                        Gs,
                        images,
                        disp0,
                        intrinsics0,
                        graph,
                        num_steps=args.iters,
                        fixedp=2,

                        # IMU bias 반환 여부
                        return_imu_bias=use_imu_data,

                        # IMU BA를 사용할 때만 forward에 imu_delta/imu_valid 전달
                        imu_delta=imu_delta if args.use_imu_ba else None,
                        imu_valid=imu_valid if args.use_imu_ba else None,
                        imu_info=imu_info if args.use_imu_ba else None,
                        imu_info_reference=imu_info_reference if args.use_imu_ba else None,
                        imu_bias_jacobian=imu_bias_jacobian if args.use_imu_ba else None,

                        # Confidence-only mode has no learned or hand-tuned
                        # global IMU attenuation.  A confidence of one means
                        # that the preintegration factor enters DBA at its
                        # full covariance-relative strength.
                        imu_ba_weight=(
                            1.0
                            if args.imu_confidence_only_weighting
                            else (None if args.use_imu_ba else 0.0)
                        ),
                        imu_confidence_only_weighting=(
                            args.imu_confidence_only_weighting
                        ),

                        # IMU residual clipping / confidence 관련 설정
                        imu_ba_max_residual=args.imu_max_residual,
                        imu_confidence_floor=args.imu_conf_floor,
                        return_imu_confidence=args.use_imu_ba,
                        use_component_imu_confidence=args.use_component_imu_confidence,
                        use_dvi_imu_confidence=args.use_dvi_imu_confidence,
                        use_dvi_imu_quality_fusion=args.use_dvi_imu_quality_fusion,

                        # full IMU BA 관련 설정
                        use_full_imu_ba=args.use_full_imu_ba,
                        imu_pose_only_steps=imu_train_pose_only_steps,
                        imu_full_pos_weight=args.imu_full_pos_weight,
                        imu_full_vel_weight=args.imu_full_vel_weight,
                        use_learned_imu_residual_weights=args.use_learned_imu_residual_weights,
                        imu_full_bias_weight=args.imu_full_bias_weight,
                        imu_velocity_init=args.imu_velocity_init,
                        imu_motion_prior_weight=args.imu_motion_prior_weight,
                        imu_local_bias_prior_weight=args.imu_local_bias_prior_weight,
                        imu_gravity=args.imu_gravity,
                        imu_full_compose_order=args.imu_full_compose_order,
                        imu_full_inverse=args.imu_full_inverse,
                        use_imu_info_weighting=args.use_imu_info_weighting,
                        imu_info_weight_clip=args.imu_info_weight_clip,
                        imu_info_weight_eps=args.imu_info_weight_eps,
                        imu_optimize_bias=not args.use_pose_velocity_imu_ba,
                        use_adaptive_imu_gate=args.use_adaptive_imu_gate,
                        imu_gate_min=args.imu_gate_min,
                        imu_gate_max=args.imu_gate_max,
                        imu_gate_visual_thresh=args.imu_gate_visual_thresh,
                        imu_gate_visual_boost=args.imu_gate_visual_boost,
                        imu_gate_motion_thresh_deg=args.imu_gate_motion_thresh_deg,
                        imu_gate_motion_boost=args.imu_gate_motion_boost,
                        imu_gate_residual_soft_deg=args.imu_gate_residual_soft_deg,
                        imu_gate_residual_hard_deg=args.imu_gate_residual_hard_deg,
                        imu_gate_residual_pos_soft=args.imu_gate_residual_pos_soft,
                        imu_gate_residual_vel_soft=args.imu_gate_residual_vel_soft,
                        return_imu_adaptive_gate=args.use_adaptive_imu_gate,
                        return_imu_motion=return_imu_motion,
                    )

                ################################################
                # 14-6-2. model output unpacking
                ################################################

                imu_confidences = []
                imu_adaptive_gates = []
                imu_motions = []

                if args.use_imu_ba:
                    poses_est, disps_est, residuals, gyro_bias, imu_confidences = model_output[:5]
                    output_index = 5
                    if args.use_adaptive_imu_gate:
                        imu_adaptive_gates = model_output[output_index]
                        output_index += 1
                    if return_imu_motion:
                        imu_motions = model_output[output_index]

                elif use_imu_data:
                    if return_imu_motion:
                        (
                            poses_est,
                            disps_est,
                            residuals,
                            gyro_bias,
                            imu_motions,
                        ) = model_output
                    else:
                        (
                            poses_est,
                            disps_est,
                            residuals,
                            gyro_bias,
                        ) = model_output

                else:
                    poses_est, disps_est, residuals = model_output
                    gyro_bias = None

                imu_loss_poses = poses_est
                imu_loss_confidences = imu_confidences
                imu_loss_motions = imu_motions
                if args.use_full_imu_ba:
                    (
                        imu_loss_poses,
                        imu_loss_confidences,
                        imu_loss_motions,
                    ) = select_full_imu_stage_outputs(
                        poses_est,
                        imu_confidences,
                        imu_motions,
                        imu_train_pose_only_steps,
                    )

                ################################################
                # 14-6-3. 기존 DROID visual loss 계산
                ################################################

                # pose geodesic loss
                geo_loss, geo_metrics = losses.geodesic_loss(
                    Ps,
                    poses_est,
                    graph,
                    do_scale=False
                )

                # reprojection residual loss
                res_loss, res_metrics = losses.residual_loss(
                    residuals
                )

                # flow loss
                # dense depth가 있을 때만 계산
                if train_flow_loss:
                    flo_loss, flo_metrics = losses.flow_loss(
                        Ps,
                        disps,
                        poses_est,
                        disps_est,
                        intrinsics,
                        graph
                    )
                else:
                    flo_loss = torch.zeros((), device=images.device)
                    flo_metrics = {
                        "f_error": 0.0,
                        "1px": 0.0
                    }

                ################################################
                # 14-6-4. IMU loss 계산
                ################################################

                if args.use_imu_loss and use_full_imu_loss:
                    # full IMU preintegration loss
                    #
                    # rotation뿐 아니라 position, velocity, bias residual까지 포함.
                    # DVI-SLAM의 정식 IMU factor에 가까운 방향.
                    core_model = model.module if hasattr(model, 'module') else model
                    if args.use_learned_imu_residual_weights:
                        learned_residual_weights = core_model.get_imu_full_residual_weights()
                        imu_loss_pos_weight = learned_residual_weights[0]
                        imu_loss_vel_weight = learned_residual_weights[1]
                    else:
                        imu_loss_pos_weight = args.imu_loss_pos_weight
                        imu_loss_vel_weight = args.imu_loss_vel_weight

                    imu_loss, imu_metrics = losses.imu_full_preintegration_loss(
                        imu_loss_poses,
                        imu_delta,
                        imu_valid=imu_valid,
                        imu_info=imu_info,
                        imu_info_reference=imu_info_reference,
                        imu_bias_jacobian=imu_bias_jacobian,
                        imu_motions=(
                            imu_loss_motions if len(imu_loss_motions) > 0 else None
                        ),
                        gyro_bias=gyro_bias,
                        accel_bias=core_model.imu_acc_bias,
                        gamma=args.imu_gamma,
                        max_residual=args.imu_max_residual,
                        smooth_beta=args.imu_smooth_beta,
                        pos_weight=imu_loss_pos_weight,
                        vel_weight=imu_loss_vel_weight,
                        rot_weight=args.imu_loss_rot_weight,
                        # In fixed-bias 9DoF mode there are no local bias
                        # states in the active DBA state, so a bias random-walk
                        # loss is intentionally not part of this objective.
                        bias_weight=(
                            0.0
                            if args.use_pose_velocity_imu_ba
                            else args.imu_loss_bias_weight
                        ),
                        gravity=args.imu_gravity,
                        imu_compose_order=args.imu_full_compose_order,
                        imu_inverse=args.imu_full_inverse,
                        use_imu_info_weighting=args.use_imu_info_weighting,
                        imu_info_weight_clip=args.imu_info_weight_clip,
                        imu_info_weight_eps=args.imu_info_weight_eps,
                        imu_optimize_bias=not args.use_pose_velocity_imu_ba,
                    )

                elif args.use_imu_loss:
                    # rotation-only IMU loss
                    #
                    # gyro preintegration 기반으로 frame 간 relative rotation이
                    # pose_est의 relative rotation과 일치하도록 학습.
                    #
                    # full 상태 v, ba, bg를 모두 다루기 전의 가벼운 IMU loss.
                    imu_loss, imu_metrics = losses.imu_rotation_bias_loss(
                        poses_est,
                        imu_delta,
                        imu_valid=imu_valid,
                        gyro_bias=gyro_bias,
                        gamma=args.imu_gamma,
                        max_residual=args.imu_max_residual,
                        smooth_beta=args.imu_smooth_beta,
                    )

                else:
                    # IMU loss를 쓰지 않는 경우 0으로 둠
                    imu_loss = torch.zeros((), device=images.device)

                    imu_metrics = {
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

                ################################################
                # 14-6-5. bias regularization loss
                ################################################

                bias_loss = torch.zeros((), device=images.device)

                # gyro bias norm regularization
                if (
                    train_global_imu_bias
                    and gyro_bias is not None
                    and args.w_imu_bias != 0.0
                ):
                    bias_loss = bias_loss + gyro_bias.norm()

                # acc bias norm regularization
                if use_acc_bias and args.w_imu_bias != 0.0:
                    core_model = model.module if hasattr(model, 'module') else model
                    bias_loss = bias_loss + core_model.imu_acc_bias.norm()

                confidence_reg_loss = torch.zeros((), device=images.device)
                if (
                    args.use_imu_ba
                    and args.w_imu_confidence_saturation != 0.0
                    and len(imu_loss_confidences) > 0
                ):
                    confidence_reg_loss = imu_confidence_saturation_loss(
                        imu_loss_confidences,
                        imu_valid=imu_valid,
                        ceiling=args.imu_confidence_saturation_ceiling,
                    )

                confidence_nll_loss = torch.zeros((), device=images.device)
                confidence_nll_metrics = {
                    "imu_confidence_nll": 0.0,
                    "imu_confidence_nll_edges": 0.0,
                    "imu_confidence_nll_mean": 0.0,
                }
                if (
                    args.use_imu_ba
                    and use_full_imu_loss
                    and args.w_imu_confidence_nll != 0.0
                    and len(imu_loss_confidences) > 0
                ):
                    core_model = model.module if hasattr(model, 'module') else model
                    if args.use_learned_imu_residual_weights:
                        learned_residual_weights = core_model.get_imu_full_residual_weights()
                        confidence_pos_weight = learned_residual_weights[0]
                        confidence_vel_weight = learned_residual_weights[1]
                    else:
                        confidence_pos_weight = args.imu_loss_pos_weight
                        confidence_vel_weight = args.imu_loss_vel_weight
                    confidence_nll_loss, confidence_nll_metrics = (
                        losses.imu_confidence_nll_loss(
                            imu_loss_poses,
                            imu_loss_confidences,
                            imu_delta,
                            imu_valid=imu_valid,
                            imu_info=imu_info,
                            imu_info_reference=imu_info_reference,
                            imu_bias_jacobian=imu_bias_jacobian,
                            imu_motions=(
                                imu_loss_motions
                                if len(imu_loss_motions) > 0
                                else None
                            ),
                            gyro_bias=gyro_bias,
                            accel_bias=core_model.imu_acc_bias,
                            gravity=args.imu_gravity,
                            gamma=args.imu_gamma,
                            max_residual=args.imu_max_residual,
                            pos_weight=confidence_pos_weight,
                            vel_weight=confidence_vel_weight,
                            rot_weight=args.imu_loss_rot_weight,
                            imu_compose_order=args.imu_full_compose_order,
                            imu_inverse=args.imu_full_inverse,
                            use_imu_info_weighting=args.use_imu_info_weighting,
                            imu_info_weight_clip=args.imu_info_weight_clip,
                            imu_info_weight_eps=args.imu_info_weight_eps,
                            imu_optimize_bias=not args.use_pose_velocity_imu_ba,
                        )
                    )

                confidence_pose_prior_loss = torch.zeros((), device=images.device)
                confidence_pose_prior_metrics = {
                    "imu_confidence_pose_prior": 0.0,
                    "imu_confidence_pose_prior_ranking": 0.0,
                    "imu_confidence_pose_prior_absolute": 0.0,
                    "imu_confidence_pose_prior_quality_only_absolute": 0.0,
                    "imu_confidence_pose_prior_target_mean": 0.0,
                    "imu_confidence_pose_prior_target_clean_mean": 0.0,
                    "imu_confidence_pose_prior_target_augmented_mean": 0.0,
                    "imu_confidence_pose_prior_quality_target_mean": 0.0,
                    "imu_confidence_pose_prior_quality_target_clean_mean": 0.0,
                    "imu_confidence_pose_prior_quality_target_augmented_mean": 0.0,
                    "imu_confidence_pose_prior_energy_log_reference": float("nan"),
                    "imu_confidence_pose_prior_pairs": 0.0,
                    "imu_confidence_pose_prior_aug_pairs": 0.0,
                    "imu_confidence_pose_prior_clean_mean": 0.0,
                    "imu_confidence_pose_prior_aug_mean": 0.0,
                }
                if (
                    args.use_imu_ba
                    and args.w_imu_confidence_pose_prior != 0.0
                    and imu_train_pose_only_steps > 0
                    and len(imu_confidences) >= imu_train_pose_only_steps
                ):
                    # Index pose_only_steps-1 is the last output of the
                    # strictly visual prefix. It has never entered full IMU
                    # DBA, whereas the confidence is generated from the same
                    # visual ConvGRU state and raw/preintegrated IMU features.
                    core_model = model.module if hasattr(model, 'module') else model
                    if args.use_learned_imu_residual_weights:
                        pose_prior_pos_weight = core_model.get_imu_full_residual_weights()[0]
                        pose_prior_vel_weight = core_model.get_imu_full_residual_weights()[1]
                    else:
                        pose_prior_pos_weight = args.imu_loss_pos_weight
                        pose_prior_vel_weight = args.imu_loss_vel_weight
                    quality_only_confidence = None
                    if args.use_dvi_imu_quality_fusion:
                        temporal_quality = temporal_imu_quality_features(
                            imu_delta,
                            imu_info,
                            num_frames=poses_est[0].data.shape[1],
                        )
                        quality_only_confidence = (
                            core_model.update.dvi_imu_quality_confidence(
                                temporal_quality
                            )
                        )
                    (
                        confidence_pose_prior_loss,
                        confidence_pose_prior_metrics,
                    ) = losses.imu_confidence_pose_prior_ranking_loss(
                        poses_est[imu_train_pose_only_steps - 1],
                        imu_confidences[imu_train_pose_only_steps - 1],
                        imu_delta,
                        imu_valid=imu_valid,
                        imu_info=imu_info,
                        imu_info_reference=imu_info_reference,
                        imu_bias_jacobian=imu_bias_jacobian,
                        gyro_bias=gyro_bias,
                        accel_bias=core_model.imu_acc_bias,
                        gravity=args.imu_gravity,
                        max_residual=None,
                        pos_weight=pose_prior_pos_weight,
                        vel_weight=pose_prior_vel_weight,
                        rot_weight=args.imu_loss_rot_weight,
                        imu_compose_order=args.imu_full_compose_order,
                        imu_inverse=args.imu_full_inverse,
                        imu_optimize_bias=not args.use_pose_velocity_imu_ba,
                        augmented_mask=imu_confidence_augmented,
                        absolute_weight=(
                            args.imu_confidence_pose_prior_absolute_weight
                        ),
                        energy_log_reference=(
                            core_model.imu_confidence_energy_log_reference
                        ),
                        energy_reference_momentum=(
                            args.imu_confidence_pose_prior_ema_momentum
                        ),
                        energy_temperature=(
                            args.imu_confidence_pose_prior_temperature
                        ),
                        absolute_target_mode=(
                            args.imu_confidence_pose_prior_absolute_target_mode
                        ),
                        quality_only_confidence=quality_only_confidence,
                    )

                confidence_schedule_loss = torch.zeros((), device=images.device)
                if (
                    args.use_imu_ba
                    and args.w_imu_confidence_schedule != 0.0
                    and len(imu_loss_confidences) > 0
                ):
                    confidence_schedule_loss = imu_confidence_schedule_loss(
                        imu_loss_confidences,
                        args.imu_confidence_schedule_targets,
                        imu_valid=imu_valid,
                    )

                residual_weight_prior_loss = torch.zeros((), device=images.device)
                if args.use_learned_imu_residual_weights:
                    core_model = model.module if hasattr(model, 'module') else model
                    residual_weight_prior_loss = imu_residual_weight_prior_loss(core_model)

                ################################################
                # 14-6-6. total loss
                ################################################

                loss = (
                    args.w1 * geo_loss
                    + args.w2 * res_loss
                    + args.w3 * flo_loss
                    + args.w_imu * imu_loss
                    + args.w_imu_bias * bias_loss
                    + args.w_imu_confidence_saturation * confidence_reg_loss
                    + args.w_imu_confidence_nll * confidence_nll_loss
                    + args.w_imu_confidence_pose_prior * confidence_pose_prior_loss
                    + args.w_imu_confidence_schedule * confidence_schedule_loss
                    + args.w_imu_residual_weight_prior * residual_weight_prior_loss
                )

                if not args.eval_only:
                    loss.backward()

                ################################################
                # 14-6-7. random restart용 초기값 갱신
                ################################################

                # 마지막 iteration pose를 다음 restart의 초기 pose로 사용
                Gs = poses_est[-1].detach()

                # 마지막 disparity를 1/8 해상도 초기값으로 사용
                disp0 = disps_est[-1][:, :, 3::8, 3::8].detach()

            ####################################################
            # 14-7. metrics 정리
            ####################################################

            metrics = {}
            metrics.update(geo_metrics)
            metrics.update(res_metrics)
            metrics.update(flo_metrics)
            metrics.update(imu_metrics)
            metrics.update(confidence_nll_metrics)
            metrics.update(confidence_pose_prior_metrics)

            # Log weighted contributions, not just raw component values.  The
            # relative magnitudes determine whether the IMU branch has enough
            # training signal without manually inspecting the graph.
            metrics["loss_geo_weighted"] = (args.w1 * geo_loss).detach().item()
            metrics["loss_residual_weighted"] = (args.w2 * res_loss).detach().item()
            metrics["loss_flow_weighted"] = (args.w3 * flo_loss).detach().item()
            metrics["loss_imu_weighted"] = (args.w_imu * imu_loss).detach().item()
            metrics["loss_bias_weighted"] = (args.w_imu_bias * bias_loss).detach().item()
            metrics["loss_confidence_weighted"] = (
                args.w_imu_confidence_saturation * confidence_reg_loss
            ).detach().item()
            metrics["loss_confidence_nll_weighted"] = (
                args.w_imu_confidence_nll * confidence_nll_loss
            ).detach().item()
            metrics["loss_confidence_pose_prior_weighted"] = (
                args.w_imu_confidence_pose_prior * confidence_pose_prior_loss
            ).detach().item()
            metrics["loss_confidence_schedule_weighted"] = (
                args.w_imu_confidence_schedule * confidence_schedule_loss
            ).detach().item()
            metrics["loss_residual_weight_prior_weighted"] = (
                args.w_imu_residual_weight_prior * residual_weight_prior_loss
            ).detach().item()
            metrics["loss_total"] = loss.detach().item()
            metrics["imu_pose_only_unroll_steps"] = float(imu_train_pose_only_steps)
            metrics["imu_full_unroll_steps"] = float(
                args.iters - imu_train_pose_only_steps
                if args.use_full_imu_ba
                else 0
            )
            if imu_confidence_augmented is not None:
                temporal_augmented = imu_confidence_augmented[:, 1:]
                metrics["imu_confidence_augmented_fraction"] = float(
                    temporal_augmented.float().mean().detach().item()
                )
                metrics["imu_confidence_dropout_fraction"] = float(
                    imu_confidence_dropout[:, 1:].float().mean().detach().item()
                    if imu_confidence_dropout is not None
                    else 0.0
                )

            ####################################################
            # 14-8. IMU bias metric 기록
            ####################################################

            if use_imu_data:
                bias = model.module.imu_gyro_bias.detach()

                metrics["imu_bias_norm"] = bias.norm().item()
                metrics["imu_bias_x"] = bias[0].item()
                metrics["imu_bias_y"] = bias[1].item()
                metrics["imu_bias_z"] = bias[2].item()

                if use_acc_bias:
                    acc_bias = model.module.imu_acc_bias.detach()

                    metrics["imu_acc_bias_norm"] = acc_bias.norm().item()
                    metrics["imu_acc_bias_x"] = acc_bias[0].item()
                    metrics["imu_acc_bias_y"] = acc_bias[1].item()
                    metrics["imu_acc_bias_z"] = acc_bias[2].item()

            ####################################################
            # 14-9. IMU confidence metric 기록
            ####################################################

            if args.use_imu_ba and len(imu_confidences) > 0:
                conf = imu_confidences[-1].detach()
                # Column zero is a neutral placeholder because there is no
                # IMU factor before the first frame. Report only real,
                # valid temporal factors; otherwise a fixed sentinel of 1.0
                # misleadingly makes every run look saturated.
                active_conf = (
                    conf[:, 1:]
                    if conf.ndim in (2, 3) and conf.shape[1] > 1
                    else conf.reshape(1, -1)[:, :0]
                )
                if imu_valid is not None and active_conf.numel() > 0:
                    valid = imu_valid.detach().to(device=conf.device, dtype=torch.bool)
                    if valid.ndim == 1:
                        valid = valid.unsqueeze(0)
                    if valid.shape[0] == 1 and active_conf.shape[0] > 1:
                        valid = valid.expand(active_conf.shape[0], -1)
                    if valid.shape[0] == active_conf.shape[0] and valid.shape[1] >= 2:
                        active_conf = active_conf[
                            valid[:, 1:active_conf.shape[1] + 1]
                        ]
                    else:
                        active_conf = active_conf.reshape(-1)
                else:
                    active_conf = active_conf.reshape(-1)

                if active_conf.numel() > 0:
                    metrics["imu_conf_mean"] = active_conf.mean().item()
                    metrics["imu_conf_min"] = active_conf.min().item()
                    metrics["imu_conf_max"] = active_conf.max().item()
                    if active_conf.ndim == 2 and active_conf.shape[-1] == 3:
                        metrics["imu_conf_rot_mean"] = active_conf[:, 0].mean().item()
                        metrics["imu_conf_vel_mean"] = active_conf[:, 1].mean().item()
                        metrics["imu_conf_pos_mean"] = active_conf[:, 2].mean().item()
                else:
                    metrics["imu_conf_mean"] = 0.0
                    metrics["imu_conf_min"] = 0.0
                    metrics["imu_conf_max"] = 0.0
                metrics["imu_conf_active_edges"] = float(active_conf.numel())

                # Report the factor strength that actually entered DBA.  The
                # legacy checkpoint scalar is intentionally retained as a
                # separate diagnostic in confidence-only mode, where it must
                # not be mistaken for the effective IMU weight.
                checkpoint_ba_weight = (
                    model.module.get_imu_ba_weight().detach().item()
                )
                metrics["imu_ba_weight"] = (
                    1.0 if args.imu_confidence_only_weighting else checkpoint_ba_weight
                )
                if args.imu_confidence_only_weighting:
                    metrics["imu_ba_checkpoint_weight_ignored"] = checkpoint_ba_weight
                metrics["imu_conf_saturation_loss"] = confidence_reg_loss.detach().item()
                metrics["imu_confidence_nll"] = confidence_nll_loss.detach().item()
                metrics["imu_confidence_schedule_loss"] = (
                    confidence_schedule_loss.detach().item()
                )

                if args.use_learned_imu_residual_weights:
                    learned_weights = model.module.get_imu_full_residual_weights().detach()
                    metrics["imu_pos_weight"] = learned_weights[0].item()
                    metrics["imu_vel_weight"] = learned_weights[1].item()
                    metrics["imu_residual_weight_prior_loss"] = (
                        residual_weight_prior_loss.detach().item()
                    )

                if args.use_adaptive_imu_gate and len(imu_adaptive_gates) > 0:
                    gate = imu_adaptive_gates[-1].detach()
                    active_gate = gate[:, 1:] if gate.shape[1] > 1 else gate
                    metrics["imu_gate_mean"] = active_gate.mean().item()
                    metrics["imu_gate_min"] = active_gate.min().item()
                    metrics["imu_gate_max"] = active_gate.max().item()

            ####################################################
            # 14-10. optimizer update
            ####################################################

            if not args.eval_only:
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.clip
                )

                optimizer.step()

                # 학습 후 gyro/acc bias norm clamp
                clamp_parameter_norm(
                    model.module.imu_gyro_bias,
                    args.imu_gyro_bias_max_norm
                )

                clamp_parameter_norm(
                    model.module.imu_acc_bias,
                    args.imu_acc_bias_max_norm
                )

                scheduler.step()

            ####################################################
            # 14-11. step 증가 및 logging
            ####################################################

            total_steps += 1

            if gpu == 0:
                logger.push(metrics)

            ####################################################
            # 14-12. checkpoint 저장
            ####################################################

            if (
                not args.eval_only
                and args.save_freq > 0
                and total_steps % args.save_freq == 0
                and gpu == 0
            ):
                PATH = 'checkpoints/%s_%06d.pth' % (
                    args.name,
                    total_steps
                )

                save_checkpoint(model, PATH)
                print(f"[CKPT] saved {PATH}")
                if args.save_training_state:
                    train_state_path = 'checkpoints/%s_trainstate_%06d.pth' % (
                        args.name,
                        total_steps,
                    )
                    save_training_state(
                        model,
                        optimizer,
                        scheduler,
                        train_state_path,
                        total_steps,
                        sampler_epoch,
                        rng,
                    )
                    print(f"[CKPT] saved training state {train_state_path}")

            ####################################################
            # 14-13. 종료 조건
            ####################################################

            if total_steps >= args.stop_after_steps:
                should_keep_training = False
                break

    ############################################################
    # 15. final checkpoint 저장
    ############################################################

    if not args.eval_only and args.save_final and gpu == 0:
        PATH = 'checkpoints/%s_final_%06d.pth' % (
            args.name,
            total_steps
        )

        save_checkpoint(model, PATH)
        print(f"[CKPT] saved {PATH}")

    ############################################################
    # 16. DDP 종료
    ############################################################

    dist.destroy_process_group()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    ############################################################
    # 1. 기본 실험 설정
    ############################################################

    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--datasets', nargs='+', help='lists of datasets for training')
    parser.add_argument('--datapath', default='datasets/TartanAir', help="path to dataset directory")
    parser.add_argument('--gpus', type=int, default=4)

    ############################################################
    # 2. 기본 학습 hyperparameter
    ############################################################

    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--iters', type=int, default=15)
    parser.add_argument('--steps', type=int, default=250000)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--n_frames', type=int, default=7)
    parser.add_argument(
        '--imu_train_frame_strides',
        type=int,
        nargs='+',
        default=[1],
        help=(
            'raw camera-frame gaps sampled within an IMU training clip; '
            'for TartanAir V2/EuRoC, e.g. 1 2 4 8 exposes composed IMU intervals'
        ),
    )
    parser.add_argument(
        '--euroc_camera_frame_stride',
        type=int,
        default=1,
        help=(
            'retain every Nth raw EuRoC camera frame before forming clips; '
            'use 2 to train on EuRoC at 10 Hz while preserving all IMU samples'
        ),
    )
    parser.add_argument(
        '--euroc_keyframe_manifest_root',
        default=None,
        help=(
            'directory of <EuRoC-sequence>.csv files exported by the actual '
            'DROID MotionFilter; ordered clips use consecutive selected '
            'keyframes while composing every raw IMU sample between them'
        ),
    )
    parser.add_argument(
        '--imu_train_gyro_noise_density',
        type=float,
        default=None,
        help=(
            'optional gyro white-noise density override for ordered IMU readers; '
            'EuRoC defaults to imu0/sensor.yaml calibration'
        ),
    )
    parser.add_argument(
        '--imu_train_accel_noise_density',
        type=float,
        default=None,
        help=(
            'optional accelerometer white-noise density override for ordered IMU readers; '
            'EuRoC defaults to imu0/sensor.yaml calibration'
        ),
    )
    parser.add_argument(
        '--euroc_sequence_include',
        nargs='+',
        default=None,
        metavar='PATTERN',
        help=(
            'raw EuRoC sequence-name/path glob(s) permitted for training, '
            'for example V1_01_easy V1_02_medium; prevents held-out leakage'
        ),
    )
    parser.add_argument(
        '--euroc_sequence_exclude',
        nargs='+',
        default=None,
        metavar='PATTERN',
        help='raw EuRoC sequence-name/path glob(s) excluded from training',
    )

    ############################################################
    # 3. 기존 DROID loss weight
    ############################################################

    # pose geodesic loss weight
    parser.add_argument('--w1', type=float, default=10.0)

    # residual loss weight
    parser.add_argument('--w2', type=float, default=0.01)

    # flow loss weight
    parser.add_argument('--w3', type=float, default=0.05)

    ############################################################
    # 4. 추가된 IMU loss weight
    ############################################################

    # IMU preintegration loss weight
    parser.add_argument('--w_imu', type=float, default=0.05)

    # gyro/acc bias regularization weight
    parser.add_argument('--w_imu_bias', type=float, default=1e-4)

    ############################################################
    # 5. dataset sampling / graph 설정
    ############################################################

    parser.add_argument('--fmin', type=float, default=8.0)
    parser.add_argument('--fmax', type=float, default=96.0)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--edges', type=int, default=24)
    parser.add_argument('--restart_prob', type=float, default=0.2)

    ############################################################
    # 6. freeze / logging / checkpoint 옵션
    ############################################################

    parser.add_argument(
        '--freeze_non_imu',
        action='store_true',
        help=(
            'freeze pretrained DROID visual/update weights and train only '
            'IMU confidence, IMU BA scale, and IMU biases'
        ),
    )
    parser.add_argument(
        '--freeze_global_imu_bias',
        action='store_true',
        help=(
            'keep checkpoint global gyro/accelerometer bias fixed during '
            'fine-tuning; local 15DoF DBA bias states remain unchanged'
        ),
    )
    parser.add_argument(
        '--balance_sequence_sampling',
        action='store_true',
        help=(
            'sample ordered multi-sequence datasets with equal expected mass '
            'per sequence instead of proportional-to-sequence-length clips'
        ),
    )

    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--save_final', action='store_true')
    parser.add_argument(
        '--save_training_state',
        action='store_true',
        help=(
            'save optimizer/scheduler/RNG continuation files at ordinary '
            'checkpoint boundaries; model .pth files remain evaluation-compatible'
        ),
    )
    parser.add_argument(
        '--resume_training_state',
        default=None,
        help=(
            'resume model, optimizer, scheduler, RNG, and step count from a '
            'training-state file produced by --save_training_state'
        ),
    )
    parser.add_argument(
        '--stop_after_steps',
        type=int,
        default=None,
        help=(
            'exit cleanly once this absolute training step is reached; use '
            'with --resume_training_state for externally validated segments'
        ),
    )
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--disable_flow_loss', action='store_true')

    ############################################################
    # 7. IMU BA 관련 옵션
    ############################################################

    # IMU residual을 BA 단계에 넣을지 여부
    parser.add_argument('--use_imu_ba', action='store_true')

    parser.add_argument(
        '--use_full_imu_ba',
        action='store_true',
        help='extend training BA with pose+velocity+accel-bias+gyro-bias IMU residuals',
    )
    parser.add_argument(
        '--imu_train_pose_only_steps',
        type=int,
        default=0,
        help=(
            'number of visual pose/depth-only DBA unrolls before full 15DoF '
            'IMU DBA; zero preserves the legacy full-IMU-from-step-zero path'
        ),
    )
    parser.add_argument(
        '--use_pose_velocity_imu_ba',
        action='store_true',
        help=(
            'use pose+velocity IMU BA/loss with fixed global gravity and IMU biases; '
            'local accel/gyro bias states are not optimized'
        ),
    )

    # IMU BA residual의 global scale 초기값
    parser.add_argument(
        '--imu_ba_weight',
        type=float,
        default=None,
        help=(
            'explicitly reset the learnable global IMU BA information scale; '
            'omitting it preserves the value from an IMU checkpoint'
        ),
    )
    parser.add_argument(
        '--freeze_imu_ba_weight',
        action='store_true',
        help=(
            'keep the checkpoint global IMU BA scale fixed; useful when the '
            'only trainable stage-3 parameter is a confidence head'
        ),
    )
    parser.add_argument(
        '--freeze_imu_confidence_head',
        action='store_true',
        help=(
            'keep the active IMU-confidence head fixed while learning the '
            'global IMU BA scale; use for a scale-only calibration run'
        ),
    )
    parser.add_argument(
        '--imu_confidence_only_weighting',
        action='store_true',
        help=(
            'fix the global IMU BA factor weight to 1.0 and let the learned '
            'temporal IMU-confidence head be the only learned attenuation; '
            'the legacy imu_ba_log_weight remains in checkpoints but is ignored'
        ),
    )

    # learned IMU confidence에 최소값을 섞어줄 때 사용
    parser.add_argument(
        '--imu_conf_floor',
        type=float,
        default=0.0,
        help='minimum value mixed into learned GRU IMU confidence',
    )
    parser.add_argument(
        '--use_component_imu_confidence',
        action='store_true',
        help=(
            'predict separate GRU confidence for IMU rotation, velocity, and '
            'position residuals and fuse preintegration quality features'
        ),
    )
    parser.add_argument(
        '--use_dvi_imu_confidence',
        action='store_true',
        help=(
            'train the DVI-style scalar confidence head: 1x1 GRU projection '
            'followed by spatial average pooling'
        ),
    )
    parser.add_argument(
        '--use_dvi_imu_quality_fusion',
        action='store_true',
        help=(
            'add direct preintegrated-IMU quality features to the DVI scalar '
            'confidence logit; keeps one scalar confidence per temporal factor'
        ),
    )
    parser.add_argument(
        '--w_imu_confidence_saturation',
        type=float,
        default=0.0,
        help='optional upper-tail regularizer for saturated learned IMU confidence',
    )
    parser.add_argument(
        '--w_imu_confidence_nll',
        type=float,
        default=0.0,
        help=(
            'weight of temporal IMU-confidence precision NLL; this calibrates '
            'the GRU confidence per consecutive IMU factor'
        ),
    )
    parser.add_argument(
        '--w_imu_confidence_pose_prior',
        type=float,
        default=0.0,
        help=(
            'pre-BA visual-pose/IMU confidence supervision; unlike '
            '--w_imu_confidence_nll this never uses a full-IMU-BA pose'
        ),
    )
    parser.add_argument(
        '--imu_confidence_pose_prior_absolute_weight',
        type=float,
        default=0.0,
        help=(
            'relative weight of data-derived absolute pre-BA confidence '
            'calibration inside the pose-prior supervisor; zero preserves '
            'ranking-only behavior'
        ),
    )
    parser.add_argument(
        '--imu_confidence_pose_prior_ema_momentum',
        type=float,
        default=0.98,
        help=(
            'EMA momentum for the clean visual/IMU log-energy reference; '
            'this is residual-unit normalization, not an IMU BA weight'
        ),
    )
    parser.add_argument(
        '--imu_confidence_pose_prior_temperature',
        type=float,
        default=0.75,
        help='softness of the data-derived absolute confidence target',
    )
    parser.add_argument(
        '--imu_confidence_pose_prior_absolute_target_mode',
        choices=(
            'median_residual',
            'nominal_clean',
            'split_nominal_quality',
            'one_sided_nominal_quality',
        ),
        default='median_residual',
        help=(
            "absolute confidence convention: 'median_residual' centers clean "
            "visual/IMU disagreement near 0.5; 'nominal_clean' treats the "
            "provided clean IMU covariance as full information and supervises "
            "only augmented intervals from residual energy; "
            "'split_nominal_quality' applies that nominal convention only to "
            "the raw-IMU quality path while the fused DVI confidence remains "
            "residual-calibrated; 'one_sided_nominal_quality' preserves full "
            "confidence in the normal clean residual range and attenuates "
            "only its learned high-residual tail"
        ),
    )
    parser.add_argument(
        '--use_imu_confidence_augmentation',
        action='store_true',
        help=(
            'inject train-only preintegration-equivalent gyro/accel noise, '
            'bias drift, and short partial IMU dropout for confidence learning'
        ),
    )
    parser.add_argument(
        '--imu_confidence_aug_probability',
        type=float,
        default=0.35,
        help='per-valid-interval probability of synthetic IMU noise corruption',
    )
    parser.add_argument(
        '--imu_confidence_aug_gyro_noise_std',
        type=float,
        default=0.03,
        help='synthetic raw-gyro white-noise density for confidence augmentation',
    )
    parser.add_argument(
        '--imu_confidence_aug_accel_noise_std',
        type=float,
        default=0.30,
        help='synthetic raw-accelerometer white-noise density for confidence augmentation',
    )
    parser.add_argument(
        '--imu_confidence_aug_gyro_bias_drift_std',
        type=float,
        default=0.01,
        help='synthetic gyro-bias random-walk density for confidence augmentation',
    )
    parser.add_argument(
        '--imu_confidence_aug_accel_bias_drift_std',
        type=float,
        default=0.10,
        help='synthetic accelerometer-bias random-walk density for confidence augmentation',
    )
    parser.add_argument(
        '--imu_confidence_aug_dropout_probability',
        type=float,
        default=0.35,
        help='per-clip probability of one contiguous partial-IMU dropout',
    )
    parser.add_argument(
        '--imu_confidence_aug_dropout_max_length',
        type=int,
        default=2,
        help='maximum selected camera intervals in a synthetic dropout run',
    )
    parser.add_argument(
        '--imu_confidence_aug_dropout_info_scale',
        type=float,
        default=0.05,
        help='remaining propagated information for a synthetic dropout interval',
    )
    parser.add_argument(
        '--w_imu_confidence_schedule',
        type=float,
        default=0.0,
        help=(
            'weight of an optional weak mean-confidence curriculum applied '
            'to successive full-IMU DBA unrolls'
        ),
    )
    parser.add_argument(
        '--imu_confidence_schedule_targets',
        type=float,
        nargs='+',
        default=None,
        metavar='C',
        help=(
            'manual confidence targets in [0,1] for full-IMU DBA unrolls; '
            'for example: 0.25 0.50 0.75 1.00'
        ),
    )
    parser.add_argument(
        '--imu_confidence_saturation_ceiling',
        type=float,
        default=0.95,
        help='confidence value where the optional saturation regularizer starts',
    )
    parser.add_argument(
        '--reset_imu_confidence_head',
        action='store_true',
        help='reset only the final GRU IMU-confidence projection before fine-tuning',
    )
    parser.add_argument(
        '--imu_confidence_reset_value',
        type=float,
        default=0.5,
        help=(
            'sigmoid confidence used with --reset_imu_confidence_head; '
            'use about 0.95 with --imu_confidence_only_weighting so IMU '
            'starts near full strength'
        ),
    )
    parser.add_argument(
        '--use_adaptive_imu_gate',
        action='store_true',
        help='apply the runtime visual/motion/residual IMU gate inside training DBA',
    )
    parser.add_argument(
        '--use_learned_imu_residual_weights',
        action='store_true',
        help='learn checkpointed relative position/velocity IMU weights while rotation anchors factor scale',
    )
    parser.add_argument(
        '--reset_imu_residual_weights',
        action='store_true',
        help='initialize learned relative position/velocity IMU weights to the fixed (0.05, 0.05) reference',
    )
    parser.add_argument(
        '--w_imu_residual_weight_prior',
        type=float,
        default=0.001,
        help='log-space prior strength for learned relative IMU position/velocity weights',
    )
    parser.add_argument('--imu_gate_min', type=float, default=0.2)
    parser.add_argument('--imu_gate_max', type=float, default=1.25)
    parser.add_argument('--imu_gate_visual_thresh', type=float, default=0.2)
    parser.add_argument('--imu_gate_visual_boost', type=float, default=0.25)
    parser.add_argument('--imu_gate_motion_thresh_deg', type=float, default=5.0)
    parser.add_argument('--imu_gate_motion_boost', type=float, default=0.15)
    parser.add_argument('--imu_gate_residual_soft_deg', type=float, default=2.5)
    parser.add_argument('--imu_gate_residual_hard_deg', type=float, default=8.0)
    parser.add_argument('--imu_gate_residual_pos_soft', type=float, default=0.0)
    parser.add_argument('--imu_gate_residual_vel_soft', type=float, default=0.0)

    ############################################################
    # 8. IMU loss 관련 옵션
    ############################################################

    # IMU loss를 학습 loss에 추가할지 여부
    parser.add_argument('--use_imu_loss', action='store_true')

    # dataset 안에서 IMU prior 파일 이름
    parser.add_argument('--imu_prior_name', type=str, default='imu_prior.csv')

    # IMU prior가 없는 sample을 허용하지 않을지 여부
    parser.add_argument('--imu_require', action='store_true')

    # iteration별 loss weight decay/gamma
    parser.add_argument('--imu_gamma', type=float, default=0.9)

    # IMU residual clipping 값
    parser.add_argument('--imu_max_residual', type=float, default=0.5)

    # smooth L1 loss beta
    parser.add_argument('--imu_smooth_beta', type=float, default=0.05)

    ############################################################
    # 9. IMU bias 초기값
    ############################################################

    # gyro bias 초기값
    parser.add_argument('--imu_bias_init', type=float, nargs=3, default=None)

    # accelerometer bias 초기값
    parser.add_argument('--imu_acc_bias_init', type=float, nargs=3, default=None)

    ############################################################
    # 10. full IMU BA 내부 weight
    ############################################################

    parser.add_argument('--imu_full_pos_weight', type=float, default=0.05)
    parser.add_argument('--imu_full_vel_weight', type=float, default=0.05)
    parser.add_argument('--imu_full_bias_weight', type=float, default=0.001)
    parser.add_argument(
        '--imu_full_compose_order',
        choices=['prev_dq', 'dq_prev'],
        default='dq_prev',
        help='quaternion compose order for full IMU BA/loss',
    )
    parser.add_argument(
        '--imu_full_inverse',
        dest='imu_full_inverse',
        action='store_true',
        default=True,
        help='invert the preintegrated IMU rotation before full IMU BA/loss',
    )
    parser.add_argument(
        '--no_imu_full_inverse',
        dest='imu_full_inverse',
        action='store_false',
        help='do not invert the preintegrated IMU rotation before full IMU BA/loss',
    )
    parser.add_argument(
        '--use_imu_info_weighting',
        action='store_true',
        help='weight full IMU residuals by median-normalized preintegration information',
    )
    parser.add_argument('--imu_info_weight_clip', type=float, default=4.0)
    parser.add_argument('--imu_info_weight_eps', type=float, default=1e-12)
    parser.add_argument(
        '--use_imu_noise_model',
        action='store_true',
        help='use unclipped covariance-relative IMU information with learned GRU confidence',
    )
    parser.add_argument(
        '--imu_gravity',
        type=float,
        nargs=3,
        default=None,
        metavar=('GX', 'GY', 'GZ'),
        help='optional gravity vector in training pose units for full IMU residuals',
    )
    parser.add_argument(
        '--allow_zero_imu_gravity',
        action='store_true',
        help='debug only: allow full IMU training residuals to use zero gravity',
    )

    # velocity 초기화 방식
    #
    # zero:
    #   velocity를 0으로 초기화
    #
    # pose:
    #   pose 차분으로 velocity를 초기화
    parser.add_argument('--imu_velocity_init', choices=['zero', 'pose'], default='pose')

    # IMU motion state에 대한 prior weight
    parser.add_argument('--imu_motion_prior_weight', type=float, default=0.0)

    # local bias smoothness/prior weight
    parser.add_argument('--imu_local_bias_prior_weight', type=float, default=0.0)

    ############################################################
    # 11. full IMU loss 관련 옵션
    ############################################################

    parser.add_argument(
        '--use_full_imu_loss',
        action='store_true',
        help='train IMU loss with position, velocity, rotation, and bias preintegration residuals',
    )

    parser.add_argument('--imu_loss_pos_weight', type=float, default=0.05)
    parser.add_argument('--imu_loss_vel_weight', type=float, default=0.05)
    parser.add_argument('--imu_loss_rot_weight', type=float, default=1.0)
    parser.add_argument('--imu_loss_bias_weight', type=float, default=0.001)

    ############################################################
    # 12. IMU bias norm clamp
    ############################################################

    parser.add_argument('--imu_gyro_bias_max_norm', type=float, default=0.0)
    parser.add_argument('--imu_acc_bias_max_norm', type=float, default=0.0)

    ############################################################
    # 13. argument parsing 및 옵션 보정
    ############################################################

    args = parser.parse_args()

    if args.stop_after_steps is None:
        args.stop_after_steps = args.steps
    if args.steps <= 0:
        raise ValueError('--steps must be positive')
    if args.stop_after_steps <= 0 or args.stop_after_steps > args.steps:
        raise ValueError(
            '--stop_after_steps must be positive and no greater than --steps'
        )
    if (
        args.resume_training_state is not None
        and not os.path.isfile(args.resume_training_state)
    ):
        raise FileNotFoundError(
            f'--resume_training_state does not exist: {args.resume_training_state}'
        )
    if args.eval_only and (
        args.resume_training_state is not None or args.save_training_state
    ):
        raise ValueError(
            '--resume_training_state / --save_training_state are training-only options'
        )

    # full IMU loss를 켜면 일반 IMU loss flag도 자동으로 켬
    if args.use_full_imu_loss:
        args.use_imu_loss = True

    # full IMU BA를 켜면 일반 IMU BA flag도 자동으로 켬
    if args.use_full_imu_ba:
        args.use_imu_ba = True

    if args.use_pose_velocity_imu_ba:
        args.use_full_imu_ba = True
        args.use_imu_ba = True

    if args.iters <= 0:
        raise ValueError('--iters must be positive')
    if args.imu_train_pose_only_steps < 0:
        raise ValueError('--imu_train_pose_only_steps must be non-negative')
    if args.imu_train_pose_only_steps > 0 and not args.use_full_imu_ba:
        raise ValueError(
            '--imu_train_pose_only_steps requires --use_full_imu_ba'
        )
    if (
        args.use_full_imu_ba
        and args.imu_train_pose_only_steps >= args.iters
    ):
        raise ValueError(
            '--imu_train_pose_only_steps must be smaller than --iters so at '
            'least one full 15DoF IMU DBA step remains'
        )

    if args.use_learned_imu_residual_weights and not args.use_full_imu_ba:
        raise ValueError('--use_learned_imu_residual_weights requires --use_full_imu_ba')
    if args.reset_imu_residual_weights and not args.use_learned_imu_residual_weights:
        raise ValueError(
            '--reset_imu_residual_weights requires '
            '--use_learned_imu_residual_weights'
        )
    if args.reset_imu_residual_weights and args.resume_training_state is not None:
        raise ValueError(
            '--reset_imu_residual_weights is incompatible with '
            '--resume_training_state; a resumed run must retain its learned '
            'relative weights'
        )
    if args.use_component_imu_confidence and not args.use_full_imu_ba:
        raise ValueError('--use_component_imu_confidence requires --use_full_imu_ba')
    if args.use_dvi_imu_confidence and not args.use_full_imu_ba:
        raise ValueError('--use_dvi_imu_confidence requires --use_full_imu_ba')
    if args.use_dvi_imu_confidence and args.use_component_imu_confidence:
        raise ValueError('choose only one of --use_dvi_imu_confidence and --use_component_imu_confidence')
    if args.use_dvi_imu_quality_fusion and not args.use_dvi_imu_confidence:
        raise ValueError(
            '--use_dvi_imu_quality_fusion requires --use_dvi_imu_confidence'
        )
    if args.imu_confidence_only_weighting:
        if not args.use_full_imu_ba:
            raise ValueError(
                '--imu_confidence_only_weighting requires --use_full_imu_ba'
            )
        if not (args.use_dvi_imu_confidence or args.use_component_imu_confidence):
            raise ValueError(
                '--imu_confidence_only_weighting requires a learned IMU '
                'confidence head (--use_dvi_imu_confidence or '
                '--use_component_imu_confidence)'
            )
        if args.imu_ba_weight is not None:
            raise ValueError(
                '--imu_ba_weight is incompatible with '
                '--imu_confidence_only_weighting; the global factor weight '
                'is fixed to 1.0'
            )
        if args.freeze_imu_confidence_head:
            raise ValueError(
                '--freeze_imu_confidence_head is incompatible with '
                '--imu_confidence_only_weighting'
            )
    if args.use_imu_noise_model and not (args.use_full_imu_ba or args.use_full_imu_loss):
        raise ValueError('--use_imu_noise_model requires full IMU BA or full IMU loss')
    if args.w_imu_residual_weight_prior < 0.0:
        raise ValueError('--w_imu_residual_weight_prior must be non-negative')
    if any(value <= 0 for value in args.imu_train_frame_strides):
        raise ValueError('--imu_train_frame_strides must contain positive integers')
    args.imu_train_frame_strides = sorted(set(args.imu_train_frame_strides))
    if args.euroc_camera_frame_stride <= 0:
        raise ValueError('--euroc_camera_frame_stride must be positive')
    for option in (
        'imu_train_gyro_noise_density',
        'imu_train_accel_noise_density',
    ):
        value = getattr(args, option)
        if value is not None and value < 0.0:
            raise ValueError(f'--{option} must be non-negative')
    configure_imu_noise_model(args, source='train')

    if args.use_adaptive_imu_gate and not args.use_imu_ba:
        raise ValueError('--use_adaptive_imu_gate requires --use_imu_ba')
    if args.reset_imu_confidence_head and not args.use_imu_ba:
        raise ValueError('--reset_imu_confidence_head requires --use_imu_ba')

    if args.w_imu_confidence_saturation < 0.0:
        raise ValueError('--w_imu_confidence_saturation must be non-negative')
    if args.w_imu_confidence_nll < 0.0:
        raise ValueError('--w_imu_confidence_nll must be non-negative')
    if args.w_imu_confidence_pose_prior < 0.0:
        raise ValueError('--w_imu_confidence_pose_prior must be non-negative')
    if args.imu_confidence_pose_prior_absolute_weight < 0.0:
        raise ValueError(
            '--imu_confidence_pose_prior_absolute_weight must be non-negative'
        )
    if not 0.0 <= args.imu_confidence_pose_prior_ema_momentum < 1.0:
        raise ValueError(
            '--imu_confidence_pose_prior_ema_momentum must lie in [0, 1)'
        )
    if args.imu_confidence_pose_prior_temperature <= 0.0:
        raise ValueError(
            '--imu_confidence_pose_prior_temperature must be positive'
        )
    if args.w_imu_confidence_schedule < 0.0:
        raise ValueError('--w_imu_confidence_schedule must be non-negative')
    if args.w_imu_confidence_schedule > 0.0:
        if not args.imu_confidence_schedule_targets:
            raise ValueError(
                '--w_imu_confidence_schedule requires '
                '--imu_confidence_schedule_targets'
            )
        if any(
            value < 0.0 or value > 1.0
            for value in args.imu_confidence_schedule_targets
        ):
            raise ValueError(
                '--imu_confidence_schedule_targets values must lie in [0,1]'
            )
    if args.w_imu_confidence_pose_prior > 0.0:
        if not args.use_full_imu_ba:
            raise ValueError(
                '--w_imu_confidence_pose_prior requires --use_full_imu_ba'
            )
        if args.imu_train_pose_only_steps <= 0:
            raise ValueError(
                '--w_imu_confidence_pose_prior requires at least one '
                'visual-only unroll via --imu_train_pose_only_steps'
            )
        if not args.use_dvi_imu_confidence:
            raise ValueError(
                '--w_imu_confidence_pose_prior currently supervises the '
                'single DVI scalar confidence; add --use_dvi_imu_confidence'
            )
        if args.w_imu_confidence_nll != 0.0:
            raise ValueError(
                'choose exactly one confidence supervisor: pose-prior ranking '
                'uses pre-BA visual poses, so --w_imu_confidence_nll must be 0'
            )
    elif args.imu_confidence_pose_prior_absolute_weight != 0.0:
        raise ValueError(
            '--imu_confidence_pose_prior_absolute_weight requires '
            '--w_imu_confidence_pose_prior > 0'
        )
    if args.use_imu_confidence_augmentation:
        if args.w_imu_confidence_pose_prior <= 0.0:
            raise ValueError(
                '--use_imu_confidence_augmentation requires '
                '--w_imu_confidence_pose_prior so degradation has the '
                'intended pre-BA confidence supervision'
            )
        if not args.use_imu_ba:
            raise ValueError('--use_imu_confidence_augmentation requires --use_imu_ba')
        probability_options = (
            'imu_confidence_aug_probability',
            'imu_confidence_aug_dropout_probability',
        )
        for option in probability_options:
            value = float(getattr(args, option))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'--{option} must lie in [0, 1]')
        nonnegative_options = (
            'imu_confidence_aug_gyro_noise_std',
            'imu_confidence_aug_accel_noise_std',
            'imu_confidence_aug_gyro_bias_drift_std',
            'imu_confidence_aug_accel_bias_drift_std',
        )
        for option in nonnegative_options:
            if float(getattr(args, option)) < 0.0:
                raise ValueError(f'--{option} must be non-negative')
        if args.imu_confidence_aug_dropout_max_length <= 0:
            raise ValueError('--imu_confidence_aug_dropout_max_length must be positive')
        if not 0.0 < args.imu_confidence_aug_dropout_info_scale <= 1.0:
            raise ValueError(
                '--imu_confidence_aug_dropout_info_scale must lie in (0, 1]'
            )
    if args.imu_confidence_only_weighting:
        if args.w_imu_confidence_saturation != 0.0:
            raise ValueError(
                '--imu_confidence_only_weighting must not use '
                '--w_imu_confidence_saturation: confidence=1 is the full-IMU '
                'reference, not a saturation error'
            )
        if args.w_imu_confidence_schedule != 0.0:
            raise ValueError(
                '--imu_confidence_only_weighting must not use a manual '
                'mean-confidence schedule'
            )
        if (
            args.w_imu_confidence_nll <= 0.0
            and args.w_imu_confidence_pose_prior <= 0.0
        ):
            raise ValueError(
                '--imu_confidence_only_weighting requires a positive '
                'confidence supervisor (--w_imu_confidence_nll or '
                '--w_imu_confidence_pose_prior) to prevent collapse'
            )
        if args.imu_conf_floor != 0.0:
            raise ValueError(
                '--imu_confidence_only_weighting requires --imu_conf_floor 0; '
                'the predicted confidence must be used directly'
            )
    if not 0.0 < args.imu_confidence_saturation_ceiling < 1.0:
        raise ValueError('--imu_confidence_saturation_ceiling must be in (0, 1)')
    if not 0.0 < args.imu_confidence_reset_value < 1.0:
        raise ValueError('--imu_confidence_reset_value must be in (0, 1)')

    if (
        (args.use_full_imu_ba or args.use_full_imu_loss)
        and args.imu_gravity is None
        and not args.allow_zero_imu_gravity
    ):
        raise ValueError(
            "Full IMU BA/loss requires gravity for position/velocity residuals. "
            "Provide --imu_gravity GX GY GZ, or pass --allow_zero_imu_gravity "
            "for a debug-only structure check."
        )

    # DDP 전체 process 수
    args.world_size = args.gpus

    print(args)

    ############################################################
    # 14. checkpoint directory 생성
    ############################################################

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    ############################################################
    # 15. DDP 환경 변수 설정
    ############################################################

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    ############################################################
    # 16. multi-GPU training 시작
    ############################################################

    mp.spawn(
        train,
        nprocs=args.gpus,
        args=(args,)
    )

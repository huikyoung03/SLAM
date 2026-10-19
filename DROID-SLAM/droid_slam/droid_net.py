"""
DROID-SLAM + Visual-Inertial 확장 네트워크

이 파일은 기존 DROID-SLAM의 recurrent visual update 구조에 다음 기능을 추가한 버전이다.

1. IMU preintegration(imu_delta)을 BA에 전달
2. Full IMU BA를 위한 velocity 기반 motion state 초기화/유지
3. 학습 가능한 gyro/accelerometer bias
4. 학습 가능한 IMU BA 및 position/velocity residual weight
5. ConvGRU hidden state 기반 IMU confidence head
6. visual/IMU 상태에 따른 adaptive IMU gate
7. BA에서 pose, disparity와 IMU motion을 반복적으로 갱신

전체 흐름:
Image
 -> FNet / CNet
 -> Correlation
 -> ConvGRU
 -> delta + visual confidence + IMU confidence
 -> Visual-Inertial BA
 -> Pose + Disparity (+ IMU motion) 갱신

주의:
IMU raw sample의 preintegration 자체는 이 파일에서 수행하지 않는다.
이 파일은 이미 계산된 imu_delta 및 관련 covariance/info/Jacobian을 받아
DROID의 update/BA 과정과 결합하는 역할을 담당한다.
"""

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from collections import OrderedDict



from modules.extractor import BasicEncoder

from modules.corr import CorrBlock

from modules.gru import ConvGRU

from modules.clipping import GradientClip



from lietorch import SE3

from geom.ba import BA

from geom.imu_factor import (

    compute_imu_adaptive_gate,

    edge_imu_quality_features,

    imu_center_from_pose_data,

    temporal_imu_confidence_from_edges,

    temporal_imu_quality_features,

)



import geom.projective_ops as pops

from geom.graph_utils import graph_to_edge_list, keyframe_indicies



from torch_scatter import scatter_mean





# =============================================================================
# [IMU motion state 초기화]
# Full Visual-Inertial BA에서 사용할 프레임별 IMU motion state를 초기화한다.
#
# motion shape : [batch, num_frames, 9]
#   motion[..., 0:3] : velocity (vx, vy, vz)
#   나머지 6차원은 현재 BA 구현에서 사용하는 local IMU motion state 영역이다.
#
# mode == "pose"인 경우:
#   IMU의 dt와 현재 visual pose에서 얻은 카메라 중심(center)을 이용해
#       v_i ~= (p_{i+1} - p_i) / dt
#   로 초기 속도를 계산한다.
#
# 중요한 점:
#   이것은 최종 IMU 최적화 결과가 아니라 BA가 시작할 때 필요한 initial guess이다.
# =============================================================================

def initialize_imu_motion(
    poses,

    imu_delta=None,

    mode="pose",

    imu_camera_translation=None,

):

    pose_data = poses.data

    batch, num_frames = pose_data.shape[:2]

    motion = pose_data.new_zeros((batch, num_frames, 9))



    if mode != "pose" or imu_delta is None or num_frames < 2:

        return motion



    imu = imu_delta.to(device=pose_data.device, dtype=pose_data.dtype)

    if imu.ndim == 2:

        imu = imu.unsqueeze(0)

    if imu.shape[0] == 1 and batch > 1:

        imu = imu.expand(batch, -1, -1)

    if imu.shape[0] != batch or imu.shape[1] < num_frames:

        return motion



    centers = imu_center_from_pose_data(pose_data, imu_camera_translation)

    dt = imu[:, 1:num_frames, 0:1].clamp_min(1e-4)

    pair_velocity = (centers[:, 1:num_frames] - centers[:, :num_frames - 1]) / dt

    motion[:, :num_frames - 1, 0:3] = pair_velocity

    motion[:, num_frames - 1, 0:3] = pair_velocity[:, -1]



    return motion


def full_imu_unroll_schedule(num_steps, use_full_imu_ba, pose_only_steps=0):
    """Return whether each unrolled DBA step uses the full IMU state.

    A two-stage local solve first gives visual DBA iterations to establish a
    pose/depth estimate. Only then are velocity and local bias states added.
    ``pose_only_steps=0`` preserves the previous full-IMU behavior.
    """

    num_steps = int(num_steps)
    pose_only_steps = int(pose_only_steps)
    if num_steps < 0:
        raise ValueError(f"num_steps must be non-negative, got {num_steps}")
    if pose_only_steps < 0:
        raise ValueError(
            f"pose_only_steps must be non-negative, got {pose_only_steps}"
        )
    if not use_full_imu_ba:
        return [False] * num_steps

    return [step >= pose_only_steps for step in range(num_steps)]


@torch.no_grad()

def temporal_visual_confidence(weight, ii, jj, num_frames):

    """Match the runtime visual-quality lookup for consecutive IMU factors.



    For each temporal interval \`(j - 1, j)\`, use only direct visual DBA edges.

    A missing direct edge has neutral confidence, matching the runtime factor

    path and avoiding leakage from unrelated loop/proximity edges.

    """



    if weight is None:

        return None



    # Visual DBA has two per-pixel channels, unlike IMU component confidence.
    # Reduce them first so the generic temporal helper keeps a scalar channel.
    visual_weight = weight.detach().float().mean(dim=-1, keepdim=True)

    return temporal_imu_confidence_from_edges(

        visual_weight,

        ii,

        jj,

        num_frames=int(num_frames),

        fallback=1.0,

    )





# =============================================================================
# [Convex upsampling]
# DROID/RAFT 계열에서 사용하는 learned convex upsampling.
# 저해상도(1/8) disparity 등의 값을 주변 3x3 값의 학습된 convex combination으로
# 원래 해상도의 8배 크기로 복원한다.
# =============================================================================
def cvx_upsample(data, mask):

    """ upsample pixel-wise transformation field """

    batch, ht, wd, dim = data.shape

    data = data.permute(0, 3, 1, 2)

    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)

    mask = torch.softmax(mask, dim=2)



    up_data = F.unfold(data, [3,3], padding=1)

    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)



    up_data = torch.sum(mask * up_data, dim=2)

    up_data = up_data.permute(0, 4, 2, 5, 3, 1)

    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)



    return up_data



def upsample_disp(disp, mask):

    batch, num, ht, wd = disp.shape

    disp = disp.view(batch*num, ht, wd, 1)

    mask = mask.view(batch*num, -1, ht, wd)

    return cvx_upsample(disp, mask).view(batch, num, 8*ht, 8*wd)





# =============================================================================
# [Graph Aggregation]
# 동일한 source frame(ii)을 공유하는 여러 graph edge의 hidden feature를
# scatter_mean으로 통합한다.
#
# 출력:
#   eta    : DBA에서 사용하는 damping 계열 값
#   upmask : disparity를 full resolution으로 복원하기 위한 upsampling mask
# =============================================================================
class GraphAgg(nn.Module):

    def __init__(self):

        super(GraphAgg, self).__init__()

        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)



        self.eta = nn.Sequential(

            nn.Conv2d(128, 1, 3, padding=1),

            GradientClip(),

            nn.Softplus())



        self.upmask = nn.Sequential(

            nn.Conv2d(128, 8*8*9, 1, padding=0))



    def forward(self, net, ii):

        batch, num, ch, ht, wd = net.shape

        net = net.view(batch*num, ch, ht, wd)



        _, ix = torch.unique(ii, return_inverse=True)

        net = self.relu(self.conv1(net))



        net = net.view(batch, num, 128, ht, wd)

        net = scatter_mean(net, ix, dim=1)

        net = net.view(-1, 128, ht, wd)



        net = self.relu(self.conv2(net))



        eta = self.eta(net).view(batch, -1, ht, wd)

        upmask = self.upmask(net).view(batch, -1, 8*8*9, ht, wd)



        return .01 * eta, upmask





# =============================================================================
# [DROID recurrent update operator + IMU confidence head]
#
# 기존 DROID:
#   correlation + flow/residual -> ConvGRU
#       -> delta  : correspondence correction
#       -> weight : visual confidence
#
# 현재 확장:
#       -> imu_confidence : 각 visual edge의 hidden state에서 IMU 신뢰도를 예측
#
# imu_confidence는 Sigmoid를 사용하므로 0~1 범위이며, 이후 temporal IMU factor
# 단위 confidence로 변환되어 BA의 IMU residual weighting에 전달된다.
# =============================================================================
class UpdateModule(nn.Module):

    def __init__(self):

        super(UpdateModule, self).__init__()

        cor_planes = 4 * (2*3 + 1)**2



        self.corr_encoder = nn.Sequential(

            nn.Conv2d(cor_planes, 128, 1, padding=0),

            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),

            nn.ReLU(inplace=True))



        self.flow_encoder = nn.Sequential(

            nn.Conv2d(4, 128, 7, padding=3),

            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=1),

            nn.ReLU(inplace=True))



        self.weight = nn.Sequential(

            nn.Conv2d(128, 128, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(128, 3, 3, padding=1),

            GradientClip(),

            nn.Sigmoid())



        self.delta = nn.Sequential(

            nn.Conv2d(128, 128, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(128, 3, 3, padding=1),

            GradientClip())



        # 학습 기반 IMU confidence head.
        # visual recurrent hidden state를 입력으로 받아 IMU factor의 상대 신뢰도를 예측한다.
        self.imu_confidence = nn.Sequential(

            nn.Conv2d(128, 128, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1, 3, padding=1),

            GradientClip(),

            nn.Sigmoid())

        # DVI-SLAM scalar confidence head.  Keep it independent from the
        # legacy 3x3 head and from the later component-confidence experiment:
        # DVI uses a 1x1 projection of the visual recurrent state followed by
        # spatial average pooling to obtain one confidence per graph edge.
        # The zero logit initialization makes its pre-training output exactly
        # 0.5, so the learned arm has a documented neutral starting point.
        self.dvi_imu_confidence = nn.Conv2d(128, 1, kernel_size=1)
        nn.init.zeros_(self.dvi_imu_confidence.weight)
        nn.init.zeros_(self.dvi_imu_confidence.bias)

        # Add direct preintegrated-IMU quality to the DVI scalar confidence
        # without changing the scalar factor contract. The final projection
        # starts at zero, so old scalar-DVI behaviour is preserved until this
        # branch is explicitly enabled and trained.
        self.dvi_imu_quality_fusion = nn.Sequential(

            nn.Linear(7, 32),

            nn.ReLU(inplace=True),

            nn.Linear(32, 1))
        nn.init.zeros_(self.dvi_imu_quality_fusion[2].weight)
        nn.init.zeros_(self.dvi_imu_quality_fusion[2].bias)

        # Component-wise IMU confidence.  The output channel order is fixed
        # as [rotation, velocity, position].  It is intentionally separate
        # from the legacy scalar head so old checkpoints and ablations remain
        # reproducible unless the new path is explicitly enabled.
        self.imu_component_confidence = nn.Sequential(

            nn.Conv2d(128, 128, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(128, 3, 3, padding=1),

            GradientClip())

        # Preintegrated interval quality is available at training and runtime:
        # [dt, angular rate, acceleration proxy, position proxy,
        #  rotation/velocity/position information].  The final layer is reset
        # to zero when component mode is initialized from a scalar checkpoint,
        # which preserves the scalar output at step zero while retaining a
        # nonzero hidden representation for gradient flow.
        self.imu_quality_fusion = nn.Sequential(

            nn.Linear(7, 32),

            nn.ReLU(inplace=True),

            nn.Linear(32, 3))



        self.gru = ConvGRU(128, 128+128+64)

        self.agg = GraphAgg()



    def initialize_component_imu_confidence_from_scalar(self):
        """Seed the three confidence channels from a trained scalar head.

        This compatibility initializer gives a component-confidence fine-tune
        the exact old behavior at step zero. The quality MLP output starts
        neutral, but its first layer keeps its standard nonzero initialization
        so the final layer can learn from IMU-quality features immediately.
        """

        with torch.no_grad():
            self.imu_component_confidence[0].weight.copy_(self.imu_confidence[0].weight)
            self.imu_component_confidence[0].bias.copy_(self.imu_confidence[0].bias)
            self.imu_component_confidence[2].weight.copy_(
                self.imu_confidence[2].weight.expand_as(
                    self.imu_component_confidence[2].weight
                )
            )
            self.imu_component_confidence[2].bias.copy_(
                self.imu_confidence[2].bias.expand_as(
                    self.imu_component_confidence[2].bias
                )
            )
            # Do not zero both MLP layers: an all-zero two-layer ReLU MLP has
            # zero gradients everywhere and can never use IMU-quality inputs.
            # Zeroing only the output projection keeps its initial additive
            # logit exactly zero while allowing that projection to learn.
            self.imu_quality_fusion[2].weight.zero_()
            self.imu_quality_fusion[2].bias.zero_()


    def initialize_dvi_imu_quality_fusion(self):
        """Reset only the scalar DVI quality contribution to neutral.

        The first layer remains randomly initialized so the output projection
        receives a nonzero gradient on the first learning step.
        """

        with torch.no_grad():
            self.dvi_imu_quality_fusion[2].weight.zero_()
            self.dvi_imu_quality_fusion[2].bias.zero_()


    def dvi_imu_quality_logits(self, imu_quality):
        """Predict one scalar reliability logit from each IMU interval.

        This path is used when an otherwise-valid temporal IMU factor has no
        direct visual graph edge. It reuses the trained DVI quality MLP and
        is not a second confidence head or a global IMU multiplier.
        """

        if imu_quality is None:
            return None
        quality = imu_quality
        if quality.ndim == 2:
            quality = quality.unsqueeze(0)
        if quality.ndim != 3 or quality.shape[-1] != 7:
            raise ValueError(
                "imu_quality must be [B, N, 7] for DVI scalar quality "
                f"confidence, got {tuple(imu_quality.shape)}"
            )
        quality = quality.to(
            device=self.dvi_imu_confidence.weight.device,
            dtype=self.dvi_imu_confidence.weight.dtype,
        )
        return self.dvi_imu_quality_fusion(quality)


    def dvi_imu_quality_confidence(self, imu_quality):
        """Return one quality-only confidence for each temporal IMU row."""

        logits = self.dvi_imu_quality_logits(imu_quality)
        return None if logits is None else torch.sigmoid(logits)


    def forward(
        self,
        net,
        inp,
        corr,
        flow=None,
        ii=None,
        jj=None,
        return_imu_confidence=False,
        return_component_imu_confidence=False,
        use_dvi_imu_confidence=False,
        use_dvi_imu_quality_fusion=False,
        imu_quality=None,
    ):

        """ RaftSLAM update operator """



        batch, num, ch, ht, wd = net.shape



        if flow is None:

            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)



        output_dim = (batch, num, -1, ht, wd)

        net = net.view(batch*num, -1, ht, wd)

        inp = inp.view(batch*num, -1, ht, wd)

        corr = corr.view(batch*num, -1, ht, wd)

        flow = flow.view(batch*num, -1, ht, wd)



        corr = self.corr_encoder(corr)

        flow = self.flow_encoder(flow)

        net = self.gru(net, inp, corr, flow)



        ### update variables ###

        delta = self.delta(net).view(*output_dim)

        weight = self.weight(net).view(*output_dim)

        if return_component_imu_confidence:
            component_logits = self.imu_component_confidence(net).view(*output_dim)
            component_logits = component_logits.permute(0, 1, 3, 4, 2).contiguous()
            if imu_quality is None:
                quality_logits = torch.zeros(
                    (batch, num, 1, 1, 3),
                    device=net.device,
                    dtype=component_logits.dtype,
                )
            else:
                quality = imu_quality.to(device=net.device, dtype=component_logits.dtype)
                if quality.ndim == 2:
                    quality = quality.unsqueeze(0)
                if quality.shape[:2] != (batch, num) or quality.shape[-1] != 7:
                    raise ValueError(
                        "imu_quality must be [B, E, 7] for component confidence, "
                        f"got {tuple(imu_quality.shape)} for B={batch}, E={num}"
                    )
                quality_logits = self.imu_quality_fusion(quality).unsqueeze(2).unsqueeze(2)
            imu_confidence = torch.sigmoid(component_logits + quality_logits)
        elif use_dvi_imu_confidence:
            # [B*E, 1, H, W] -> one DVI scalar per edge.  Retaining the two
            # singleton spatial dimensions keeps the tensor compatible with
            # the existing edge-to-temporal confidence aggregation helper.
            dvi_logits = self.dvi_imu_confidence(net)
            dvi_logits = dvi_logits.mean(dim=(2, 3), keepdim=True).view(
                batch, num, 1, 1, 1
            )
            if use_dvi_imu_quality_fusion:
                if imu_quality is None:
                    quality_logits = torch.zeros_like(dvi_logits)
                else:
                    quality_logits = self.dvi_imu_quality_logits(imu_quality)
                    if quality_logits.shape[:2] != (batch, num):
                        raise ValueError(
                            "imu_quality must be [B, E, 7] for DVI scalar "
                            f"quality fusion, got {tuple(imu_quality.shape)} for "
                            f"B={batch}, E={num}"
                        )
                    quality_logits = quality_logits.to(
                        device=net.device, dtype=net.dtype
                    ).view(
                        batch, num, 1, 1, 1
                    )
                imu_confidence = torch.sigmoid(dvi_logits + quality_logits)
            else:
                imu_confidence = torch.sigmoid(dvi_logits)
        else:
            imu_confidence = self.imu_confidence(net).view(*output_dim)



        delta = delta.permute(0,1,3,4,2)[...,:2].contiguous()

        weight = weight.permute(0,1,3,4,2)[...,:2].contiguous()

        if not return_component_imu_confidence:
            imu_confidence = imu_confidence.permute(0,1,3,4,2)[...,:1].contiguous()



        net = net.view(*output_dim)



        if ii is not None:

            eta, upmask = self.agg(net, ii.to(net.device))

            if return_imu_confidence:

                return net, delta, weight, eta, upmask, imu_confidence

            return net, delta, weight, eta, upmask



        else:

            if return_imu_confidence:

                return net, delta, weight, imu_confidence

            return net, delta, weight





# =============================================================================
# [전체 DROID-SLAM + IMU 네트워크]
#
# Visual branch
#   images -> FNet/CNet -> CorrBlock -> ConvGRU
#          -> delta / visual weight -> Dense BA
#
# IMU extension
#   imu_delta (preintegration)
#       + learned IMU confidence
#       + adaptive gate
#       + gyro/acc bias
#       + position/velocity residual weight
#       -> Visual-Inertial BA
#
# 이 파일의 핵심 역할은 IMU를 직접 적분하는 것이 아니라,
# preintegrated IMU 정보를 visual recurrent update 및 BA와 연결하는 것이다.
# =============================================================================
class DroidNet(nn.Module):

    def __init__(self):

        super(DroidNet, self).__init__()

        # FNet: 두 프레임 사이의 correspondence를 찾기 위한 feature encoder
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')

        # CNet: recurrent update에 필요한 context feature 생성
        # 256채널 출력은 이후 hidden state(net) 128 + context input(inp) 128로 분리된다.
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')

        self.update = UpdateModule()

        # 전역 학습 파라미터 형태의 gyroscope bias
        self.imu_gyro_bias = nn.Parameter(torch.zeros(3))

        # 전역 학습 파라미터 형태의 accelerometer bias
        self.imu_acc_bias = nn.Parameter(torch.zeros(3))

        # 전체 IMU factor strength.
        # log-domain parameter를 Softplus로 양수 weight로 변환한다.
        self.imu_ba_log_weight = nn.Parameter(torch.tensor(-3.0))

        # Data-derived scale for the pre-BA visual/IMU calibration loss. This
        # is deliberately not a BA factor multiplier or a fixed confidence
        # schedule; it merely makes the pose/IMU energy dimensionless while
        # preserving the same reference across resumed training segments.
        self.register_buffer(
            "imu_confidence_energy_log_reference", torch.tensor(float("nan"))
        )

        # Relative translation terms are learned independently from the global

        # IMU factor strength. Rotation remains the reference term, preventing

        # a global-scale ambiguity between all three residual categories.

        self.imu_pos_log_weight = nn.Parameter(

            torch.tensor(np.log(np.expm1(0.05)), dtype=torch.float32)

        )

        self.imu_vel_log_weight = nn.Parameter(

            torch.tensor(np.log(np.expm1(0.05)), dtype=torch.float32)

        )



    def set_imu_ba_weight(self, value):

        value = max(float(value), 1e-8)

        with torch.no_grad():

            self.imu_ba_log_weight.fill_(float(np.log(np.expm1(value))))



    def get_imu_ba_weight(self):

        return F.softplus(self.imu_ba_log_weight)



    def set_imu_full_residual_weights(self, pos_value, vel_value):

        values = [max(float(pos_value), 1e-8), max(float(vel_value), 1e-8)]

        with torch.no_grad():

            self.imu_pos_log_weight.fill_(float(np.log(np.expm1(values[0]))))

            self.imu_vel_log_weight.fill_(float(np.log(np.expm1(values[1]))))



    def get_imu_full_residual_weights(self):

        return torch.stack(

            [F.softplus(self.imu_pos_log_weight), F.softplus(self.imu_vel_log_weight)]

        )


    def initialize_component_imu_confidence_from_scalar(self):
        self.update.initialize_component_imu_confidence_from_scalar()


    def initialize_dvi_imu_quality_fusion(self):
        self.update.initialize_dvi_imu_quality_fusion()


    def reset_imu_confidence_calibration(self):
        """Forget an old run's pre-BA residual reference for a fresh run."""

        with torch.no_grad():
            self.imu_confidence_energy_log_reference.fill_(float("nan"))





    def extract_features(self, images):

        """ run feeature extraction networks """



        # normalize images

        images = images[:, :, [2,1,0]] / 255.0

        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)

        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)

        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])



        fmaps = self.fnet(images)

        net = self.cnet(images)



        net, inp = net.split([128,128], dim=2)

        net = torch.tanh(net)

        inp = torch.relu(inp)

        return fmaps, net, inp





    def forward(

        self,

        Gs,

        images,

        disps,

        intrinsics,

        graph=None,

        num_steps=12,

        fixedp=2,

        return_imu_bias=False,

        imu_delta=None,

        imu_valid=None,

        imu_info=None,

        imu_info_reference=None,

        imu_bias_jacobian=None,

        imu_camera_translation=None,

        imu_ba_weight=None,

        imu_confidence_only_weighting=False,

        imu_ba_max_residual=0.5,

        imu_confidence_floor=0.0,

        return_imu_confidence=False,

        use_component_imu_confidence=False,

        use_dvi_imu_confidence=False,

        use_dvi_imu_quality_fusion=False,

        use_full_imu_ba=False,

        imu_pose_only_steps=0,

        imu_full_pos_weight=0.05,

        imu_full_vel_weight=0.05,

        use_learned_imu_residual_weights=False,

        imu_full_bias_weight=0.001,

        imu_velocity_init="pose",

        imu_motion_prior_weight=0.0,

        imu_local_bias_prior_weight=0.0,

        imu_gravity=None,

        imu_full_compose_order="dq_prev",

        imu_full_inverse=True,

        use_imu_info_weighting=False,

        imu_info_weight_clip=4.0,

        imu_info_weight_eps=1e-12,

        imu_optimize_bias=True,

        use_adaptive_imu_gate=False,

        imu_gate_min=0.2,

        imu_gate_max=1.25,

        imu_gate_visual_thresh=0.2,

        imu_gate_visual_boost=0.25,

        imu_gate_motion_thresh_deg=5.0,

        imu_gate_motion_boost=0.15,

        imu_gate_residual_soft_deg=2.5,

        imu_gate_residual_hard_deg=8.0,

        imu_gate_residual_pos_soft=0.0,

        imu_gate_residual_vel_soft=0.0,

        return_imu_adaptive_gate=False,

        return_imu_motion=False,

    ):

        """ Estimates SE3 or Sim3 between pair of frames """



        u = keyframe_indicies(graph)

        ii, jj, kk = graph_to_edge_list(graph)



        ii = ii.to(device=images.device, dtype=torch.long)

        jj = jj.to(device=images.device, dtype=torch.long)



        # 1) 영상 feature/context 추출
        fmaps, net, inp = self.extract_features(images)

        net, inp = net[:,ii], inp[:,ii]

        # 2) graph edge (ii -> jj)에 대해 multi-scale correlation volume 구성
        corr_fn = CorrBlock(fmaps[:,ii], fmaps[:,jj], num_levels=4, radius=3)



        ht, wd = images.shape[-2:]

        coords0 = pops.coords_grid(ht//8, wd//8, device=images.device)



        # 3) 현재 pose와 inverse depth로 source pixel을 target frame에 재투영
        coords1, _ = pops.projective_transform(Gs, disps, intrinsics, ii, jj)

        target = coords1.clone()



        # imu_delta가 전달된 경우 visual-only BA가 아니라 IMU factor도 활성화
        use_imu_ba = imu_delta is not None

        use_component_imu_confidence = bool(
            use_component_imu_confidence and (return_imu_confidence or use_imu_ba)
        )
        use_dvi_imu_confidence = bool(
            use_dvi_imu_confidence
            and not use_component_imu_confidence
            and (return_imu_confidence or use_imu_ba)
        )
        use_dvi_imu_quality_fusion = bool(
            use_dvi_imu_quality_fusion and use_dvi_imu_confidence
        )

        use_full_imu_ba = bool(use_full_imu_ba and use_imu_ba)

        if use_imu_ba:

            # In this mode confidence is the only learned attenuation of a
            # temporal IMU factor.  Keep ``imu_ba_log_weight`` in the module
            # for old checkpoint compatibility, but do not read it here.
            # This makes c=1 mean full covariance-relative IMU strength.
            if imu_confidence_only_weighting:
                imu_ba_weight_value = 1.0
            else:
                imu_ba_weight_value = (
                    self.get_imu_ba_weight()
                    if imu_ba_weight is None
                    else imu_ba_weight
                )

        else:

            imu_ba_weight_value = 0.0



        if use_full_imu_ba and use_learned_imu_residual_weights:

            learned_residual_weights = self.get_imu_full_residual_weights()

            imu_full_pos_weight = learned_residual_weights[0]

            imu_full_vel_weight = learned_residual_weights[1]



        # Delay the full inertial state initialization until the visual-only
        # prefix has refined pose/depth. This matches the runtime two-stage
        # local DBA instead of deriving velocity from the raw pose seed.
        full_imu_schedule = full_imu_unroll_schedule(
            num_steps,
            use_full_imu_ba,
            pose_only_steps=imu_pose_only_steps,
        )
        imu_motion = None
        imu_motion_prior = None



        Gs_list, disp_list, residual_list = [], [], []

        imu_confidence_list, imu_adaptive_gate_list, imu_motion_list = [], [], []
        imu_edge_quality = (
            edge_imu_quality_features(
                imu_delta,
                imu_info,
                ii,
                jj,
                num_frames=Gs.data.shape[1],
            )
            if use_component_imu_confidence or use_dvi_imu_quality_fusion
            else None
        )
        # In a streaming factor graph, an IMU interval may remain valid while
        # its direct visual edge is absent.  Prepare a quality-only confidence
        # for those rows now; direct temporal edges still receive the normal
        # visual-GRU + quality prediction below.
        imu_temporal_quality = (
            temporal_imu_quality_features(
                imu_delta,
                imu_info,
                num_frames=Gs.data.shape[1],
            )
            if use_dvi_imu_quality_fusion
            else None
        )
        imu_quality_fallback = (
            self.update.dvi_imu_quality_confidence(imu_temporal_quality)
            if imu_temporal_quality is not None
            else None
        )

        # DROID는 한 번에 pose를 예측하지 않고 ConvGRU -> BA를 여러 번 unroll한다.
        for step in range(num_steps):

            step_use_full_imu_ba = full_imu_schedule[step]
            # A pose-only prefix must be genuinely visual-only: passing
            # imu_delta to BA would otherwise activate the legacy rotation
            # factor even while full IMU BA is disabled for this step.
            step_use_imu_ba = use_imu_ba and (
                not use_full_imu_ba or step_use_full_imu_ba
            )

            Gs = Gs.detach()

            disps = disps.detach()

            if step_use_full_imu_ba and imu_motion is None:
                imu_motion = initialize_imu_motion(
                    Gs,
                    imu_delta,
                    mode=imu_velocity_init,
                    imu_camera_translation=imu_camera_translation,
                )
                imu_motion_prior = imu_motion.clone()

            if imu_motion is not None:

                # BA updates velocity jointly with pose.  Reinitializing it

                # from pose differences here would erase that update at every

                # unrolled step, unlike runtime local DBA.  Keep the initial

                # visual velocity only as the optional motion prior.

                imu_motion = imu_motion.detach()

                if imu_motion_prior is not None:

                    imu_motion_prior = imu_motion_prior.detach()

            coords1 = coords1.detach()

            target = target.detach()



            # extract motion features

            # 현재 correspondence 위치에서 correlation feature lookup
            corr = corr_fn(coords1)

            # 이전 iteration의 target과 현재 geometric projection 사이 residual
            resd = target - coords1

            # 현재 추정된 optical/geometric flow
            flow = coords1 - coords0



            motion = torch.cat([flow, resd], dim=-1)

            motion = motion.permute(0,1,4,2,3).clamp(-64.0, 64.0)



            update_out = self.update(

                net,

                inp,

                corr,

                motion,

                ii,

                jj,

                return_imu_confidence=return_imu_confidence or step_use_imu_ba,

                return_component_imu_confidence=use_component_imu_confidence,

                use_dvi_imu_confidence=use_dvi_imu_confidence,

                use_dvi_imu_quality_fusion=use_dvi_imu_quality_fusion,

                imu_quality=imu_edge_quality,

            )



            if return_imu_confidence or step_use_imu_ba:

                net, delta, weight, eta, upmask, imu_edge_confidence = update_out

                imu_confidence = temporal_imu_confidence_from_edges(

                    imu_edge_confidence,

                    ii,

                    jj,

                    num_frames=Gs.data.shape[1],

                    fallback=(
                        imu_quality_fallback
                        if imu_quality_fallback is not None
                        else 1.0
                    ),

                )

                if imu_confidence_floor > 0.0:

                    floor = float(imu_confidence_floor)

                    imu_confidence = floor + (1.0 - floor) * imu_confidence

            else:

                net, delta, weight, eta, upmask = update_out

                imu_confidence = None



            imu_adaptive_gate = None

            if step_use_imu_ba and use_adaptive_imu_gate:

                visual_confidence = temporal_visual_confidence(

                    weight,

                    ii,

                    jj,

                    Gs.data.shape[1],

                )

                imu_adaptive_gate = compute_imu_adaptive_gate(

                    Gs.data.detach(),

                    imu_delta,

                    visual_confidence=visual_confidence,

                    imu_motion=imu_motion.detach() if imu_motion is not None else None,

                    imu_bias_jacobian=imu_bias_jacobian,

                    gyro_bias=self.imu_gyro_bias.detach(),

                    accel_bias=self.imu_acc_bias.detach(),

                    gravity=imu_gravity,

                    imu_optimize_bias=imu_optimize_bias,

                    compose_order=(
                        imu_full_compose_order
                        if step_use_full_imu_ba
                        else "prev_dq"
                    ),

                    inverse_imu=imu_full_inverse if step_use_full_imu_ba else False,

                    gate_min=imu_gate_min,

                    gate_max=imu_gate_max,

                    visual_threshold=imu_gate_visual_thresh,

                    visual_boost=imu_gate_visual_boost,

                    motion_threshold_deg=imu_gate_motion_thresh_deg,

                    motion_boost=imu_gate_motion_boost,

                    residual_soft_deg=imu_gate_residual_soft_deg,

                    residual_hard_deg=imu_gate_residual_hard_deg,

                    residual_pos_soft=imu_gate_residual_pos_soft,

                    residual_vel_soft=imu_gate_residual_vel_soft,

                )



            # ConvGRU가 예측한 delta를 적용해 DBA가 맞춰야 할 correspondence target 갱신
            target = coords1 + delta



            for i in range(2):

                ba_out = BA(

                    target,

                    weight,

                    eta,

                    Gs,

                    disps,

                    intrinsics,

                    ii,

                    jj,

                    fixedp=fixedp,

                    imu_delta=imu_delta if step_use_imu_ba else None,

                    imu_valid=imu_valid if step_use_imu_ba else None,

                    imu_info=imu_info if step_use_imu_ba else None,

                    imu_info_reference=(
                        imu_info_reference if step_use_imu_ba else None
                    ),

                    imu_bias_jacobian=(
                        imu_bias_jacobian if step_use_imu_ba else None
                    ),

                    gyro_bias=self.imu_gyro_bias,

                    imu_factor_weight=(
                        imu_ba_weight_value if step_use_imu_ba else 0.0
                    ),

                    imu_confidence=(
                        imu_confidence if step_use_imu_ba else None
                    ),

                    imu_adaptive_gate=(
                        imu_adaptive_gate if step_use_imu_ba else None
                    ),

                    imu_max_residual=imu_ba_max_residual,

                    imu_motion=imu_motion,

                    imu_motion_prior=imu_motion_prior,

                    accel_bias=self.imu_acc_bias,

                    imu_camera_translation=imu_camera_translation,

                    use_full_imu=step_use_full_imu_ba,

                    imu_full_pos_weight=imu_full_pos_weight,

                    imu_full_vel_weight=imu_full_vel_weight,

                    imu_full_bias_weight=imu_full_bias_weight,

                    imu_motion_prior_weight=imu_motion_prior_weight,

                    imu_local_bias_prior_weight=imu_local_bias_prior_weight,

                    imu_gravity=imu_gravity,

                    imu_full_compose_order=imu_full_compose_order,

                    imu_full_inverse=imu_full_inverse,

                    use_imu_info_weighting=use_imu_info_weighting,

                    imu_info_weight_clip=imu_info_weight_clip,

                    imu_info_weight_eps=imu_info_weight_eps,

                    imu_optimize_bias=imu_optimize_bias,

                )

                if step_use_full_imu_ba:

                    Gs, disps, imu_motion = ba_out

                else:

                    Gs, disps = ba_out



            coords1, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj)

            residual = (target - coords1)



            Gs_list.append(Gs)

            disp_list.append(upsample_disp(disps, upmask))

            residual_list.append(valid_mask * residual)

            if imu_confidence is not None:

                imu_confidence_list.append(imu_confidence)

            if imu_adaptive_gate is not None:

                imu_adaptive_gate_list.append(imu_adaptive_gate)

            # This list deliberately begins at the full-IMU stage. Training
            # slices pose/confidence outputs by the same prefix before
            # evaluating r_p, r_v, r_R, r_ba, r_bg.
            if return_imu_motion and step_use_full_imu_ba and imu_motion is not None:

                imu_motion_list.append(imu_motion)



        if return_imu_confidence:

            out = [Gs_list, disp_list, residual_list, self.imu_gyro_bias, imu_confidence_list]

            if return_imu_adaptive_gate:

                out.append(imu_adaptive_gate_list)

            if return_imu_motion:

                out.append(imu_motion_list)

            return tuple(out)



        if return_imu_bias:

            out = [Gs_list, disp_list, residual_list, self.imu_gyro_bias]

            if return_imu_motion:

                out.append(imu_motion_list)

            return tuple(out)



        return Gs_list, disp_list, residual_list

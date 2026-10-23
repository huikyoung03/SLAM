import torch
import lietorch
import numpy as np
import csv

import matplotlib.pyplot as plt
from pathlib import Path
from lietorch import SE3
from modules.corr import CorrBlock, AltCorrBlock
import geom.projective_ops as pops
from geom.imu_factor import (
    quat_inverse,
    quat_multiply,
    camera_center_from_pose_data,
    imu_center_from_pose_data,
    predict_imu_rotation,
    quat_rotate,
    quat_to_rotmat,
    quat_to_rotvec,
    rotvec_to_quat,
    full_imu_factor_linearization,
    bias_random_walk_relative_sqrt_information,
    select_imu_bias_jacobian,
    skew_symmetric,
    so3_left_jacobian,
    so3_left_jacobian_inv,
    edge_imu_quality_features,
    temporal_imu_confidence_from_edges,
    temporal_imu_precision_from_edges,
    temporal_imu_quality_features,
)

from cuda_timer import CudaTimer
from functools import partial

try:
    from imu_prior import compose_imu_prior_rows
except ImportError:
    from droid_slam.imu_prior import compose_imu_prior_rows


'''
프레임 간 edge/factor를 만들고,
neural update로 correspondence target과 weight를 예측한 뒤,
DepthVideo.ba()를 호출해서 pose와 disparity를 최적화하는 핵심 그래프 관리 코드

add_factors(ii, jj)
프레임 ii → jj edge를 factor graph에 추가

add_neighborhood_factors()
초기화 때 시간적으로 가까운 프레임들을 연결

add_proximity_factors()
pose/depth 기반 거리로 가까운 프레임들을 찾아 edge 추가

update()
frontend용 graph update + dense BA

update_lowmem()
backend용 low-memory graph update + dense BA

rm_factors()
오래되거나 불필요한 edge 제거

rm_keyframe()
중복 keyframe 제거 및 index 재정렬

DepthVideo에 keyframe 저장
        ↓
FactorGraph.add_factors()
        ↓
CorrBlock / AltCorrBlock 생성
        ↓
DepthVideo.reproject()
        ↓
motion feature 생성
        ↓
DroidNet.update()
        ↓
target, weight, damping 예측
        ↓
DepthVideo.ba()
        ↓
droid_backends.ba()
        ↓
pose / disparity 최적화

'''


# PyTorch 버전에 따라 autocast 사용 방식 분기
# autocast는 mixed precision 연산을 사용해서 GPU 메모리 사용량을 줄이고 속도를 높인다.
if torch.__version__.startswith("2"):
    autocast = partial(torch.autocast, device_type="cuda")
else:
    autocast = torch.cuda.amp.autocast


class FactorGraph:
    def __init__(
        self,
        video,
        update_op,
        device="cuda",
        corr_impl="volume",
        max_factors=-1,
        upsample=False,
        imu_regularizer=None,
        apply_imu_residual=None,
        use_learned_imu_confidence=False,
        use_learned_imu_precision=False,
        use_component_imu_confidence=False,
        use_dvi_imu_confidence=False,
        use_dvi_imu_quality_fusion=False,
        imu_confidence_floor=0.0,
        imu_tuned_confidence_schedule=None,
        use_imu_ba_prior=False,
        imu_ba_prior_weight=0.0,
        imu_ba_prior_max_deg=45.0,
        use_full_imu_ba=False,
        require_imu_alignment=False,
        imu_full_pos_weight=0.05,
        imu_full_vel_weight=0.05,
        imu_full_bias_weight=0.001,
        use_imu_bias_random_walk=False,
        use_imu_full_covariance_weighting=False,
        imu_motion_prior_weight=0.0,
        imu_local_bias_prior_weight=0.0,
        imu_gravity=None,
        imu_full_compose_order="dq_prev",
        imu_full_inverse=True,
        imu_full_max_dt=0.5,
        imu_full_max_dv=5.0,
        imu_full_max_dp=1.0,
        use_imu_info_weighting=False,
        imu_info_weight_clip=4.0,
        imu_info_weight_eps=1e-12,
        imu_info_reference=None,
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
        imu_optimize_bias=True,
        imu_gyro_bias=None,
        imu_acc_bias=None,
        use_joint_shared_bias_ba=False,
        use_joint_shared_bias_15dof=False,
        imu_joint_bias_interval=10,
        imu_joint_bias_window=30,
        imu_joint_bias_damping=1e-4,
        imu_joint_bias_step_scale=1.0,
        imu_joint_bias_max_accel_step=0.15,
        imu_joint_bias_max_gyro_step=0.015,
        imu_joint_bias_max_accel_norm=0.5,
        imu_joint_bias_max_gyro_norm=0.1,
        imu_joint_bias_min_edges=8,
        imu_joint_bias_min_duration=0.5,
        imu_joint_bias_min_rotation_deg=2.0,
        imu_joint_bias_max_condition=1e8,
        imu_joint_bias_accel_random_walk=-1.0,
        imu_joint_bias_gyro_random_walk=-1.0,
        imu_joint_bias_accel_prior_std=0.05,
        imu_joint_bias_gyro_prior_std=0.005,
        imu_joint_bias_min_improvement=1e-10,
        imu_joint_bias_relinearize_accel=0.05,
        imu_joint_bias_relinearize_gyro=0.005,
        imu_ba_outer_relinearize=False,
        imu_ba_outer_relinearize_interval=10,
        imu_ba_outer_window=0,
        imu_ba_debug=False,
        imu_ba_debug_path=None,
        imu_ba_debug_max_rows=20000,
        imu_ba_debug_stage="graph",
    ):
        """
        DROID-SLAM의 factor graph를 관리하는 클래스.

        역할:
            1. 프레임 간 edge, 즉 factor를 저장한다.
            2. 각 edge에 대해 correlation volume을 구성한다.
            3. 현재 pose/depth로 source frame을 target frame에 reproject한다.
            4. update network를 이용해 correspondence target과 confidence weight를 예측한다.
            5. DepthVideo.ba()를 호출해 pose와 disparity를 최적화한다.
            6. 오래된 factor, 중복 factor, 나쁜 factor를 제거한다.

        video:
            DepthVideo 객체.
            poses, disps, intrinsics, fmaps, nets, inps 등이 저장되어 있음.

        update_op:
            DroidNet의 update block.
            correlation + motion feature를 입력으로 받아 delta, weight, damping 등을 예측한다.

        corr_impl:
            correlation 계산 방식.
            "volume"은 일반 frontend용 correlation volume.
            "alt"는 backend에서 쓰는 low-memory correlation 방식.

        max_factors:
            factor graph에 유지할 최대 edge 수.
            -1이면 제한 없음.

        upsample:
            disparity upsampling 수행 여부.
        """

        # DepthVideo 저장소
        self.video = video

        # DroidNet update operator
        self.update_op = update_op

        # 연산 device
        self.device = device

        # 최대 factor 수
        self.max_factors = max_factors

        # correlation 구현 방식
        self.corr_impl = corr_impl

        # disparity upsampling 여부
        self.upsample = upsample

        # Optional rotation-only inertial residual regularizer.
        # This is the first-stage E^u hook: it runs after visual DBA and softly
        # corrects pose rotations using preintegrated IMU priors.
        self.imu_regularizer = imu_regularizer
        self.apply_imu_residual = (
            imu_regularizer is not None
            if apply_imu_residual is None
            else bool(apply_imu_residual)
        )
        # This legacy-named route is distinct from old scalar confidence. Its
        # three R/v/p outputs are covariance-relative reliabilities: one
        # retains the physical IMU factor and values below one attenuate it.
        self.use_learned_imu_precision = bool(use_learned_imu_precision)
        self.use_learned_imu_confidence = bool(
            use_learned_imu_confidence and not self.use_learned_imu_precision
        )
        self.use_learned_imu_prediction = bool(
            self.use_learned_imu_confidence or self.use_learned_imu_precision
        )
        self.use_component_imu_confidence = bool(
            use_component_imu_confidence and self.use_learned_imu_confidence
        )
        self.use_dvi_imu_confidence = bool(
            use_dvi_imu_confidence
            and self.use_learned_imu_confidence
            and not self.use_component_imu_confidence
        )
        self.use_dvi_imu_quality_fusion = bool(
            use_dvi_imu_quality_fusion and self.use_dvi_imu_confidence
        )
        self.imu_confidence_floor = float(imu_confidence_floor)
        if imu_tuned_confidence_schedule is None:
            self.imu_tuned_confidence_schedule = None
        else:
            schedule = tuple(float(value) for value in imu_tuned_confidence_schedule)
            if not schedule or any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in schedule):
                raise ValueError(
                    "imu_tuned_confidence_schedule must contain finite values in [0, 1]"
                )
            self.imu_tuned_confidence_schedule = schedule
        # The schedule advances once per outer GRU/DBA update, never per
        # inner CUDA-LM iteration.  This is the manual-iteration control in
        # the DVI four-way ablation, not an extra optimization mechanism.
        self._imu_tuned_confidence_iteration = 0
        self.use_imu_ba_prior = bool(use_imu_ba_prior)
        self.imu_ba_prior_weight = float(imu_ba_prior_weight)
        self.imu_ba_prior_max_deg = float(imu_ba_prior_max_deg)
        self.use_full_imu_ba = bool(use_full_imu_ba)
        self.require_imu_alignment = bool(require_imu_alignment)
        self.imu_full_pos_weight = float(imu_full_pos_weight)
        self.imu_full_vel_weight = float(imu_full_vel_weight)
        self.imu_full_bias_weight = float(imu_full_bias_weight)
        # Legacy checkpoints used one manually selected coefficient for both
        # bias-transition blocks.  The optional calibrated mode replaces that
        # coefficient per temporal edge with sensor random-walk information.
        self.use_imu_bias_random_walk = bool(use_imu_bias_random_walk)
        # In the compatibility path, p/v/R retain the historical hand-tuned
        # cross-block weights (0.05, 0.05, 1.0).  This calibrated path instead
        # puts all three preintegration blocks in one covariance reference
        # frame.  It must be opt-in: older priors only supply relative row
        # quality and cannot support a physically meaningful cross-block ratio.
        self.use_imu_full_covariance_weighting = bool(
            use_imu_full_covariance_weighting
        )
        self.imu_motion_prior_weight = float(imu_motion_prior_weight)
        self.imu_local_bias_prior_weight = float(imu_local_bias_prior_weight)
        self.imu_gravity = imu_gravity
        self.imu_full_compose_order = str(imu_full_compose_order)
        self.imu_full_inverse = bool(imu_full_inverse)
        self.imu_full_max_dt = float(imu_full_max_dt)
        self.imu_full_max_dv = float(imu_full_max_dv)
        self.imu_full_max_dp = float(imu_full_max_dp)
        self.use_imu_info_weighting = bool(use_imu_info_weighting)
        self.imu_info_weight_clip = float(imu_info_weight_clip)
        self.imu_info_weight_eps = float(imu_info_weight_eps)
        self.imu_info_reference = imu_info_reference
        self._imu_info_reference_mode = "unavailable"
        self.use_adaptive_imu_gate = bool(use_adaptive_imu_gate)
        self.imu_gate_min = max(0.0, float(imu_gate_min))
        self.imu_gate_max = max(self.imu_gate_min, float(imu_gate_max))
        self.imu_gate_visual_thresh = float(imu_gate_visual_thresh)
        self.imu_gate_visual_boost = max(0.0, float(imu_gate_visual_boost))
        self.imu_gate_motion_thresh_deg = float(imu_gate_motion_thresh_deg)
        self.imu_gate_motion_boost = max(0.0, float(imu_gate_motion_boost))
        self.imu_gate_residual_soft_deg = float(imu_gate_residual_soft_deg)
        self.imu_gate_residual_hard_deg = float(imu_gate_residual_hard_deg)
        self.imu_gate_residual_pos_soft = float(imu_gate_residual_pos_soft)
        self.imu_gate_residual_vel_soft = float(imu_gate_residual_vel_soft)
        self.imu_optimize_bias = bool(imu_optimize_bias)
        # Preserve the requested full-IMU configuration while the frontend
        # temporarily disables every inertial term for its visual stage.
        self._configured_full_imu_ba = self.use_full_imu_ba
        self._configured_imu_ba_prior = self.use_imu_ba_prior
        self._configured_imu_optimize_bias = self.imu_optimize_bias
        self.imu_gyro_bias = imu_gyro_bias
        self.imu_acc_bias = imu_acc_bias
        # Keep one shared accelerometer/gyroscope bias inside a DBA normal
        # system. The legacy mode is local 9DoF; the 15DoF mode is reserved
        # for the final full-sequence backend and couples the same shared 6D
        # block to the existing per-keyframe [T, v, ba, bg] state.
        self.use_joint_shared_bias_ba = bool(use_joint_shared_bias_ba)
        self.use_joint_shared_bias_15dof = bool(use_joint_shared_bias_15dof)
        self.imu_joint_bias_interval = max(1, int(imu_joint_bias_interval))
        self.imu_joint_bias_window = max(2, int(imu_joint_bias_window))
        self.imu_joint_bias_damping = max(0.0, float(imu_joint_bias_damping))
        self.imu_joint_bias_step_scale = max(0.0, float(imu_joint_bias_step_scale))
        self.imu_joint_bias_max_accel_step = max(0.0, float(imu_joint_bias_max_accel_step))
        self.imu_joint_bias_max_gyro_step = max(0.0, float(imu_joint_bias_max_gyro_step))
        self.imu_joint_bias_max_accel_norm = max(0.0, float(imu_joint_bias_max_accel_norm))
        self.imu_joint_bias_max_gyro_norm = max(0.0, float(imu_joint_bias_max_gyro_norm))
        self.imu_joint_bias_min_edges = max(1, int(imu_joint_bias_min_edges))
        self.imu_joint_bias_min_duration = max(0.0, float(imu_joint_bias_min_duration))
        self.imu_joint_bias_min_rotation_deg = max(0.0, float(imu_joint_bias_min_rotation_deg))
        self.imu_joint_bias_max_condition = max(1.0, float(imu_joint_bias_max_condition))
        self.imu_joint_bias_accel_random_walk = float(imu_joint_bias_accel_random_walk)
        self.imu_joint_bias_gyro_random_walk = float(imu_joint_bias_gyro_random_walk)
        self.imu_joint_bias_accel_prior_std = max(0.0, float(imu_joint_bias_accel_prior_std))
        self.imu_joint_bias_gyro_prior_std = max(0.0, float(imu_joint_bias_gyro_prior_std))
        self.imu_joint_bias_min_improvement = max(0.0, float(imu_joint_bias_min_improvement))
        self.imu_joint_bias_relinearize_accel = max(0.0, float(imu_joint_bias_relinearize_accel))
        self.imu_joint_bias_relinearize_gyro = max(0.0, float(imu_joint_bias_relinearize_gyro))
        # A joint shared-bias update must use one CUDA step per
        # linearization: after the step, the recovered bias changes the IMU
        # residual.  This switch exposes that same outer schedule for a
        # fixed-bias control experiment, so the Schur term is the only
        # deliberate difference between the two runs.
        self.imu_ba_outer_relinearize = bool(imu_ba_outer_relinearize)
        self.imu_ba_outer_relinearize_interval = max(
            1, int(imu_ba_outer_relinearize_interval)
        )
        self.imu_ba_outer_window = max(0, int(imu_ba_outer_window))
        self.last_joint_shared_bias_status = {"enabled": 0.0, "reason": "not_run"}
        self._joint_bias_update_count = 0
        self._last_joint_bias_attempt_frame = -self.imu_joint_bias_interval
        self._last_outer_relinearization_frame = -self.imu_ba_outer_relinearize_interval
        self._reported_imu_ba_prior = False
        self._reported_imu_alignment_wait = False
        self.last_imu_confidence = None
        self.imu_ba_debug = bool(imu_ba_debug)
        self.imu_ba_debug_path = imu_ba_debug_path
        self.imu_ba_debug_max_rows = int(imu_ba_debug_max_rows)
        self.imu_ba_debug_stage = str(imu_ba_debug_stage)
        self._imu_ba_debug_rows = []
        self._imu_ba_debug_seen = 0

        # IMU state aliases.
        #
        # Pose/depth are stored in DepthVideo in this codebase, not directly in
        # FactorGraph. Keep aliases here so a future inertial BA implementation
        # can access M_i = (v_i, ba_i, bg_i) and the preintegrated measurements
        # from the same graph object without changing call sites again.
        self.velocities = getattr(video, "velocities", None)
        self.bias_acc = getattr(video, "bias_acc", None)
        self.bias_gyro = getattr(video, "bias_gyro", None)
        self.imu_delta = getattr(video, "imu_delta", None)
        self.imu_valid = getattr(video, "imu_valid", None)
        self.imu_weight = getattr(video, "imu_weight", None)
        self.imu_info = getattr(video, "imu_info", None)
        self.imu_bias_jacobian = getattr(video, "imu_bias_jacobian", None)

        ############################################################
        # feature resolution
        ############################################################

        # DROID는 원본 이미지의 1/8 해상도에서 dense correspondence를 다룬다.
        # 예: 원본 480x640이면 feature resolution은 60x80.
        self.ht = ht = video.ht // 8
        self.wd = wd = video.wd // 8

        # 기준 좌표 grid.
        # shape은 대략 [ht, wd, 2].
        # 각 feature map 위치의 기본 좌표를 나타낸다.
        self.coords0 = pops.coords_grid(ht, wd, device=device)

        ############################################################
        # active factor index
        ############################################################

        # active edge의 source frame index
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)

        # active edge의 target frame index
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)

        # 각 factor가 몇 번 update 되었는지 나타내는 age
        # 오래된 factor 제거 기준으로 사용됨
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        ############################################################
        # factor-specific feature storage
        ############################################################

        # corr:
        #   active factor들의 correlation block
        #
        # net:
        #   source frame의 recurrent hidden state
        #
        # inp:
        #   source frame의 context input feature
        self.corr, self.net, self.inp = None, None, None

        # BA에서 사용하는 damping 값
        # frame별 disparity shape과 같은 형태로 초기화
        self.damping = 1e-6 * torch.ones_like(self.video.disps)

        ############################################################
        # active factor target / weight
        ############################################################

        # target:
        #   각 edge에 대해 source frame point가 target frame에서 있어야 할 목표 좌표.
        #   초기에는 현재 pose/depth로 reproject한 좌표가 들어가고,
        #   update network가 예측한 delta가 더해지며 갱신된다.
        #
        # shape:
        #   [1, num_edges, ht, wd, 2]
        self.target = torch.zeros(
            [1, 0, ht, wd, 2],
            device=device,
            dtype=torch.float
        )

        # weight:
        #   각 target correspondence의 confidence.
        #   BA에서 residual에 곱해지는 신뢰도 역할.
        #
        # shape:
        #   [1, num_edges, ht, wd, 2]
        self.weight = torch.zeros(
            [1, 0, ht, wd, 2],
            device=device,
            dtype=torch.float
        )

        ############################################################
        # inactive / bad factor storage
        ############################################################

        # inactive factor:
        #   현재 active graph에서는 빠졌지만,
        #   store=True로 저장해둔 factor.
        #   update(use_inactive=True)에서 다시 참고될 수 있음.
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)

        # bad factor:
        #   confidence가 낮아 나쁜 edge로 판단된 factor.
        #   다시 추가되지 않도록 기록하는 용도.
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=device)

        # inactive factor의 target과 weight 저장
        self.target_inac = torch.zeros(
            [1, 0, ht, wd, 2],
            device=device,
            dtype=torch.float
        )

        self.weight_inac = torch.zeros(
            [1, 0, ht, wd, 2],
            device=device,
            dtype=torch.float
        )

    def set_imu_ba_stage(self, stage):
        """Switch a local graph between visual-only and full 15DoF DBA."""

        if stage == "pose_only":
            # Stage one must not admit either the legacy fixed-bias 9DoF
            # factor or the post-DBA rotation residual.
            self.use_full_imu_ba = False
            self.use_imu_ba_prior = False
            self.imu_optimize_bias = False
            self.apply_imu_residual = False
            return

        if stage == "full_15dof":
            if not (
                self._configured_full_imu_ba
                and self._configured_imu_ba_prior
                and self._configured_imu_optimize_bias
            ):
                raise RuntimeError(
                    "full_15dof stage requires configured full IMU BA with "
                    "per-keyframe bias optimization"
                )
            self.use_full_imu_ba = True
            self.use_imu_ba_prior = True
            self.imu_optimize_bias = True
            # The full preintegration factor already contains rotation.
            self.apply_imu_residual = False
            self._imu_tuned_confidence_iteration = 0
            return

        raise ValueError(f"unknown IMU BA stage: {stage}")

    def _edge_confidence_to_frame_confidence(
        self, edge_confidence, ii=None, jj=None, return_coverage=False
    ):
        """
        Convert update-head confidence maps into temporal IMU-factor confidence.

        A full IMU factor at keyframe ``j`` represents exactly the interval
        ``(j-1, j)``.  We therefore use only direct visual graph edges for that
        interval instead of averaging every edge that happens to touch either
        frame.  The returned `[B, N]` layout remains compatible with existing
        factor code: column ``j`` is the confidence for the factor ending at
        ``j``.
        """

        if edge_confidence is None:
            return (None, None) if return_coverage else None

        ii = self.ii if ii is None else ii
        jj = self.jj if jj is None else jj

        if ii.numel() == 0 or jj.numel() == 0:
            return (None, None) if return_coverage else None

        num_frames = max(
            int(self.video.counter.value),
            int(torch.maximum(ii.max(), jj.max()).item()) + 1,
        )
        if self.use_learned_imu_precision:
            quality_fallback = self._temporal_imu_quality_precision(num_frames)
            frame_confidence, coverage = temporal_imu_precision_from_edges(
                edge_confidence.to(device=self.device, dtype=torch.float),
                ii,
                jj,
                num_frames=num_frames,
                fallback=quality_fallback if quality_fallback is not None else 1.0,
                return_coverage=True,
            )
        else:
            quality_fallback = self._temporal_imu_quality_confidence(num_frames)
            frame_confidence, coverage = temporal_imu_confidence_from_edges(
                edge_confidence.to(device=self.device, dtype=torch.float),
                ii,
                jj,
                num_frames=num_frames,
                fallback=quality_fallback if quality_fallback is not None else 1.0,
                return_coverage=True,
            )

            if self.imu_confidence_floor > 0.0:
                floor = float(self.imu_confidence_floor)
                frame_confidence = floor + (1.0 - floor) * frame_confidence

            frame_confidence = frame_confidence.clamp(0.0, 1.0)
        if return_coverage:
            return frame_confidence, coverage
        return frame_confidence

    def _persist_imu_confidence(self, frame_confidence, coverage):
        """Cache only direct temporal predictions for terminal VI refinement."""

        if (
            self.use_learned_imu_precision
            or frame_confidence is None
            or coverage is None
        ):
            return 0

        if not hasattr(self.video, "update_imu_learned_confidence"):
            return 0

        try:
            return self.video.update_imu_learned_confidence(
                frame_confidence,
                coverage=coverage,
                count=int(self.video.counter.value),
            )
        except Exception as exc:
            print(f"[IMU-CONFIDENCE WARNING] cache update failed: {exc}")
            return 0

    def _edge_imu_quality(self, ii, jj):
        """Return direct temporal IMU quality descriptors for GRU edges."""

        # Scalar DVI confidence and component confidence use the same
        # preintegrated IMU descriptor.  Previously this branch admitted only
        # component mode, so a runtime DVI-quality-fusion run silently passed
        # ``None`` to the MLP even though training supplied its seven IMU
        # quality features.  That creates a train/runtime input mismatch and
        # turns the learned scalar quality contribution into an all-zero logit.
        if not (
            self.use_learned_imu_precision
            or self.use_component_imu_confidence
            or self.use_dvi_imu_quality_fusion
        ):
            return None
        count = max(
            int(self.video.counter.value),
            int(torch.maximum(ii.max(), jj.max()).item()) + 1 if ii.numel() else 0,
        )
        if count <= 0:
            return None
        return edge_imu_quality_features(
            self.video.imu_delta[:count],
            self.video.imu_info[:count],
            ii,
            jj,
            num_frames=count,
        )

    def _temporal_imu_quality_confidence(self, count):
        """Predict fallback confidence for temporal rows lacking visual edges.

        The direct visual-GRU branch remains preferable when a temporal edge
        exists.  For an uncovered but valid interval, the raw preintegration
        quality MLP supplies either one scalar (DVI mode) or three
        [rotation, velocity, position] confidences (component mode).  This
        prevents a missing visual edge from silently becoming fixed c=1.
        """

        if (
            count <= 0
            or not (
                self.use_component_imu_confidence
                or self.use_dvi_imu_quality_fusion
            )
        ):
            return None
        predictor_name = (
            "component_imu_quality_confidence"
            if self.use_component_imu_confidence
            else "dvi_imu_quality_confidence"
        )
        predictor = getattr(self.update_op, predictor_name, None)
        if predictor is None:
            return None
        quality = temporal_imu_quality_features(
            self.video.imu_delta[:count],
            self.video.imu_info[:count],
            num_frames=count,
        )
        if quality is None:
            return None
        confidence = predictor(quality)
        if confidence is None:
            return None
        if confidence.ndim == 2:
            confidence = confidence.unsqueeze(-1)
        if (
            confidence.ndim != 3
            or confidence.shape[0] != 1
            or confidence.shape[1] < count
            or confidence.shape[-1]
            != (3 if self.use_component_imu_confidence else 1)
        ):
            raise ValueError(
                "IMU quality fallback must be [1, N, 1|3], got "
                f"{tuple(confidence.shape)} for N={count}"
            )
        return confidence[:, :count]

    def _temporal_imu_quality_precision(self, count):
        """Positive R/v/p fallback weights for direct-precision mode.

        A direct temporal visual edge remains preferred.  This fallback only
        covers valid IMU intervals that have no active visual edge, and uses
        exactly the same seven preintegration features as the GRU fusion path.
        """

        if count <= 0 or not self.use_learned_imu_precision:
            return None
        predictor = getattr(self.update_op, "imu_precision_quality_weights", None)
        if predictor is None:
            return None
        quality = temporal_imu_quality_features(
            self.video.imu_delta[:count],
            self.video.imu_info[:count],
            num_frames=count,
        )
        if quality is None:
            return None
        precision = predictor(quality)
        if (
            precision is None
            or precision.ndim != 3
            or precision.shape[0] != 1
            or precision.shape[1] < count
            or precision.shape[-1] != 3
        ):
            raise ValueError(
                "IMU precision quality fallback must be [1, N, 3], got "
                f"{None if precision is None else tuple(precision.shape)} for N={count}"
            )
        # Keep the BA-side invariant even if a stale or externally loaded
        # checkpoint predates the bounded attenuation parameterization.
        return precision[:, :count].clamp(1e-12, 1.0)

    def _apply_imu_residual(self, stage):
        if self.imu_regularizer is None or not self.apply_imu_residual:
            return

        try:
            imu_confidence = (
                self.last_imu_confidence
                if self.use_learned_imu_confidence
                else None
            )
            applied = self.imu_regularizer.apply(
                self.video,
                stage=stage,
                imu_confidence=imu_confidence,
            )
            if applied > 0 and self.video.counter.value < 30:
                print(f"[IMU-RESIDUAL] stage={stage}, applied={applied}")
        except Exception as e:
            print(f"[IMU-RESIDUAL WARNING] stage={stage}, failed: {e}")

    def _confidence_components_for_pose_index(self, pose_ix):
        """Return `[rotation, velocity, position]` confidence for one row."""

        if self.imu_tuned_confidence_schedule is not None:
            schedule_ix = min(
                self._imu_tuned_confidence_iteration,
                len(self.imu_tuned_confidence_schedule) - 1,
            )
            value = self.imu_tuned_confidence_schedule[schedule_ix]
            return np.full(3, value, dtype=np.float64)

        if not self.use_learned_imu_prediction or self.last_imu_confidence is None:
            return np.ones(3, dtype=np.float64)

        try:
            conf = self.last_imu_confidence
            if conf.ndim == 3:
                value = conf[0, pose_ix]
            elif (
                conf.ndim == 2
                and self.use_component_imu_confidence
                and conf.shape[-1] == 3
                and conf.shape[0] > pose_ix
            ):
                # Compatibility for an unbatched component cache [N, 3].
                # A scalar cache is [B, N]; do not misread its N=3 case as
                # three components.
                value = conf[pose_ix]
            elif conf.ndim == 2:
                value = conf[0, pose_ix]
            else:
                value = conf.reshape(-1)[pose_ix]
            values = np.asarray(value.detach().cpu().reshape(-1).tolist(), dtype=np.float64)
        except Exception:
            return np.ones(3, dtype=np.float64)

        if values.size == 1:
            values = np.repeat(values, 3)
        if values.size < 3 or not np.all(np.isfinite(values[:3])):
            return np.ones(3, dtype=np.float64)

        if self.use_learned_imu_precision:
            return np.clip(values[:3], 1e-12, 1.0)
        return np.clip(values[:3], 0.0, 1.0)

    def _confidence_for_pose_index(self, pose_ix):
        """Legacy scalar confidence for rotation-only residual consumers."""

        return float(self._confidence_components_for_pose_index(pose_ix)[0])

    def _imu_confidence_mode(self):
        if self.imu_tuned_confidence_schedule is not None:
            return "tuned"
        if self.use_learned_imu_precision:
            return "learned_precision"
        if self.use_learned_imu_confidence:
            return "learned"
        return "fixed"

    def _current_tuned_confidence_value(self):
        if self.imu_tuned_confidence_schedule is None:
            return 1.0
        schedule_ix = min(
            self._imu_tuned_confidence_iteration,
            len(self.imu_tuned_confidence_schedule) - 1,
        )
        return float(self.imu_tuned_confidence_schedule[schedule_ix])

    def _advance_tuned_confidence_schedule(self):
        """Advance after one full outer IMU DBA update, if one actually ran."""

        if (
            self.imu_tuned_confidence_schedule is not None
            and self.use_full_imu_ba
            and self.use_imu_ba_prior
            and (not self.require_imu_alignment or self._has_imu_alignment())
        ):
            self._imu_tuned_confidence_iteration += 1

    def _visual_confidence_for_pose_pair(self, prev_ix, curr_ix):
        """Return the visual DBA confidence around one temporal IMU edge."""

        if (
            self.weight is None
            or self.weight.numel() == 0
            or self.ii.numel() == 0
            or self.jj.numel() == 0
        ):
            return 1.0

        try:
            direct = (
                ((self.ii == prev_ix) & (self.jj == curr_ix))
                | ((self.ii == curr_ix) & (self.jj == prev_ix))
            )
            touching = (
                (self.ii == prev_ix)
                | (self.ii == curr_ix)
                | (self.jj == prev_ix)
                | (self.jj == curr_ix)
            )
            mask = direct if bool(direct.any().detach().cpu().item()) else touching
            if not bool(mask.any().detach().cpu().item()):
                return 1.0
            value = self.weight[:, mask].detach().float().mean()
            out = float(value.cpu().item())
        except Exception:
            return 1.0

        return float(np.clip(out, 0.0, 1.0)) if np.isfinite(out) else 1.0

    def _adaptive_imu_gate(
        self,
        prev_ix,
        curr_ix,
        dr_norm,
        residual_deg,
        residual_pos_norm,
        residual_vel_norm,
    ):
        """Compute a bounded runtime IMU multiplier for one temporal factor.

        The learned IMU confidence remains the primary learned signal. This
        deterministic safety gate reduces its influence when visual and IMU
        motion disagree, while increasing it only in bounded weak-visual or
        fast-rotation cases.
        """

        visual_confidence = self._visual_confidence_for_pose_pair(prev_ix, curr_ix)
        parts = {
            "visual_confidence": visual_confidence,
            "visual_gate": 1.0,
            "motion_gate": 1.0,
            "consistency_gate": 1.0,
            "adaptive_gate": 1.0,
        }
        if not self.use_adaptive_imu_gate:
            return 1.0, parts

        visual_thresh = self.imu_gate_visual_thresh
        if visual_thresh > 0.0 and visual_confidence < visual_thresh:
            deficit = (visual_thresh - visual_confidence) / max(visual_thresh, 1e-6)
            parts["visual_gate"] = 1.0 + self.imu_gate_visual_boost * float(
                np.clip(deficit, 0.0, 1.0)
            )

        motion_deg = float(dr_norm) * 180.0 / np.pi
        motion_thresh = self.imu_gate_motion_thresh_deg
        if motion_thresh > 0.0 and motion_deg > motion_thresh:
            excess = (motion_deg - motion_thresh) / max(motion_thresh, 1e-6)
            parts["motion_gate"] = 1.0 + self.imu_gate_motion_boost * float(
                np.clip(excess, 0.0, 1.0)
            )

        def attenuation(value, soft, hard):
            if soft <= 0.0:
                return 1.0
            hard = max(hard, soft + 1e-6)
            alpha = float(np.clip((value - soft) / (hard - soft), 0.0, 1.0))
            return 1.0 + alpha * (self.imu_gate_min - 1.0)

        consistency = attenuation(
            residual_deg,
            self.imu_gate_residual_soft_deg,
            self.imu_gate_residual_hard_deg,
        )
        if self.imu_gate_residual_pos_soft > 0.0:
            consistency *= attenuation(
                residual_pos_norm,
                self.imu_gate_residual_pos_soft,
                2.0 * self.imu_gate_residual_pos_soft,
            )
        if self.imu_gate_residual_vel_soft > 0.0:
            consistency *= attenuation(
                residual_vel_norm,
                self.imu_gate_residual_vel_soft,
                2.0 * self.imu_gate_residual_vel_soft,
            )
        parts["consistency_gate"] = float(np.clip(consistency, self.imu_gate_min, 1.0))

        gate = (
            parts["visual_gate"]
            * parts["motion_gate"]
            * parts["consistency_gate"]
        )
        parts["adaptive_gate"] = float(
            np.clip(gate, self.imu_gate_min, self.imu_gate_max)
        )
        return parts["adaptive_gate"], parts

    def _record_imu_ba_debug(self, row):
        if not self.imu_ba_debug:
            return

        self._imu_ba_debug_seen += 1
        if len(self._imu_ba_debug_rows) >= self.imu_ba_debug_max_rows:
            return

        clean = {
            "stage": self.imu_ba_debug_stage,
            "confidence_mode": self._imu_confidence_mode(),
            "confidence_iteration": int(self._imu_tuned_confidence_iteration),
            "confidence_schedule_value": self._current_tuned_confidence_value(),
        }
        for key, value in row.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
            clean[key] = value
        self._imu_ba_debug_rows.append(clean)

    def _flush_imu_ba_debug(self):
        if not self.imu_ba_debug or len(self._imu_ba_debug_rows) == 0:
            return

        if self.imu_ba_debug_path in (None, ""):
            path = Path("outputs") / "imu_ba_debug.csv"
        else:
            path = Path(self.imu_ba_debug_path)

        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in self._imu_ba_debug_rows for key in row.keys()})
        write_header = not path.exists()

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(self._imu_ba_debug_rows)

        print(
            f"[IMU-BA] wrote {len(self._imu_ba_debug_rows)} debug rows "
            f"to {path} (seen={self._imu_ba_debug_seen})"
        )
        self._imu_ba_debug_rows.clear()

    def _build_imu_ba_pose_prior(self, t0, t1):
        """
        Build an external 6D pose-prior normal equation from preintegrated IMU.

        This is the first in-DBA E^u path. It keeps the existing CUDA solver's
        state block as pose+depth, then adds a rotation-only IMU contribution to
        the pose Hessian/gradient before Schur complement solve:

            r_R = Log((q_i * Deltaq_imu)^-1 * q_j)

        Velocity and bias states are intentionally not optimized here yet; those
        require expanding the CUDA block dimension from 6D pose to 15D inertial
        state in the next stage.
        """

        if (
            not self.use_imu_ba_prior
            or self.imu_regularizer is None
            or self.imu_ba_prior_weight <= 0.0
        ):
            return None, None, None, None

        # VI alignment must start from a visual trajectory.  Feeding the older
        # rotation-only prior before scale/gravity/bias are observable quietly
        # turns the initializer into a self-fulfilling IMU fit and can hide a
        # bad initial bias/extrinsic.  Full 9DoF factors are gated below for
        # the same reason.
        if self.require_imu_alignment and not self._has_imu_alignment():
            return None, None, None, None

        t0 = int(t0)
        t1 = int(t1)
        P = t1 - t0
        if P <= 0:
            return None, None, None, None

        n = int(self.video.counter.value)
        if n < 2:
            return None, None, None, None

        poses = self.video.poses
        device = poses.device
        dtype = poses.dtype
        max_pose_ix = min(t1, n)

        H_blocks = []
        H_ii = []
        H_jj = []
        v_prior = torch.zeros((P, 6), device=device, dtype=dtype)
        eye3 = torch.eye(3, device=device, dtype=dtype)
        rad_to_deg = 180.0 / np.pi
        used = 0
        confidence_sum = 0.0

        def add_rot_block(i_local, j_local, value):
            H = torch.zeros((6, 6), device=device, dtype=dtype)
            H[3:6, 3:6] = value
            H_blocks.append(H)
            H_ii.append(int(i_local))
            H_jj.append(int(j_local))

        for pose_ix in range(max(1, t0), max_pose_ix):
            prev_ix = pose_ix - 1
            curr_ix = pose_ix
            prev_local = prev_ix - t0
            curr_local = curr_ix - t0

            if curr_local < 0 or curr_local >= P:
                continue

            prev_stamp = self.video.tstamp[prev_ix].detach().cpu().item()
            curr_stamp = self.video.tstamp[curr_ix].detach().cpu().item()
            q_imu_np, row_scale, info = self.imu_regularizer._compose_prior_between(
                prev_stamp,
                curr_stamp,
            )
            if q_imu_np is None or row_scale <= 0.0:
                continue

            q_i = poses[prev_ix, 3:7].detach()
            q_j = poses[curr_ix, 3:7].detach()
            q_imu = torch.as_tensor(q_imu_np, device=device, dtype=dtype)

            if self.imu_regularizer.compose_order == "prev_dq":
                q_pred = quat_multiply(q_i, q_imu)
            elif self.imu_regularizer.compose_order == "dq_prev":
                q_pred = quat_multiply(q_imu, q_i)
            else:
                continue

            q_err = quat_multiply(quat_inverse(q_pred), q_j)
            residual = quat_to_rotvec(q_err).to(dtype=dtype)
            residual_norm = torch.linalg.norm(residual)

            if not torch.isfinite(residual_norm):
                continue

            residual_deg = float(residual_norm.detach().cpu().item() * rad_to_deg)
            if (
                self.imu_ba_prior_max_deg > 0.0
                and residual_deg > self.imu_ba_prior_max_deg
            ):
                continue

            confidence = self._confidence_for_pose_index(curr_ix)
            adaptive_gate, _ = self._adaptive_imu_gate(
                prev_ix,
                curr_ix,
                dr_norm=float(torch.linalg.norm(residual).detach().cpu().item()),
                residual_deg=residual_deg,
                residual_pos_norm=0.0,
                residual_vel_norm=0.0,
            )
            weight = (
                float(self.imu_ba_prior_weight)
                * float(row_scale)
                * confidence
                * adaptive_gate
            )
            if weight <= 0.0 or not np.isfinite(weight):
                continue

            w = torch.as_tensor(weight, device=device, dtype=dtype)
            R_i = quat_to_rotmat(q_i)
            R_pred = quat_to_rotmat(q_pred)
            Jl_inv = so3_left_jacobian_inv(residual)

            if self.imu_regularizer.compose_order == "dq_prev":
                rot_i_map = R_i.transpose(0, 1)
            else:
                rot_i_map = R_pred.transpose(0, 1)

            Ji_rot = Jl_inv @ rot_i_map
            Jj_rot = -(Jl_inv @ R_pred.transpose(0, 1))

            Hii = w * (Ji_rot.transpose(0, 1) @ Ji_rot)
            Hij = w * (Ji_rot.transpose(0, 1) @ Jj_rot)
            Hji = w * (Jj_rot.transpose(0, 1) @ Ji_rot)
            Hjj = w * (Jj_rot.transpose(0, 1) @ Jj_rot)
            vi = w * (Ji_rot.transpose(0, 1) @ residual)
            vj = w * (Jj_rot.transpose(0, 1) @ residual)

            if 0 <= prev_local < P:
                add_rot_block(prev_local, prev_local, Hii)
                add_rot_block(prev_local, curr_local, Hij)
                add_rot_block(curr_local, prev_local, Hji)
                v_prior[prev_local, 3:6] += vi

            add_rot_block(curr_local, curr_local, Hjj)
            v_prior[curr_local, 3:6] += vj
            used += 1
            confidence_sum += confidence

        if used == 0 or len(H_blocks) == 0:
            return None, None, None, None

        if not self._reported_imu_ba_prior:
            conf_mean = confidence_sum / max(used, 1)
            print(
                f"[IMU-BA] pose prior enabled: window=[{t0},{t1}), "
                f"edges={used}, blocks={len(H_blocks)}, "
                f"weight={self.imu_ba_prior_weight}, conf_mean={conf_mean:.4f}"
            )
            self._reported_imu_ba_prior = True

        pose_prior_H = torch.stack(H_blocks, dim=0).contiguous()
        pose_prior_ii = torch.as_tensor(H_ii, device=device, dtype=torch.long).contiguous()
        pose_prior_jj = torch.as_tensor(H_jj, device=device, dtype=torch.long).contiguous()
        return pose_prior_H, v_prior.contiguous(), pose_prior_ii, pose_prior_jj

    def _global_imu_bias(self, value, device, dtype):
        if value is None:
            return torch.zeros(3, device=device, dtype=dtype)

        if torch.is_tensor(value):
            tensor = value.detach().to(device=device, dtype=dtype)
        else:
            tensor = torch.as_tensor(value, device=device, dtype=dtype)

        if tensor.numel() < 3:
            return torch.zeros(3, device=device, dtype=dtype)

        return tensor.reshape(-1)[:3]

    def _has_imu_alignment(self):
        ready = getattr(self.video, "imu_alignment_ready", None)
        if ready is None:
            return False
        if hasattr(ready, "value"):
            return bool(ready.value)
        if torch.is_tensor(ready):
            return bool(ready.detach().reshape(-1)[0].cpu().item())
        return bool(ready)

    def _aligned_or_default_bias(self, aligned_attr, default, device, dtype):
        if self._has_imu_alignment():
            aligned = getattr(self.video, aligned_attr, None)
            if aligned is not None:
                return self._global_imu_bias(aligned, device, dtype)
        return self._global_imu_bias(default, device, dtype)

    def _runtime_imu_gravity(self, device, dtype):
        # A visual-inertial alignment writes gravity in the current DROID world
        # unit. Before that, a CLI gravity value is interpreted in metres and
        # converted through imu_unit_scale for the legacy path.
        if self._has_imu_alignment():
            gravity = getattr(self.video, "imu_gravity", None)
            if gravity is not None:
                return self._global_imu_bias(gravity, device, dtype)
        return self._global_imu_bias(self.imu_gravity, device, dtype) * self._imu_metric_scale(device, dtype)

    def _imu_metric_scale(self, device, dtype):
        scale = getattr(self.video, "imu_unit_scale", None)
        if scale is None:
            return torch.ones((), device=device, dtype=dtype)

        if torch.is_tensor(scale):
            return scale.detach().to(device=device, dtype=dtype).reshape(-1)[0]

        return torch.as_tensor(float(scale), device=device, dtype=dtype)

    def _imu_info_reference(self, t0, max_pose_ix, device, dtype):
        if (
            not self.use_imu_info_weighting
            or getattr(self.video, "imu_info", None) is None
            or max_pose_ix <= max(1, int(t0))
        ):
            return None

        configured_reference = self.imu_info_reference
        if configured_reference is not None:
            reference = torch.as_tensor(
                configured_reference,
                device=device,
                dtype=dtype,
            ).reshape(-1)
            if reference.numel() >= 3:
                reference = reference[:3].clone()
                valid_reference = torch.isfinite(reference) & (
                    reference > self.imu_info_weight_eps
                )
                if bool(valid_reference.all().detach().cpu().item()):
                    # Translation information is stored in DROID's current
                    # visual unit, while the session reference starts in the
                    # physical IMU unit used by imu_prior.csv.
                    metric_scale = self._imu_metric_scale(device, dtype).clamp_min(1e-12)
                    reference[1:3] /= metric_scale * metric_scale
                    self._imu_info_reference_mode = "raw_camera"
                    return reference

        info = self.video.imu_info[max(1, int(t0)) : max_pose_ix].detach().to(
            device=device,
            dtype=dtype,
        )
        if info.numel() == 0 or info.shape[-1] < 3:
            return None

        valid = torch.isfinite(info) & (info > self.imu_info_weight_eps)
        if not bool(valid.any().detach().cpu().item()):
            return None

        ref = torch.ones(3, device=device, dtype=dtype)
        for k in range(3):
            values = info[:, k][valid[:, k]]
            if values.numel() > 0:
                ref[k] = values.median().clamp_min(self.imu_info_weight_eps)

        self._imu_info_reference_mode = "window_median"
        return ref

    def _imu_info_term_scale(self, curr_ix, info_ref, device, dtype):
        if (
            info_ref is None
            or getattr(self.video, "imu_info", None) is None
            or curr_ix < 0
            or curr_ix >= int(self.video.imu_info.shape[0])
        ):
            return torch.ones(15, device=device, dtype=dtype), (1.0, 1.0, 1.0)

        info = self.video.imu_info[curr_ix].detach().to(device=device, dtype=dtype)
        if info.numel() < 3:
            return torch.ones(15, device=device, dtype=dtype), (1.0, 1.0, 1.0)

        info = info.reshape(-1)[:3]
        valid = torch.isfinite(info) & (info > self.imu_info_weight_eps)
        ratio = torch.ones(3, device=device, dtype=dtype)
        ratio[valid] = info[valid] / info_ref[valid].clamp_min(self.imu_info_weight_eps)

        if self.imu_info_weight_clip > 0.0:
            ratio = ratio.clamp(
                min=1.0 / self.imu_info_weight_clip,
                max=self.imu_info_weight_clip,
            )
        else:
            ratio = ratio.clamp_min(0.0)

        info_scale = ratio.sqrt()
        rot_scale, vel_scale, pos_scale = info_scale[0], info_scale[1], info_scale[2]
        term_scale = torch.cat(
            [
                pos_scale.expand(3),
                vel_scale.expand(3),
                rot_scale.expand(3),
                torch.ones(6, device=device, dtype=dtype),
            ],
            dim=0,
        )
        return term_scale, (
            float(rot_scale.detach().cpu().item()),
            float(vel_scale.detach().cpu().item()),
            float(pos_scale.detach().cpu().item()),
        )

    def _imu_full_covariance_term_scale(self, curr_ix, info_ref, device, dtype):
        """Return p/v/R row scales in one rotation-information reference.

        ``imu_info`` contains the diagonal information for [R, v, p] after
        the translation rows have been converted into the current DROID unit.
        The historical implementation divided each kind by *its own* median
        and then multiplied p/v by arbitrary 0.05 weights.  That removes the
        covariance relation between position, velocity, and rotation.

        Here every row is instead normalized by the calibrated rotation
        information reference.  The result remains dimensionless relative to
        DROID's visual normal equations (so this is not a new global lambda),
        but preserves the measured p/v/R covariance ratios.  The source
        preintegrator currently provides a diagonal approximation, hence the
        name deliberately says "term" rather than claiming a dense covariance
        whitening.
        """
        if (
            info_ref is None
            or getattr(self.video, "imu_info", None) is None
            or curr_ix < 0
            or curr_ix >= int(self.video.imu_info.shape[0])
            or info_ref.numel() < 3
        ):
            return None, None

        info = self.video.imu_info[curr_ix].detach().to(device=device, dtype=dtype)
        info = info.reshape(-1)[:3]
        if info.numel() < 3:
            return None, None

        rotation_ref = info_ref.reshape(-1)[0]
        valid = (
            torch.isfinite(info).all()
            & torch.isfinite(rotation_ref)
            & (info > self.imu_info_weight_eps).all()
            & (rotation_ref > self.imu_info_weight_eps)
        )
        if not bool(valid.detach().cpu().item()):
            return None, None

        # imu_info is [R, v, p], while the factor residual is [p, v, R].
        row_scale = (info / rotation_ref).clamp_min(0.0).sqrt()
        rot_scale, vel_scale, pos_scale = row_scale[0], row_scale[1], row_scale[2]
        term_scale = torch.cat(
            [
                pos_scale.expand(3),
                vel_scale.expand(3),
                rot_scale.expand(3),
                torch.ones(6, device=device, dtype=dtype),
            ],
            dim=0,
        )
        return term_scale, (
            float(rot_scale.detach().cpu().item()),
            float(vel_scale.detach().cpu().item()),
            float(pos_scale.detach().cpu().item()),
        )

    def refine_shared_bias_block(
        self,
        *,
        window=30,
        max_dt=0.5,
        damping=1e-4,
        step_scale=0.5,
        max_accel_step=0.15,
        max_gyro_step=0.015,
        max_accel_norm=0.5,
        max_gyro_norm=0.1,
        robust_delta=3.0,
        min_edges=8,
        min_duration=0.5,
        min_rotation_deg=2.0,
        max_condition=1e8,
        min_relative_improvement=1e-7,
        prior_elapsed_sec=None,
        accel_random_walk=-1.0,
        gyro_random_walk=-1.0,
        accel_prior_std=0.05,
        gyro_prior_std=0.005,
    ):
        """Solve the shared 6D bias block for the current local IMU window.

        The surrounding caller runs another ordinary 9DoF CUDA DBA pass after
        an accepted update. This creates a bounded block-coordinate solve over
        pose/velocity/depth and one shared accelerometer/gyroscope bias pair
        without allocating two additional bias vectors for every keyframe.
        """

        if not self.use_full_imu_ba or not self.use_imu_ba_prior:
            return {"updated": 0.0, "reason": "full_imu_disabled"}
        if self.require_imu_alignment and not self._has_imu_alignment():
            return {"updated": 0.0, "reason": "not_aligned"}

        count = int(self.video.counter.value)
        if count < 2:
            return {"updated": 0.0, "reason": "insufficient_frames"}

        from vi_initialization import refine_window_biases

        start = max(0, count - max(2, int(window)))
        info_reference = self._imu_info_reference(
            start,
            count,
            self.video.poses.device,
            self.video.poses.dtype,
        )
        confidence = (
            self.last_imu_confidence if self.use_learned_imu_confidence else None
        )
        # The legacy shared-bias refiner has one scalar likelihood per edge.
        # Use the component mean there; full DBA still receives all three
        # component weights directly.
        if confidence is not None and confidence.ndim == 3:
            confidence = confidence.mean(dim=-1)
        return refine_window_biases(
            self.video,
            compose_order=self.imu_full_compose_order,
            inverse_imu=self.imu_full_inverse,
            window=window,
            max_dt=max_dt,
            pos_weight=self.imu_full_pos_weight,
            vel_weight=self.imu_full_vel_weight,
            damping=damping,
            step_scale=step_scale,
            max_accel_step=max_accel_step,
            max_gyro_step=max_gyro_step,
            max_accel_norm=max_accel_norm,
            max_gyro_norm=max_gyro_norm,
            global_weight=self.imu_ba_prior_weight,
            imu_confidence=confidence,
            use_imu_info_weighting=self.use_imu_info_weighting,
            imu_info_reference=info_reference,
            imu_info_weight_clip=self.imu_info_weight_clip,
            imu_info_weight_eps=self.imu_info_weight_eps,
            robust_delta=robust_delta,
            min_edges=min_edges,
            min_duration=min_duration,
            min_rotation_deg=min_rotation_deg,
            max_condition=max_condition,
            min_relative_improvement=min_relative_improvement,
            prior_elapsed_sec=prior_elapsed_sec,
            accel_random_walk=accel_random_walk,
            gyro_random_walk=gyro_random_walk,
            accel_prior_std=accel_prior_std,
            gyro_prior_std=gyro_prior_std,
        )

    def _compose_adjacent_video_imu_delta(self, first_ix, second_ix):
        """
        Compose video IMU deltas when the middle keyframe is removed.

        `imu_delta[k]` stores the preintegrated interval from keyframe k-1 to k.
        If keyframe B at `first_ix` is removed from A-B-C, old rows
        `first_ix` (A->B) and `second_ix` (B->C) must become one A->C row.
        """

        if (
            getattr(self.video, "imu_delta", None) is None
            or getattr(self.video, "imu_valid", None) is None
            or first_ix <= 0
            or second_ix >= int(self.video.counter.value)
        ):
            return None

        delta_ab = self.video.imu_delta[first_ix].detach().clone()
        delta_bc = self.video.imu_delta[second_ix].detach().clone()
        if delta_ab.numel() < 10 or delta_bc.numel() < 10:
            return None

        q_ab = rotvec_to_quat(delta_ab[1:4])
        q_bc = rotvec_to_quat(delta_bc[1:4])
        q_ac = quat_multiply(q_ab, q_bc)

        dt_bc = delta_bc[0].clamp_min(0.0)
        dv_ab = delta_ab[4:7]
        dv_bc = delta_bc[4:7]
        dp_ab = delta_ab[7:10]
        dp_bc = delta_bc[7:10]

        delta_ac = delta_ab.clone()
        delta_ac[0] = delta_ab[0].clamp_min(0.0) + dt_bc
        delta_ac[1:4] = quat_to_rotvec(q_ac)
        delta_ac[4:7] = dv_ab + quat_rotate(q_ab, dv_bc)
        delta_ac[7:10] = dp_ab + dv_ab * dt_bc + quat_rotate(q_ab, dp_bc)

        valid = self.video.imu_valid[first_ix] & self.video.imu_valid[second_ix]
        weight = torch.minimum(
            self.video.imu_weight[first_ix],
            self.video.imu_weight[second_ix],
        )
        used_steps = self.video.imu_used_steps[first_ix] + self.video.imu_used_steps[second_ix]

        info = torch.zeros_like(self.video.imu_info[first_ix])
        info_ab = self.video.imu_info[first_ix].detach()
        info_bc = self.video.imu_info[second_ix].detach()
        has_info = (info_ab > 1e-18) & (info_bc > 1e-18)
        var_sum = torch.zeros_like(info_ab)
        var_sum[has_info] = 1.0 / info_ab[has_info] + 1.0 / info_bc[has_info]
        info[has_info] = 1.0 / var_sum[has_info].clamp_min(1e-18)

        bias_jac_ac = None
        if getattr(self.video, "imu_bias_jacobian", None) is not None:
            dst = torch.as_tensor(
                [int(first_ix), int(second_ix)],
                device=delta_ab.device,
                dtype=torch.long,
            )
            dt_edges = torch.stack(
                [delta_ab[0].clamp_min(0.0), delta_bc[0].clamp_min(0.0)]
            ).view(1, 2, 1)
            bias_jac = select_imu_bias_jacobian(
                self.video.imu_bias_jacobian,
                dst,
                batch=1,
                num_frames=int(self.video.counter.value),
                device=delta_ab.device,
                dtype=delta_ab.dtype,
                dt=dt_edges,
            )[0]

            # Keep keyframe-removal composition in the same representation as
            # MotionFilter.  `J_R_bg` is stored as a rotation-vector derivative
            # downstream, so mixing a left-tangent rotation formula here with
            # the right-tangent dv/dp terms would create a wrong bias Jacobian
            # after a keyframe is removed.  The shared CPU helper applies each
            # row's correction and relinearizes the composed delta directly.
            def as_prior_row(delta, jacobian):
                row = {
                    "dt": float(delta[0].detach().cpu().item()),
                    "dr_x": float(delta[1].detach().cpu().item()),
                    "dr_y": float(delta[2].detach().cpu().item()),
                    "dr_z": float(delta[3].detach().cpu().item()),
                    "dv_x": float(delta[4].detach().cpu().item()),
                    "dv_y": float(delta[5].detach().cpu().item()),
                    "dv_z": float(delta[6].detach().cpu().item()),
                    "dp_x": float(delta[7].detach().cpu().item()),
                    "dp_y": float(delta[8].detach().cpu().item()),
                    "dp_z": float(delta[9].detach().cpu().item()),
                }
                prefixes = ("J_R_bg", "J_v_ba", "J_v_bg", "J_p_ba", "J_p_bg")
                values = jacobian.detach().cpu()
                for prefix_ix, prefix in enumerate(prefixes):
                    for row_ix in range(3):
                        for col_ix in range(3):
                            row[f"{prefix}_{row_ix}{col_ix}"] = float(
                                values[prefix_ix, row_ix, col_ix].item()
                            )
                return row

            composed_rows = compose_imu_prior_rows(
                [
                    as_prior_row(delta_ab, bias_jac[0]),
                    as_prior_row(delta_bc, bias_jac[1]),
                ]
            )
            if composed_rows is not None:
                prefixes = ("J_R_bg", "J_v_ba", "J_v_bg", "J_p_ba", "J_p_bg")
                bias_jac_ac = torch.as_tensor(
                    [
                        [
                            [
                                float(composed_rows[f"{prefix}_{row_ix}{col_ix}"])
                                for col_ix in range(3)
                            ]
                            for row_ix in range(3)
                        ]
                        for prefix in prefixes
                    ],
                    device=delta_ab.device,
                    dtype=delta_ab.dtype,
                )

        bias_random_walk = torch.maximum(
            self.video.imu_bias_random_walk[first_ix],
            self.video.imu_bias_random_walk[second_ix],
        )
        return delta_ac, valid, weight, used_steps, info, bias_jac_ac, bias_random_walk

    def _initialize_runtime_velocity(self, t0, t1):
        if (
            getattr(self.video, "velocities", None) is None
            or getattr(self.video, "imu_delta", None) is None
        ):
            return

        n = int(self.video.counter.value)
        max_pose_ix = min(int(t1), n)
        poses = self.video.poses
        centers = imu_center_from_pose_data(
            poses[:max_pose_ix],
            self.video.imu_camera_translation_in_current_units(),
        )
        velocities = self.video.velocities
        gravity = self._runtime_imu_gravity(poses.device, poses.dtype)

        for pose_ix in range(max(1, int(t0)), max_pose_ix):
            if (
                getattr(self.video, "imu_valid", None) is not None
                and not bool(self.video.imu_valid[pose_ix].detach().cpu().item())
            ):
                continue

            dt = float(self.video.imu_delta[pose_ix, 0].detach().cpu().item())
            if dt <= 1e-4 or not np.isfinite(dt):
                continue

            if self.imu_full_max_dt > 0.0 and dt > self.imu_full_max_dt:
                continue

            dt_t = torch.as_tensor(dt, device=poses.device, dtype=poses.dtype)
            dp = centers[pose_ix] - centers[pose_ix - 1]
            v = (dp - 0.5 * gravity * dt_t * dt_t) / dt_t
            if torch.linalg.norm(velocities[pose_ix - 1]).item() < 1e-9:
                velocities[pose_ix - 1] = v
            if torch.linalg.norm(velocities[pose_ix]).item() < 1e-9:
                velocities[pose_ix] = v

    def _joint_shared_bias_due(self):
        """Return whether the next DBA may update the carried shared bias."""

        if not self._joint_shared_bias_enabled():
            return False
        frame = int(self.video.counter.value)
        return frame - self._last_joint_bias_attempt_frame >= self.imu_joint_bias_interval

    def _joint_shared_bias_enabled(self):
        """Whether either explicitly supported shared-bias formulation is active."""

        return bool(
            getattr(self, "use_joint_shared_bias_ba", False)
            or getattr(self, "use_joint_shared_bias_15dof", False)
        )

    def _outer_imu_ba_relinearization_due(self):
        """Return whether the fixed-bias control should use outer BA steps.

        This is intentionally independent of the joint-bias scheduler.  It
        lets an ablation run reproduce the same one-step/rebuild schedule and
        the same expanded local state window without adding a shared 6D block.
        """

        if not self.imu_ba_outer_relinearize:
            return False
        frame = int(self.video.counter.value)
        return (
            frame - self._last_outer_relinearization_frame
            >= self.imu_ba_outer_relinearize_interval
        )

    @staticmethod
    def _schur_eliminate_shared_bias(Hxb, Hbb, gb):
        """Marginalize one six-variable bias block from a DBA state system.

        ``Hxb[i]`` is the ``state_dim``-by-6 normal-equation cross block for
        state ``i`` and the shared bias. ``state_dim`` is 9 for the legacy
        pose/velocity path and 15 for the two-stage full-state path. With
        DROID's ``H dx = v`` convention, eliminating ``db`` adds the returned
        negative Hessian and gradient corrections to the CUDA state system.
        """

        if Hxb.ndim != 3 or Hxb.shape[-1] != 6 or Hxb.shape[-2] not in (9, 15):
            raise ValueError("Hxb must have shape [window, 9|15, 6]")
        if Hbb.shape != (6, 6) or gb.shape != (6,):
            raise ValueError("shared bias normal block must be 6D")

        hbb_inv_gb = torch.linalg.solve(Hbb, gb)
        # Arrange each H_bx block as six rows so one solve handles every
        # state-coordinate column at once.
        hbb_inv_hbx = torch.linalg.solve(
            Hbb,
            Hxb.permute(2, 0, 1).reshape(6, -1),
        ).reshape(6, Hxb.shape[0], Hxb.shape[1])
        hessian_correction = -torch.einsum(
            "aik,kbj->abij", Hxb, hbb_inv_hbx
        )
        gradient_correction = -torch.einsum("aik,k->ai", Hxb, hbb_inv_gb)
        return hessian_correction, gradient_correction

    def _build_full_imu_ba_prior(
        self,
        t0,
        t1,
        return_joint_info=False,
        enable_joint_bias=None,
        mark_joint_attempt=True,
    ):
        """
        Build an inertial prior for CUDA DBA.

        State layout per frame:
            full: [pose(6), velocity(3), accel_bias(3), gyro_bias(3)]
            fixed-bias: [pose(6), velocity(3)]

        Residual layout:
            full: [r_p, r_v, r_R, r_ba, r_bg]
            fixed-bias: [r_p, r_v, r_R]
        """

        def empty_result():
            result = (None, None, None, None)
            return (*result, None) if return_joint_info else result

        if (
            not self.use_full_imu_ba
            or not self.use_imu_ba_prior
            or self.imu_ba_prior_weight <= 0.0
            or getattr(self.video, "imu_delta", None) is None
        ):
            return empty_result()

        if self.require_imu_alignment and not self._has_imu_alignment():
            if not self._reported_imu_alignment_wait:
                print("[VI-INIT] delaying 9D IMU DBA until visual-inertial alignment succeeds")
                self._reported_imu_alignment_wait = True
            return empty_result()

        t0 = int(t0)
        t1 = int(t1)
        P = t1 - t0
        if P <= 0:
            return empty_result()

        n = int(self.video.counter.value)
        if n < 2:
            return empty_result()

        self._initialize_runtime_velocity(t0, t1)

        poses = self.video.poses
        velocities = self.video.velocities
        bias_acc = self.video.bias_acc
        bias_gyro = self.video.bias_gyro
        imu_delta = self.video.imu_delta
        imu_valid = self.video.imu_valid
        imu_weight = self.video.imu_weight

        device = poses.device
        dtype = poses.dtype
        max_pose_ix = min(t1, n)
        lever_arm = self.video.imu_camera_translation_in_current_units().detach()
        centers = imu_center_from_pose_data(poses[:max_pose_ix], lever_arm)
        optimize_bias = bool(self.imu_optimize_bias)
        state_dim = 15 if optimize_bias else 9
        residual_dim = 15 if optimize_bias else 9

        H_blocks = []
        H_ii = []
        H_jj = []
        v_prior = torch.zeros((P, state_dim), device=device, dtype=dtype)
        eye3 = torch.eye(3, device=device, dtype=dtype)
        # Both supported shared-bias variants are Schur-reduced into the
        # existing CUDA state system. The legacy variant owns 9D [T, v] local
        # states; the final-global variant owns the normal 15D [T, v, ba, bg]
        # states and therefore cannot be replaced by the old 9D path.
        if enable_joint_bias is None:
            enable_joint_bias = self._joint_shared_bias_due()
        legacy_joint_bias = bool(
            self.use_joint_shared_bias_ba
            and not optimize_bias
            and state_dim == 9
        )
        full_joint_bias = bool(
            self.use_joint_shared_bias_15dof
            and optimize_bias
            and state_dim == 15
        )
        joint_bias_enabled = bool(
            enable_joint_bias and (legacy_joint_bias or full_joint_bias)
        )
        if joint_bias_enabled and mark_joint_attempt:
            self._last_joint_bias_attempt_frame = int(self.video.counter.value)
        # In final-global 15DoF mode, the shared variable must see the same
        # complete [t0, t1) state interval as global BA. Restricting it to the
        # last online window would recreate the post-hoc observability issue
        # this path is designed to test. The legacy 9DoF path intentionally
        # retains its bounded local window.
        joint_start_ix = (
            t0
            if full_joint_bias
            else max(t0, max_pose_ix - self.imu_joint_bias_window)
        )
        joint_Hxb = torch.zeros((P, state_dim, 6), device=device, dtype=dtype)
        joint_Hbb_data = torch.zeros((6, 6), device=device, dtype=dtype)
        joint_gb = torch.zeros(6, device=device, dtype=dtype)
        joint_dst = []
        joint_used = 0
        joint_duration = 0.0
        joint_rotation_deg = 0.0
        metric_scale = self._imu_metric_scale(device, dtype)
        # Bias remains in physical IMU units. Translational preintegration
        # Jacobians are already scaled into DROID coordinates by DepthVideo.
        ba_global = self._aligned_or_default_bias(
            "imu_global_bias_acc", self.imu_acc_bias, device, dtype
        )
        bg_global = self._aligned_or_default_bias(
            "imu_global_bias_gyro", self.imu_gyro_bias, device, dtype
        )
        gravity = self._runtime_imu_gravity(device, dtype)
        if self.use_learned_imu_precision or self.use_imu_full_covariance_weighting:
            # p/v/R are already put in one calibrated covariance reference
            # per edge below.  Keeping 0.05 here would reintroduce the old
            # hand-tuned position/velocity suppression.
            base_weights = [1.0] * 9
            compatibility_base_weights = (
                [self.imu_full_pos_weight] * 3
                + [self.imu_full_vel_weight] * 3
                + [1.0] * 3
            )
        else:
            base_weights = (
                [self.imu_full_pos_weight] * 3
                + [self.imu_full_vel_weight] * 3
                + [1.0] * 3
            )
            compatibility_base_weights = list(base_weights)
        if optimize_bias:
            # In calibrated random-walk mode the two bias blocks receive
            # their per-edge scale below.  Do not retain the legacy 0.001
            # coefficient as a hidden second bias prior.
            bias_base_weight = (
                1.0
                if self.use_imu_bias_random_walk
                else self.imu_full_bias_weight
            )
            base_weights += [bias_base_weight] * 3
            base_weights += [bias_base_weight] * 3
            compatibility_base_weights += [bias_base_weight] * 3
            compatibility_base_weights += [bias_base_weight] * 3
        base_scale = torch.as_tensor(base_weights, device=device, dtype=dtype).clamp_min(0.0).sqrt()
        compatibility_base_scale = torch.as_tensor(
            compatibility_base_weights,
            device=device,
            dtype=dtype,
        ).clamp_min(0.0).sqrt()
        info_ref = self._imu_info_reference(t0, max_pose_ix, device, dtype)
        rad_to_deg = 180.0 / np.pi
        used = 0
        confidence_sum = 0.0
        adaptive_gate_sum = 0.0
        skipped = {}

        def mark_skip(reason):
            skipped[reason] = skipped.get(reason, 0) + 1

        def add_block(i_local, j_local, block):
            H_blocks.append(block)
            H_ii.append(int(i_local))
            H_jj.append(int(j_local))

        def record_edge(
            reason,
            prev_ix,
            curr_ix,
            dt_value=0.0,
            row_weight=0.0,
            row_scale=0.0,
            confidence=0.0,
            confidence_rot=1.0,
            confidence_vel=1.0,
            confidence_pos=1.0,
            weight=0.0,
            dr_norm=0.0,
            dv_norm=0.0,
            dp_norm=0.0,
            dv_norm_internal=0.0,
            dp_norm_internal=0.0,
            rp_norm=0.0,
            rv_norm=0.0,
            rrot_deg=0.0,
            rba_norm=0.0,
            rbg_norm=0.0,
            info_rot_scale=1.0,
            info_vel_scale=1.0,
            info_pos_scale=1.0,
            visual_confidence=1.0,
            visual_gate=1.0,
            motion_gate=1.0,
            consistency_gate=1.0,
            adaptive_gate=1.0,
            bias_accel_rw_scale=1.0,
            bias_gyro_rw_scale=1.0,
            bias_rw_mode="legacy_fixed",
        ):
            if reason != "used":
                mark_skip(reason)
            if not self.imu_ba_debug:
                return

            self._record_imu_ba_debug({
                "window_t0": t0,
                "window_t1": t1,
                "prev_ix": int(prev_ix),
                "curr_ix": int(curr_ix),
                "reason": reason,
                "dt": float(dt_value),
                "row_weight": float(row_weight),
                "row_scale": float(row_scale),
                "confidence": float(confidence),
                "confidence_rot": float(confidence_rot),
                "confidence_vel": float(confidence_vel),
                "confidence_pos": float(confidence_pos),
                "weight": float(weight),
                "metric_scale": float(metric_scale.detach().cpu().item()),
                "alignment_ready": int(self._has_imu_alignment()),
                "gravity_x": float(gravity[0].detach().cpu().item()),
                "gravity_y": float(gravity[1].detach().cpu().item()),
                "gravity_z": float(gravity[2].detach().cpu().item()),
                "compose_order": self.imu_full_compose_order,
                "inverse_imu": int(self.imu_full_inverse),
                "position_mode": "imu_center",
                "lever_arm_x": float(lever_arm[0].detach().cpu().item()),
                "lever_arm_y": float(lever_arm[1].detach().cpu().item()),
                "lever_arm_z": float(lever_arm[2].detach().cpu().item()),
                "dr_norm": float(dr_norm),
                "dv_norm": float(dv_norm),
                "dp_norm": float(dp_norm),
                "dv_norm_internal": float(dv_norm_internal),
                "dp_norm_internal": float(dp_norm_internal),
                "r_p_norm": float(rp_norm),
                "r_v_norm": float(rv_norm),
                "r_R_deg": float(rrot_deg),
                "r_ba_norm": float(rba_norm),
                "r_bg_norm": float(rbg_norm),
                "info_rot_scale": float(info_rot_scale),
                "info_vel_scale": float(info_vel_scale),
                "info_pos_scale": float(info_pos_scale),
                "visual_confidence": float(visual_confidence),
                "visual_gate": float(visual_gate),
                "motion_gate": float(motion_gate),
                "consistency_gate": float(consistency_gate),
                "adaptive_gate": float(adaptive_gate),
                "bias_accel_rw_scale": float(bias_accel_rw_scale),
                "bias_gyro_rw_scale": float(bias_gyro_rw_scale),
                "bias_rw_mode": str(bias_rw_mode),
            })

        def debug_norm(value):
            if not self.imu_ba_debug:
                return 0.0
            return float(torch.linalg.norm(value).detach().cpu().item())

        for pose_ix in range(max(1, t0), max_pose_ix):
            prev_ix = pose_ix - 1
            curr_ix = pose_ix
            prev_local = prev_ix - t0
            curr_local = curr_ix - t0

            if curr_local < 0 or curr_local >= P:
                record_edge("outside_window", prev_ix, curr_ix)
                continue

            if not bool(imu_valid[curr_ix].detach().cpu().item()):
                record_edge("invalid", prev_ix, curr_ix)
                continue

            row = imu_delta[curr_ix]
            dt = row[0].clamp_min(0.0)
            dt_value = float(dt.detach().cpu().item())
            if dt_value <= 1e-6:
                record_edge("bad_dt", prev_ix, curr_ix, dt_value=dt_value)
                continue

            if self.imu_full_max_dt > 0.0 and dt_value > self.imu_full_max_dt:
                record_edge("skip_dt", prev_ix, curr_ix, dt_value=dt_value)
                continue

            row_weight = float(imu_weight[curr_ix].detach().cpu().item())
            row_scale = float(np.clip(row_weight / 0.001, 0.0, 1.0))
            if row_scale <= 0.0:
                record_edge(
                    "skip_row_weight",
                    prev_ix,
                    curr_ix,
                    dt_value=dt_value,
                    row_weight=row_weight,
                    row_scale=row_scale,
                )
                continue

            dr_norm = float(torch.linalg.norm(row[1:4]).detach().cpu().item())
            dv_norm_internal = float(torch.linalg.norm(row[4:7]).detach().cpu().item())
            dp_norm_internal = float(torch.linalg.norm(row[7:10]).detach().cpu().item())
            metric_scale_value = max(float(metric_scale.detach().cpu().item()), 1e-12)
            dv_norm = dv_norm_internal / metric_scale_value
            dp_norm = dp_norm_internal / metric_scale_value
            if (
                self.imu_full_max_dv > 0.0
                and dv_norm > self.imu_full_max_dv
            ):
                record_edge(
                    "skip_dv",
                    prev_ix,
                    curr_ix,
                    dt_value=dt_value,
                    row_weight=row_weight,
                    row_scale=row_scale,
                    dr_norm=dr_norm,
                    dv_norm=dv_norm,
                    dp_norm=dp_norm,
                    dv_norm_internal=dv_norm_internal,
                    dp_norm_internal=dp_norm_internal,
                )
                continue

            if (
                self.imu_full_max_dp > 0.0
                and dp_norm > self.imu_full_max_dp
            ):
                record_edge(
                    "skip_dp",
                    prev_ix,
                    curr_ix,
                    dt_value=dt_value,
                    row_weight=row_weight,
                    row_scale=row_scale,
                    dr_norm=dr_norm,
                    dv_norm=dv_norm,
                    dp_norm=dp_norm,
                    dv_norm_internal=dv_norm_internal,
                    dp_norm_internal=dp_norm_internal,
                )
                continue

            v_i = velocities[prev_ix].detach()
            v_j = velocities[curr_ix].detach()
            if optimize_bias:
                ba_i = bias_acc[prev_ix].detach()
                ba_j = bias_acc[curr_ix].detach()
                bg_i = bias_gyro[prev_ix].detach()
                bg_j = bias_gyro[curr_ix].detach()
            else:
                ba_i = torch.zeros(3, device=device, dtype=dtype)
                ba_j = torch.zeros(3, device=device, dtype=dtype)
                bg_i = torch.zeros(3, device=device, dtype=dtype)
                bg_j = torch.zeros(3, device=device, dtype=dtype)

            bias_jac_source = getattr(self.video, "imu_bias_jacobian", None)
            bias_jac = select_imu_bias_jacobian(
                bias_jac_source,
                torch.as_tensor([curr_ix], device=device, dtype=torch.long),
                batch=1,
                num_frames=int(
                    bias_jac_source.shape[0]
                    if bias_jac_source is not None
                    else imu_delta.shape[0]
                ),
                device=device,
                dtype=dtype,
                dt=dt.view(1, 1, 1),
            )[0, 0]
            J_R_bg = bias_jac[0]
            J_v_ba = bias_jac[1]
            J_v_bg = bias_jac[2]
            J_p_ba = bias_jac[3]
            J_p_bg = bias_jac[4]
            try:
                residual, Ji, Jj, factor_details = full_imu_factor_linearization(
                    poses[prev_ix].detach(),
                    poses[curr_ix].detach(),
                    v_i,
                    v_j,
                    ba_i,
                    ba_j,
                    bg_i,
                    bg_j,
                    ba_global,
                    bg_global,
                    row,
                    bias_jac,
                    gravity,
                    lever_arm,
                    compose_order=self.imu_full_compose_order,
                    inverse_imu=self.imu_full_inverse,
                    optimize_bias=optimize_bias,
                )
            except ValueError:
                record_edge(
                    "bad_compose_order",
                    prev_ix,
                    curr_ix,
                    dt_value=dt_value,
                    row_weight=row_weight,
                    row_scale=row_scale,
                    dr_norm=dr_norm,
                    dv_norm=dv_norm,
                    dp_norm=dp_norm,
                    dv_norm_internal=dv_norm_internal,
                    dp_norm_internal=dp_norm_internal,
                )
                continue

            # Keep the physical residual/Jacobian construction in
            # ``geom.imu_factor`` so the finite-difference test exercises the
            # exact code path used for runtime DBA.
            q_i = factor_details["q_i"]
            q_j = factor_details["q_j"]
            q_pred = factor_details["q_pred"]
            rotvec = factor_details["rotvec"]
            r_p = factor_details["r_p"]
            r_v = factor_details["r_v"]
            r_R = factor_details["r_R"]
            r_ba = factor_details["r_ba"]
            r_bg = factor_details["r_bg"]
            R_i = factor_details["R_i"]
            R_pred = factor_details["R_pred"]
            Jl_inv = factor_details["Jl_inv"]

            if not torch.isfinite(residual).all():
                record_edge(
                    "nonfinite_residual",
                    prev_ix,
                    curr_ix,
                    dt_value=dt_value,
                    row_weight=row_weight,
                    row_scale=row_scale,
                    dr_norm=dr_norm,
                    dv_norm=dv_norm,
                    dp_norm=dp_norm,
                    dv_norm_internal=dv_norm_internal,
                    dp_norm_internal=dp_norm_internal,
                )
                continue

            residual_deg = float(torch.linalg.norm(r_R).detach().cpu().item() * rad_to_deg)
            if (
                self.imu_ba_prior_max_deg > 0.0
                and residual_deg > self.imu_ba_prior_max_deg
            ):
                record_edge(
                    "skip_rot",
                    prev_ix,
                    curr_ix,
                    dt_value=dt_value,
                    row_weight=row_weight,
                    row_scale=row_scale,
                    dr_norm=dr_norm,
                    dv_norm=dv_norm,
                    dp_norm=dp_norm,
                    dv_norm_internal=dv_norm_internal,
                    dp_norm_internal=dp_norm_internal,
                    rp_norm=debug_norm(r_p),
                    rv_norm=debug_norm(r_v),
                    rrot_deg=residual_deg,
                    rba_norm=debug_norm(r_ba),
                    rbg_norm=debug_norm(r_bg),
                )
                continue

            rp_norm = float(torch.linalg.norm(r_p).detach().cpu().item())
            rv_norm = float(torch.linalg.norm(r_v).detach().cpu().item())
            confidence_components = self._confidence_components_for_pose_index(curr_ix)
            confidence_rot = float(confidence_components[0])
            confidence_vel = float(confidence_components[1])
            confidence_pos = float(confidence_components[2])
            confidence = float(np.mean(confidence_components))
            adaptive_gate, gate_parts = self._adaptive_imu_gate(
                prev_ix,
                curr_ix,
                dr_norm=dr_norm,
                residual_deg=residual_deg,
                residual_pos_norm=rp_norm,
                residual_vel_norm=rv_norm,
            )
            weight = (
                float(self.imu_ba_prior_weight)
                * row_scale
                * adaptive_gate
            )
            if (
                weight <= 0.0
                or not np.isfinite(weight)
                or max(confidence_rot, confidence_vel, confidence_pos) <= 0.0
            ):
                record_edge(
                    "skip_weight",
                    prev_ix,
                    curr_ix,
                    dt_value=dt_value,
                    row_weight=row_weight,
                    row_scale=row_scale,
                    confidence=confidence,
                    confidence_rot=confidence_rot,
                    confidence_vel=confidence_vel,
                    confidence_pos=confidence_pos,
                    weight=weight,
                    dr_norm=dr_norm,
                    dv_norm=dv_norm,
                    dp_norm=dp_norm,
                    dv_norm_internal=dv_norm_internal,
                    dp_norm_internal=dp_norm_internal,
                    rp_norm=rp_norm,
                    rv_norm=rv_norm,
                    rrot_deg=residual_deg,
                    rba_norm=debug_norm(r_ba),
                    rbg_norm=debug_norm(r_bg),
                    **gate_parts,
                )
                continue

            covariance_scale = None
            if self.use_imu_full_covariance_weighting:
                covariance_scale, info_scales = self._imu_full_covariance_term_scale(
                    curr_ix,
                    info_ref,
                    device,
                    dtype,
                )
            else:
                info_scales = None

            covariance_available = covariance_scale is not None
            if covariance_available:
                info_term_scale = covariance_scale
            else:
                # A malformed legacy row cannot safely supply cross-block
                # covariance.  Preserve the established relative-row path for
                # that individual edge, and expose the fallback in debug logs.
                info_term_scale, fallback_info_scales = self._imu_info_term_scale(
                    curr_ix,
                    info_ref,
                    device,
                    dtype,
                )
                if info_scales is None:
                    info_scales = fallback_info_scales
            if not optimize_bias:
                info_term_scale = info_term_scale[:9]

            # The old fixed ``imu_full_bias_weight`` made a local bias step
            # equally cheap at every time interval and for both sensors.  A
            # real bias random walk instead has covariance rw^2 * dt.  Its
            # information is normalized by the calibrated rotation reference,
            # just as p/v/R information is normalized elsewhere in this
            # factor; this retains a coherent visual-vs-IMU scale without an
            # additional global lambda.
            bias_rw_scale = None
            bias_rw_mode = "legacy_fixed"
            if optimize_bias and self.use_imu_bias_random_walk:
                random_walk = getattr(self.video, "imu_bias_random_walk", None)
                rotation_reference = (
                    info_ref[0]
                    if info_ref is not None and info_ref.numel() >= 1
                    else None
                )
                if (
                    random_walk is not None
                    and curr_ix < int(random_walk.shape[0])
                    and rotation_reference is not None
                ):
                    candidate_scale, candidate_valid = (
                        bias_random_walk_relative_sqrt_information(
                            random_walk[curr_ix].detach().to(
                                device=device,
                                dtype=dtype,
                            ),
                            dt,
                            rotation_reference,
                            eps=self.imu_info_weight_eps,
                        )
                    )
                    if bool(candidate_valid.detach().cpu().item()):
                        bias_rw_scale = candidate_scale
                        bias_rw_mode = "sensor_rw_relative_to_rotation"
                # A malformed/legacy prior has no physical rw value.  Preserve
                # the exact historical behavior for that edge rather than
                # pretending an arbitrary density is calibrated.
                if bias_rw_scale is None:
                    bias_rw_scale = torch.full(
                        (6,),
                        max(self.imu_full_bias_weight, 0.0) ** 0.5,
                        device=device,
                        dtype=dtype,
                    )
                    bias_rw_mode = "legacy_fixed_missing_rw"

            # Residual rows are [p, v, R, ba, bg] while learned confidence is
            # [R, v, p].  Scaling both r and J by sqrt(c) adds the intended
            # component-wise Mahalanobis weight directly to the normal system.
            # Direct precision learns the R/v/p likelihood itself.  Bias
            # random-walk rows remain a physical sensor model, rather than a
            # fourth learned quality channel.
            bias_confidence = (
                1.0
                if self.use_learned_imu_precision
                else float(np.mean(confidence_components))
            )
            component_scale = torch.as_tensor(
                [confidence_pos] * 3
                + [confidence_vel] * 3
                + [confidence_rot] * 3
                + ([bias_confidence] * 6 if optimize_bias else []),
                device=device,
                dtype=dtype,
            )
            if self.use_learned_imu_precision:
                component_scale = component_scale.clamp(1e-12, 1.0).sqrt()
            else:
                component_scale = component_scale.clamp(0.0, 1.0).sqrt()
            # If one corrupt/missing IMU-information row occurs, retain the
            # exact compatibility p/v/R scale for that row rather than turning
            # it into an accidental high-weight factor.
            edge_base_scale = (
                # Direct precision has no legacy p/v compatibility scale,
                # including on a malformed covariance row.  The information
                # helper above supplies its safe per-row fallback instead.
                base_scale
                if (
                    self.use_learned_imu_precision
                    or not self.use_imu_full_covariance_weighting
                    or covariance_available
                )
                else compatibility_base_scale
            )
            scale = edge_base_scale * info_term_scale * component_scale
            if bias_rw_scale is not None:
                scale[9:15] = scale[9:15] * bias_rw_scale
            residual = residual * scale
            Ji = Ji * scale.view(residual_dim, 1)
            Jj = Jj * scale.view(residual_dim, 1)
            w = torch.as_tensor(weight, device=device, dtype=dtype)

            # Shared global-bias Jacobian, in the same ``r - J dx`` convention
            # as ``Ji/Jj``. In 15DoF mode the global bias has exactly the
            # measurement-row derivative of source local [ba, bg], but it has
            # *no* derivative for local random-walk rows r_ba/r_bg. Keeping
            # those rows zero is essential: a global bias is constant over an
            # edge, whereas local biases are innovations around that centre.
            if joint_bias_enabled and prev_ix >= joint_start_ix:
                if full_joint_bias:
                    Jb = torch.zeros((15, 6), device=device, dtype=dtype)
                    # ``Ji`` has already received every row's information,
                    # confidence, and random-walk scaling. Its source-local
                    # bias measurement rows are therefore the exact scaled
                    # global derivative we need for r_p/r_v/r_R.
                    Jb[:9] = Ji[:9, 9:15]
                else:
                    Jb = torch.zeros((9, 6), device=device, dtype=dtype)
                    Jb[0:3, 0:3] = J_p_ba
                    Jb[0:3, 3:6] = J_p_bg
                    Jb[3:6, 0:3] = J_v_ba
                    Jb[3:6, 3:6] = J_v_bg
                    gyro_sign = -1.0 if self.imu_full_inverse else 1.0
                    preintegration_map = gyro_sign * so3_left_jacobian(
                        gyro_sign * rotvec
                    )
                    if self.imu_full_compose_order == "dq_prev":
                        gyro_map = R_pred.transpose(0, 1)
                    else:  # ``full_imu_factor_linearization`` has validated it.
                        gyro_map = R_pred.transpose(0, 1) @ R_i
                    Jb[6:9, 3:6] = (
                        Jl_inv @ gyro_map @ preintegration_map @ J_R_bg
                    )
                    Jb = Jb * scale[:9].view(9, 1)

                # The first global-backend factor can connect fixed frame 0
                # to optimized frame 1 (t0=1). Do not let Python's negative
                # index accidentally attach that fixed endpoint to the final
                # state block of the Schur system.
                if 0 <= prev_local < P:
                    joint_Hxb[prev_local] += w * (Ji.transpose(0, 1) @ Jb)
                joint_Hxb[curr_local] += w * (Jj.transpose(0, 1) @ Jb)
                joint_Hbb_data += w * (Jb.transpose(0, 1) @ Jb)
                # The legacy 9D factor has only measurement rows. The 15D
                # factor additionally contains local random-walk rows; Jb is
                # intentionally zero there, so retain the full residual to
                # keep the matrix product/state layout exact.
                joint_residual = residual if full_joint_bias else residual[:9]
                joint_gb += w * (Jb.transpose(0, 1) @ joint_residual)
                joint_dst.append(curr_ix)
                joint_used += 1
                joint_duration += dt_value
                joint_rotation_deg += dr_norm * rad_to_deg

            Hii = w * (Ji.transpose(0, 1) @ Ji)
            Hij = w * (Ji.transpose(0, 1) @ Jj)
            Hji = w * (Jj.transpose(0, 1) @ Ji)
            Hjj = w * (Jj.transpose(0, 1) @ Jj)
            vi = w * (Ji.transpose(0, 1) @ residual)
            vj = w * (Jj.transpose(0, 1) @ residual)

            if 0 <= prev_local < P:
                add_block(prev_local, prev_local, Hii)
                add_block(prev_local, curr_local, Hij)
                add_block(curr_local, prev_local, Hji)
                v_prior[prev_local] += vi

            add_block(curr_local, curr_local, Hjj)
            v_prior[curr_local] += vj
            used += 1
            confidence_sum += confidence
            adaptive_gate_sum += adaptive_gate
            record_edge(
                "used",
                prev_ix,
                curr_ix,
                dt_value=dt_value,
                row_weight=row_weight,
                row_scale=row_scale,
                confidence=confidence,
                confidence_rot=confidence_rot,
                confidence_vel=confidence_vel,
                confidence_pos=confidence_pos,
                weight=weight,
                dr_norm=dr_norm,
                dv_norm=dv_norm,
                dp_norm=dp_norm,
                dv_norm_internal=dv_norm_internal,
                dp_norm_internal=dp_norm_internal,
                rp_norm=rp_norm,
                rv_norm=rv_norm,
                rrot_deg=residual_deg,
                rba_norm=debug_norm(r_ba),
                rbg_norm=debug_norm(r_bg),
                info_rot_scale=info_scales[0],
                info_vel_scale=info_scales[1],
                info_pos_scale=info_scales[2],
                bias_accel_rw_scale=(
                    float(bias_rw_scale[0].detach().cpu().item())
                    if bias_rw_scale is not None
                    else 1.0
                ),
                bias_gyro_rw_scale=(
                    float(bias_rw_scale[3].detach().cpu().item())
                    if bias_rw_scale is not None
                    else 1.0
                ),
                bias_rw_mode=bias_rw_mode,
                **gate_parts,
            )

        if self.imu_motion_prior_weight > 0.0 or self.imu_local_bias_prior_weight > 0.0:
            for pose_ix in range(t0, min(t1, n)):
                local = pose_ix - t0
                if local < 0 or local >= P:
                    continue

                H = torch.zeros((state_dim, state_dim), device=device, dtype=dtype)
                v = torch.zeros(state_dim, device=device, dtype=dtype)

                if self.imu_motion_prior_weight > 0.0:
                    if pose_ix > 0:
                        dt = float(imu_delta[pose_ix, 0].detach().cpu().item())
                        if dt > 1e-4 and np.isfinite(dt):
                            v_ref = (
                                centers[pose_ix].detach()
                                - centers[pose_ix - 1].detach()
                            ) / dt
                        else:
                            v_ref = torch.zeros(3, device=device, dtype=dtype)
                    else:
                        v_ref = torch.zeros(3, device=device, dtype=dtype)

                    residual_v = velocities[pose_ix].detach() - v_ref
                    wv = torch.as_tensor(self.imu_motion_prior_weight, device=device, dtype=dtype)
                    H[6:9, 6:9] += wv * eye3
                    v[6:9] += -wv * residual_v

                if optimize_bias and self.imu_local_bias_prior_weight > 0.0:
                    wb = torch.as_tensor(
                        self.imu_local_bias_prior_weight,
                        device=device,
                        dtype=dtype,
                    )
                    H[9:12, 9:12] += wb * eye3
                    H[12:15, 12:15] += wb * eye3
                    v[9:12] += -wb * bias_acc[pose_ix].detach()
                    v[12:15] += -wb * bias_gyro[pose_ix].detach()

                if torch.count_nonzero(H).item() > 0:
                    add_block(local, local, H)
                    v_prior[local] += v

        joint_info = None
        if joint_bias_enabled:
            joint_reason = None
            if joint_used < self.imu_joint_bias_min_edges:
                joint_reason = "insufficient_edges"
            elif joint_duration < self.imu_joint_bias_min_duration:
                joint_reason = "insufficient_duration"
            elif joint_rotation_deg < self.imu_joint_bias_min_rotation_deg:
                joint_reason = "insufficient_rotation"

            if joint_reason is None:
                data_hessian = 0.5 * (joint_Hbb_data + joint_Hbb_data.transpose(0, 1))
                try:
                    eigenvalues = torch.linalg.eigvalsh(data_hessian)
                    largest = float(eigenvalues[-1].detach().cpu().item())
                    rank_threshold = max(largest * 1e-6, 1e-12)
                    rank = int((eigenvalues > rank_threshold).sum().detach().cpu().item())
                    smallest = float(eigenvalues[0].detach().cpu().item())
                    condition = (
                        largest / max(smallest, rank_threshold)
                        if largest > 0.0
                        else float("inf")
                    )
                    if rank < 6 or not np.isfinite(condition):
                        joint_reason = "poor_observability"
                    elif condition > self.imu_joint_bias_max_condition:
                        joint_reason = "ill_conditioned"
                except RuntimeError:
                    joint_reason = "eigendecomposition_failed"

            if joint_reason is None:
                from vi_initialization import _shared_bias_prior_std

                joint_dst_tensor = torch.as_tensor(
                    joint_dst, device=device, dtype=torch.long
                )
                prior_std = _shared_bias_prior_std(
                    self.video,
                    joint_dst_tensor,
                    joint_duration,
                    self.imu_joint_bias_accel_random_walk,
                    self.imu_joint_bias_gyro_random_walk,
                    self.imu_joint_bias_accel_prior_std,
                    self.imu_joint_bias_gyro_prior_std,
                )
                prior_precision = torch.zeros(6, device=device, dtype=dtype)
                usable_prior = torch.isfinite(prior_std) & (prior_std > 0.0)
                prior_precision[usable_prior] = 1.0 / prior_std[usable_prior].square()
                joint_hessian = data_hessian + torch.diag(prior_precision)
                joint_hessian += self.imu_joint_bias_damping * torch.eye(
                    6, device=device, dtype=dtype
                )

                local_start = max(0, joint_start_ix - t0)
                local_stop = min(P, max_pose_ix - t0)
                local_indices = torch.arange(
                    local_start, local_stop, device=device, dtype=torch.long
                )
                try:
                    joint_hxb = joint_Hxb[local_indices]
                    correction_h, correction_v = self._schur_eliminate_shared_bias(
                        joint_hxb, joint_hessian, joint_gb
                    )
                except RuntimeError:
                    joint_reason = "singular"
                else:
                    local_count = int(local_indices.numel())
                    H_blocks.extend(
                        correction_h.reshape(-1, state_dim, state_dim).unbind(0)
                    )
                    block_i = local_indices[:, None].expand(local_count, local_count)
                    block_j = local_indices[None, :].expand(local_count, local_count)
                    H_ii.extend(block_i.reshape(-1).detach().cpu().tolist())
                    H_jj.extend(block_j.reshape(-1).detach().cpu().tolist())
                    v_prior[local_indices] += correction_v
                    joint_info = {
                        "Hbb": joint_hessian,
                        "gb": joint_gb,
                        "Hxb": joint_hxb,
                        "state_dim": int(state_dim),
                        "mode": "full_15dof" if full_joint_bias else "fixed_bias_9dof",
                        "state_locals": local_indices,
                        "edges": float(joint_used),
                        "duration_sec": float(joint_duration),
                        "rotation_excitation_deg": float(joint_rotation_deg),
                        "condition": float(condition),
                        "rank": float(rank),
                        "prior_accel_std": float(prior_std[:3].mean().detach().cpu().item()),
                        "prior_gyro_std": float(prior_std[3:].mean().detach().cpu().item()),
                        "ba_t0": int(t0),
                        "window_t0": int(joint_start_ix),
                        "window_t1": int(max_pose_ix),
                    }

            self.last_joint_shared_bias_status = {
                "enabled": 1.0,
                "accepted_for_joint_system": float(joint_info is not None),
                "reason": "ready" if joint_info is not None else str(joint_reason),
                "edges": float(joint_used),
                "duration_sec": float(joint_duration),
                "rotation_excitation_deg": float(joint_rotation_deg),
            }

        if used == 0 or len(H_blocks) == 0:
            return empty_result()

        if not self._reported_imu_ba_prior:
            conf_mean = confidence_sum / max(used, 1)
            gate_mean = adaptive_gate_sum / max(used, 1)
            print(
                f"[IMU-BA] {'full' if optimize_bias else 'pose-velocity'} prior enabled: "
                f"window=[{t0},{t1}), "
                f"edges={used}, blocks={len(H_blocks)}, "
                f"weight={self.imu_ba_prior_weight}, conf_mean={conf_mean:.4f}, "
                f"adaptive_gate_mean={gate_mean:.4f}, "
                f"pos_w={'covariance' if self.use_imu_full_covariance_weighting else self.imu_full_pos_weight}, "
                f"vel_w={'covariance' if self.use_imu_full_covariance_weighting else self.imu_full_vel_weight}, "
                f"bias_w={'sensor_random_walk' if (optimize_bias and self.use_imu_bias_random_walk) else (self.imu_full_bias_weight if optimize_bias else 0.0)}, "
                f"state_dim={state_dim}, max_dt={self.imu_full_max_dt}, "
                f"compose_order={self.imu_full_compose_order}, "
                f"inverse_imu={self.imu_full_inverse}, "
                f"info_weighting={self.use_imu_info_weighting}, "
                f"info_clip={self.imu_info_weight_clip}, "
                f"info_reference={self._imu_info_reference_mode}, "
                f"gravity={tuple(float(x) for x in gravity.detach().cpu().tolist())}, "
                f"skipped={skipped}"
            )
            self._reported_imu_ba_prior = True

        state_prior_H = torch.stack(H_blocks, dim=0).contiguous()
        state_prior_ii = torch.as_tensor(H_ii, device=device, dtype=torch.long).contiguous()
        state_prior_jj = torch.as_tensor(H_jj, device=device, dtype=torch.long).contiguous()
        result = (state_prior_H, v_prior.contiguous(), state_prior_ii, state_prior_jj)
        return (*result, joint_info) if return_joint_info else result

    def _apply_joint_shared_bias_update(self, ba_result, joint_info, stage):
        """Recover the shared-bias increment after one coupled DBA step.

        CUDA receives the Schur-reduced 9D or 15D state system. Once it
        returns the matching state increment, back-substitution recovers the
        same six-dimensional global-bias update that a monolithic
        ``[T, v, ba, bg, b_global]`` solve would produce for this linearization.
        """

        if joint_info is None:
            return {"updated": 0.0, "reason": "joint_block_unavailable"}
        if ba_result is None or len(ba_result) == 0:
            return {"updated": 0.0, "reason": "missing_ba_increment"}

        dx = ba_result[0]
        state_dim = int(joint_info.get("state_dim", 9))
        if state_dim not in (9, 15):
            return {"updated": 0.0, "reason": "invalid_joint_state_dim"}
        if not torch.is_tensor(dx) or dx.ndim != 2 or dx.shape[-1] < state_dim:
            return {"updated": 0.0, "reason": "invalid_ba_increment"}

        state_locals = joint_info["state_locals"]
        if state_locals.numel() == 0 or int(state_locals.max().item()) >= dx.shape[0]:
            return {"updated": 0.0, "reason": "state_increment_out_of_range"}

        Hbb = joint_info["Hbb"]
        Hxb = joint_info["Hxb"]
        gb = joint_info["gb"]
        dx_local = dx[state_locals, :state_dim].to(
            device=Hbb.device, dtype=Hbb.dtype
        )
        rhs = gb - torch.einsum("aik,ai->k", Hxb, dx_local)
        try:
            step = torch.linalg.solve(Hbb, rhs) * self.imu_joint_bias_step_scale
        except RuntimeError:
            return {"updated": 0.0, "reason": "singular_backsubstitution"}
        if not torch.isfinite(step).all():
            return {"updated": 0.0, "reason": "nonfinite_backsubstitution"}

        accel_step = step[:3]
        gyro_step = step[3:]
        accel_step_norm = accel_step.norm().clamp_min(1e-12)
        gyro_step_norm = gyro_step.norm().clamp_min(1e-12)
        if self.imu_joint_bias_max_accel_step > 0.0:
            accel_step = accel_step * min(
                1.0,
                self.imu_joint_bias_max_accel_step / float(accel_step_norm.item()),
            )
        if self.imu_joint_bias_max_gyro_step > 0.0:
            gyro_step = gyro_step * min(
                1.0,
                self.imu_joint_bias_max_gyro_step / float(gyro_step_norm.item()),
            )

        device = Hbb.device
        dtype = Hbb.dtype
        accel_bias = self.video.imu_global_bias_acc.detach().to(
            device=device, dtype=dtype
        ).clone()
        gyro_bias = self.video.imu_global_bias_gyro.detach().to(
            device=device, dtype=dtype
        ).clone()
        candidate_accel = accel_bias + accel_step
        candidate_gyro = gyro_bias + gyro_step
        if self.imu_joint_bias_max_accel_norm > 0.0:
            norm = candidate_accel.norm().clamp_min(1e-12)
            candidate_accel *= min(
                1.0,
                self.imu_joint_bias_max_accel_norm / float(norm.item()),
            )
        if self.imu_joint_bias_max_gyro_norm > 0.0:
            norm = candidate_gyro.norm().clamp_min(1e-12)
            candidate_gyro *= min(
                1.0,
                self.imu_joint_bias_max_gyro_norm / float(norm.item()),
            )

        candidate_step = torch.cat(
            [candidate_accel - accel_bias, candidate_gyro - gyro_bias], dim=0
        )
        improvement = torch.dot(candidate_step, rhs) - 0.5 * torch.dot(
            candidate_step, Hbb @ candidate_step
        )
        improvement_value = float(improvement.detach().cpu().item())
        if (
            not np.isfinite(improvement_value)
            or improvement_value <= self.imu_joint_bias_min_improvement
        ):
            return {
                "updated": 0.0,
                "reason": "no_quadratic_improvement",
                "quadratic_improvement": improvement_value,
            }

        accel_step = candidate_accel - accel_bias
        gyro_step = candidate_gyro - gyro_bias
        self.video.imu_global_bias_acc[:] = candidate_accel
        self.video.imu_global_bias_gyro[:] = candidate_gyro
        absolute_indices = state_locals + int(joint_info["ba_t0"])
        self.video.dirty[absolute_indices] = True

        relinearize = bool(
            (
                self.imu_joint_bias_relinearize_accel > 0.0
                and float(accel_step.norm().item())
                > self.imu_joint_bias_relinearize_accel
            )
            or (
                self.imu_joint_bias_relinearize_gyro > 0.0
                and float(gyro_step.norm().item())
                > self.imu_joint_bias_relinearize_gyro
            )
        )
        self.video.imu_bias_relinearization_required = relinearize
        result = {
            "updated": 1.0,
            "joint": 1.0,
            "joint_state_dim": float(state_dim),
            "joint_full_15dof": float(state_dim == 15),
            "edges": float(joint_info["edges"]),
            "duration_sec": float(joint_info["duration_sec"]),
            "rotation_excitation_deg": float(joint_info["rotation_excitation_deg"]),
            "condition": float(joint_info["condition"]),
            "rank": float(joint_info["rank"]),
            "prior_accel_std": float(joint_info["prior_accel_std"]),
            "prior_gyro_std": float(joint_info["prior_gyro_std"]),
            "quadratic_improvement": improvement_value,
            "accel_step_norm": float(accel_step.norm().detach().cpu().item()),
            "gyro_step_norm": float(gyro_step.norm().detach().cpu().item()),
            "accel_bias_norm": float(candidate_accel.norm().detach().cpu().item()),
            "gyro_bias_norm": float(candidate_gyro.norm().detach().cpu().item()),
            "relinearization_required": float(relinearize),
            "window_t0": float(joint_info["window_t0"]),
            "window_t1": float(joint_info["window_t1"]),
        }
        try:
            from vi_initialization import record_window_bias_refinement

            record_window_bias_refinement(self.video, result, f"joint-{stage}")
        except Exception as exc:
            print(f"[VI-BIAS-JOINT WARNING] history write failed: {exc}")

        updates = getattr(self, "_joint_bias_update_count", 0) + 1
        self._joint_bias_update_count = updates
        if updates <= 3 or updates % 20 == 0:
            print(
                "[VI-BIAS-JOINT] coupled shared block accepted "
                f"stage={stage}, state_dim={state_dim}, edges={int(result['edges'])}, "
                f"gain={improvement_value:.6g}, "
                f"|dba|={result['accel_step_norm']:.6g}, "
                f"|dbg|={result['gyro_step_norm']:.6g}, "
                f"cond={result['condition']:.3g}"
            )
        if relinearize:
            print(
                "[VI-BIAS-JOINT] bias correction exceeded the first-order "
                "relinearization threshold; raw-IMU re-preintegration is recommended."
            )
        return result

    def _run_dba_with_imu_prior(
        self,
        *,
        target,
        weight,
        damping,
        ii,
        jj,
        ba_t0,
        ba_t1,
        itrs,
        lm,
        ep,
        motion_only,
        stage,
        joint_bias_requested=False,
        outer_relinearize=False,
    ):
        """Run visual DBA with a freshly linearized IMU prior each outer step.

        DROID's CUDA backend can run several inner iterations, but it only
        returns the final state increment.  A Schur-eliminated shared-bias
        block needs the *matching* increment to recover ``db``.  Therefore a
        coupled update is evaluated as one CUDA iteration at a time:

        ``build H(T, v, b) -> CUDA one step -> back-substitute db -> rebuild``.

        ``outer_relinearize`` also exposes this exact schedule for fixed-bias
        ablations.  That prevents a comparison from accidentally measuring a
        different number of visual DBA linearizations instead of the bias
        coupling itself.
        """

        cuda_iterations = max(1, int(itrs))
        outer_iterations = 1
        if outer_relinearize:
            cuda_iterations = 1
            outer_iterations = max(1, int(itrs))

        ba_result = None
        for outer_index in range(outer_iterations):
            joint_bias_info = None
            if self.use_full_imu_ba:
                if self._joint_shared_bias_enabled():
                    (
                        pose_prior_H,
                        pose_prior_v,
                        pose_prior_ii,
                        pose_prior_jj,
                        joint_bias_info,
                    ) = self._build_full_imu_ba_prior(
                        ba_t0,
                        ba_t1,
                        return_joint_info=True,
                        enable_joint_bias=joint_bias_requested,
                        # The caller marks the interval once before this
                        # loop.  Rebuilding an outer step must not consume an
                        # additional scheduling interval.
                        mark_joint_attempt=False,
                    )
                else:
                    (
                        pose_prior_H,
                        pose_prior_v,
                        pose_prior_ii,
                        pose_prior_jj,
                    ) = self._build_full_imu_ba_prior(ba_t0, ba_t1)
            else:
                (
                    pose_prior_H,
                    pose_prior_v,
                    pose_prior_ii,
                    pose_prior_jj,
                ) = self._build_imu_ba_pose_prior(ba_t0, ba_t1)

            ba_result = self.video.ba(
                target,
                weight,
                damping,
                ii,
                jj,
                ba_t0,
                ba_t1,
                itrs=cuda_iterations,
                lm=lm,
                ep=ep,
                motion_only=motion_only,
                pose_prior_H=pose_prior_H,
                pose_prior_v=pose_prior_v,
                pose_prior_ii=pose_prior_ii,
                pose_prior_jj=pose_prior_jj,
            )

            if joint_bias_info is not None:
                update_stage = stage
                if outer_iterations > 1:
                    update_stage = f"{stage}-outer{outer_index + 1}"
                self._apply_joint_shared_bias_update(
                    ba_result, joint_bias_info, stage=update_stage
                )

        return ba_result

    def flush_imu_debug(self):
        if self.imu_regularizer is not None:
            try:
                self.imu_regularizer.flush_debug()
            except Exception as e:
                print(f"[IMU-RESIDUAL WARNING] debug flush failed: {e}")

        try:
            self._flush_imu_ba_debug()
        except Exception as e:
            print(f"[IMU-BA WARNING] debug flush failed: {e}")

    def __filter_repeated_edges(self, ii, jj):
        """
        중복 edge 제거 함수.

        새로 추가하려는 edge (ii, jj)가
        이미 active factor 또는 inactive factor에 있으면 제거한다.
        """

        # active edge와 중복되는지 확인
        if len(self.ii) > 0:
            mask = ((ii[:, None] == self.ii) & (jj[:, None] == self.jj)).any(dim=-1)
            ii = ii[~mask]
            jj = jj[~mask]

        # inactive edge와 중복되는지 확인
        if len(self.ii_inac) > 0:
            mask = ((ii[:, None] == self.ii_inac) & (jj[:, None] == self.jj_inac)).any(
                dim=-1
            )
            ii = ii[~mask]
            jj = jj[~mask]

        return ii, jj

    def print_edges(self):
        """
        현재 active edge 목록을 출력하는 디버깅 함수.

        출력:
            source index, target index, 평균 weight
        """

        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        # source index 기준 정렬
        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        # edge별 평균 weight 계산
        w = torch.mean(self.weight, dim=[0, 2, 3, 4]).cpu().numpy()
        w = w[ix]

        for e in zip(ii, jj, w):
            print(e)
        print()

    def filter_edges(self):
        """
        confidence가 낮은 나쁜 edge 제거.

        조건:
            1. 시간적으로 충분히 떨어진 frame pair이고,
            2. 평균 confidence가 매우 낮으면 bad edge로 판단.

        bad edge는 ii_bad, jj_bad에 저장해서 이후 proximity factor 추가 시
        다시 선택되지 않도록 한다.
        """

        # edge별 평균 confidence
        conf = torch.mean(self.weight, dim=[0, 2, 3, 4])

        # frame index 차이가 2보다 크고 confidence가 거의 0이면 bad edge
        mask = (torch.abs(self.ii - self.jj) > 2) & (conf < 0.001)

        # bad edge 목록에 저장
        self.ii_bad = torch.cat([self.ii_bad, self.ii[mask]])
        self.jj_bad = torch.cat([self.jj_bad, self.jj[mask]])

        # active graph에서는 제거
        self.rm_factors(mask, store=False)

    def clear_edges(self):
        """
        현재 active factor를 전부 제거한다.

        주로 동기식 backend에서 전체 graph update 후
        edge를 정리할 때 사용된다.
        """

        # self.ii >= 0은 모든 active factor를 의미
        self.rm_factors(self.ii >= 0)

        # factor-specific feature도 초기화
        self.net = None
        self.inp = None

    @autocast(enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """
        factor graph에 새 edge를 추가한다.

        ii:
            source frame indices

        jj:
            target frame indices

        remove:
            max_factors를 초과할 때 오래된 factor를 제거할지 여부.

        처리 순서:
            1. ii, jj를 CUDA tensor로 변환
            2. 중복 edge 제거
            3. max_factors 초과 시 오래된 factor 제거
            4. source frame의 net/inp feature 가져오기
            5. correlation volume 생성
            6. 현재 pose/depth 기준으로 ii -> jj reproject
            7. target/weight/edge index 저장
        """

        # ii가 tensor가 아니면 tensor로 변환
        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        # jj가 tensor가 아니면 tensor로 변환
        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # active/inactive edge와 중복되는 edge 제거
        ii, jj = self.__filter_repeated_edges(ii, jj)

        # 추가할 edge가 없으면 종료
        if ii.shape[0] == 0:
            return

        ############################################################
        # max_factors 제한 처리
        ############################################################

        # factor 수가 max_factors를 초과할 경우,
        # age가 큰 오래된 factor를 제거하고 새 factor를 넣는다.
        if (
            self.max_factors > 0
            and self.ii.shape[0] + ii.shape[0] > self.max_factors
            and self.corr is not None
            and remove
        ):

            # age 기준으로 정렬
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]

            # max_factors를 넘는 오래된 factor 제거
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        ############################################################
        # source frame context feature
        ############################################################

        # source frame ii의 hidden state를 가져온다.
        # shape: [1, num_edges, 128, ht, wd]
        net = self.video.nets[ii].to(self.device).unsqueeze(0)

        ############################################################
        # correlation volume 생성
        ############################################################

        if self.corr_impl == "volume":

            # stereo일 경우 ii == jj인 self edge는 오른쪽 이미지 feature를 사용할 수 있음.
            # monocular에서는 c=0이므로 일반적으로 첫 번째 feature를 사용.
            c = (ii == jj).long()

            # source frame feature
            fmap1 = self.video.fmaps[ii, 0].to(self.device).unsqueeze(0)

            # target frame feature
            # stereo인 경우 c에 따라 left/right feature 선택 가능
            fmap2 = self.video.fmaps[jj, c].to(self.device).unsqueeze(0)

            # CorrBlock 생성
            # 각 edge별로 fmap1과 fmap2 사이 correlation volume을 만든다.
            corr = CorrBlock(fmap1, fmap2)

            # 기존 corr이 있으면 cat으로 이어 붙임
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            # source frame의 context input feature
            inp = self.video.inps[ii].to(self.device).unsqueeze(0)

            # 기존 inp와 새 inp를 edge dimension으로 연결
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        ############################################################
        # 초기 target 생성
        ############################################################

        with autocast(enabled=False):

            # 현재 video.poses, video.disps, intrinsics를 이용해서
            # source frame ii의 픽셀을 target frame jj로 reproject한다.
            #
            # target 초기값은 현재 pose/depth가 예측하는 correspondence 좌표다.
            target, _ = self.video.reproject(ii, jj)

            # 초기 weight는 0.
            # 이후 update network가 confidence weight를 예측한다.
            weight = torch.zeros_like(target)

        ############################################################
        # active factor state 저장
        ############################################################

        # active edge index 추가
        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)

        # 새 factor의 age는 0부터 시작
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # source frame hidden state 저장
        self.net = net if self.net is None else torch.cat([self.net, net], 1)

        # target / weight 저장
        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)

    @autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """
        factor graph에서 edge를 제거한다.

        mask:
            제거할 active factor 위치를 나타내는 boolean tensor.

        store:
            True이면 제거되는 factor의 ii, jj, target, weight를 inactive storage에 저장.
            False이면 완전히 제거.

        inactive factor는 이후 update(use_inactive=True)에서 다시 사용할 수 있다.
        """

        ############################################################
        # inactive factor로 저장
        ############################################################

        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_inac = torch.cat([self.target_inac, self.target[:, mask]], 1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[:, mask]], 1)

        ############################################################
        # active factor 제거
        ############################################################

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]

        # volume correlation을 쓰는 경우 corr에서도 해당 edge 제거
        if self.corr_impl == "volume":
            self.corr = self.corr[~mask]

        # hidden/context feature 제거
        if self.net is not None:
            self.net = self.net[:, ~mask]

        if self.inp is not None:
            self.inp = self.inp[:, ~mask]

        # target / weight 제거
        self.target = self.target[:, ~mask]
        self.weight = self.weight[:, ~mask]

    @autocast(enabled=True)
    def rm_keyframe(self, ix):
        """
        특정 keyframe을 video buffer와 factor graph에서 제거한다.

        ix:
            제거할 keyframe index.

        사용 위치:
            DroidFrontend._update()에서 frame distance가 너무 작으면
            중복 keyframe으로 판단하고 rm_keyframe(self.t1 - 3)을 호출한다.

        처리:
            1. DepthVideo의 ix 이후 데이터를 한 칸씩 앞으로 당긴다.
            2. inactive factor 중 ix와 연결된 edge 제거
            3. ix보다 큰 factor index를 1씩 감소
            4. active factor 중 ix와 연결된 edge 제거
        """

        # 현재 저장된 frame 수
        t = self.video.counter.value
        composed_imu = None
        if 0 < ix and ix + 1 < t:
            composed_imu = self._compose_adjacent_video_imu_delta(ix, ix + 1)

        ############################################################
        # DepthVideo buffer에서 keyframe 제거
        ############################################################

        # ix 위치를 제거하기 위해 ix+1:t 구간을 ix:t-1로 한 칸 당김
        self.video.images[ix : t - 1] = self.video.images[ix + 1 : t].clone()
        self.video.poses[ix : t - 1] = self.video.poses[ix + 1 : t].clone()
        self.video.velocities[ix : t - 1] = self.video.velocities[ix + 1 : t].clone()
        self.video.bias_acc[ix : t - 1] = self.video.bias_acc[ix + 1 : t].clone()
        self.video.bias_gyro[ix : t - 1] = self.video.bias_gyro[ix + 1 : t].clone()
        self.video.imu_delta[ix : t - 1] = self.video.imu_delta[ix + 1 : t].clone()
        self.video.imu_valid[ix : t - 1] = self.video.imu_valid[ix + 1 : t].clone()
        self.video.imu_weight[ix : t - 1] = self.video.imu_weight[ix + 1 : t].clone()
        self.video.imu_used_steps[ix : t - 1] = self.video.imu_used_steps[ix + 1 : t].clone()
        self.video.imu_info[ix : t - 1] = self.video.imu_info[ix + 1 : t].clone()
        self.video.imu_learned_confidence[ix : t - 1] = (
            self.video.imu_learned_confidence[ix + 1 : t].clone()
        )
        self.video.imu_learned_confidence_valid[ix : t - 1] = (
            self.video.imu_learned_confidence_valid[ix + 1 : t].clone()
        )
        self.video.imu_bias_random_walk[ix : t - 1] = (
            self.video.imu_bias_random_walk[ix + 1 : t].clone()
        )
        self.video.imu_bias_jacobian[ix : t - 1] = (
            self.video.imu_bias_jacobian[ix + 1 : t].clone()
        )
        if composed_imu is not None:
            (
                delta_ac,
                valid_ac,
                weight_ac,
                used_steps_ac,
                info_ac,
                bias_jac_ac,
                bias_random_walk_ac,
            ) = composed_imu
            self.video.imu_delta[ix] = delta_ac
            self.video.imu_valid[ix] = valid_ac
            self.video.imu_weight[ix] = weight_ac
            self.video.imu_used_steps[ix] = used_steps_ac
            self.video.imu_info[ix] = info_ac
            self.video.imu_bias_random_walk[ix] = bias_random_walk_ac
            if bias_jac_ac is not None:
                self.video.imu_bias_jacobian[ix] = bias_jac_ac
        # The successor now occupies ix. Whether its IMU row was composed or
        # merely shifted, the old direct visual temporal edge is no longer the
        # same interval. Re-estimate it later or retain neutral IMU weighting.
        self.video.imu_learned_confidence[ix] = 1.0
        self.video.imu_learned_confidence_valid[ix] = False
        self.video.imu_learned_confidence[t - 1] = 1.0
        self.video.imu_learned_confidence_valid[t - 1] = False
        self.video.disps[ix : t - 1] = self.video.disps[ix + 1 : t].clone()
        self.video.disps_sens[ix : t - 1] = self.video.disps_sens[ix + 1 : t].clone()
        self.video.intrinsics[ix : t - 1] = self.video.intrinsics[ix + 1 : t].clone()

        # neural feature들도 동일하게 한 칸씩 당김
        self.video.nets[ix : t - 1] = self.video.nets[ix + 1 : t].clone()
        self.video.inps[ix : t - 1] = self.video.inps[ix + 1 : t].clone()
        self.video.fmaps[ix : t - 1] = self.video.fmaps[ix + 1 : t].clone()

        # timestamp도 한 칸씩 당김
        self.video.tstamp[ix : t - 1] = self.video.tstamp[ix + 1 : t].clone()

        ############################################################
        # inactive factor index 보정
        ############################################################

        # inactive factor 중 제거할 keyframe ix와 연결된 edge
        m = (self.ii_inac == ix) | (self.jj_inac == ix)

        # ix 이후 frame index는 하나씩 앞으로 당겨졌으므로 factor index도 -1
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        # ix와 연결된 inactive factor 제거
        if torch.any(m):
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[:, ~m]
            self.weight_inac = self.weight_inac[:, ~m]

        ############################################################
        # active factor index 보정
        ############################################################

        # active factor 중 제거할 keyframe ix와 연결된 edge
        m = (self.ii == ix) | (self.jj == ix)

        # ix 이후 frame index는 하나씩 앞으로 당겨졌으므로 factor index도 -1
        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1

        # ix와 연결된 active factor 제거
        self.rm_factors(m, store=False)

    @autocast(enabled=True)
    def update(
        self,
        t0=None,
        t1=None,
        itrs=2,
        use_inactive=False,
        EP=1e-7,
        motion_only=False
    ):
        """
        active factor graph에 대해 update operator와 dense BA를 수행한다.

        주로 DroidFrontend에서 사용된다.

        처리 순서:
            1. 현재 pose/depth로 ii -> jj reproject
            2. motion feature 생성
            3. correlation feature 샘플링
            4. update network로 delta, weight, damping 예측
            5. target = coords1 + delta로 갱신
            6. inactive factor를 포함할지 결정
            7. target/weight shape을 BA 입력 형태로 변환
            8. video.ba() 호출
            9. 필요 시 disparity upsample
            10. factor age 증가
        """

        ############################################################
        # motion feature 생성
        ############################################################

        with autocast(enabled=False):

            # 현재 pose/depth 기준으로 ii frame의 point를 jj frame으로 reproject
            coords1, mask = self.video.reproject(self.ii, self.jj)

            # motion feature 구성
            #
            # coords1 - coords0:
            #   현재 추정 pose/depth가 만드는 optical flow
            #
            # self.target - coords1:
            #   이전 update에서 예측한 target과 현재 reprojection 사이의 residual
            #
            # 두 값을 concat해서 update network의 motion input으로 사용
            motn = torch.cat(
                [coords1 - self.coords0, self.target - coords1],
                dim=-1
            )

            # shape 변경:
            # [1, edges, ht, wd, 4] -> [1, edges, 4, ht, wd]
            #
            # clamp:
            # 너무 큰 motion 값이 network 입력을 불안정하게 만들지 않도록 제한
            motn = motn.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

        ############################################################
        # correlation feature
        ############################################################

        # CorrBlock에서 현재 coords1 위치의 correlation feature를 가져온다.
        corr = self.corr(coords1)

        ############################################################
        # neural update
        ############################################################

        # update_op는 DROID의 핵심 neural optimizer 역할.
        #
        # 입력:
        #   self.net:
        #       recurrent hidden state
        #   self.inp:
        #       context input
        #   corr:
        #       correlation feature
        #   motn:
        #       motion feature
        #   self.ii, self.jj:
        #       frame index pair
        #
        # 출력:
        #   self.net:
        #       업데이트된 hidden state
        #   delta:
        #       현재 coords1에서 target으로 이동해야 할 2D correction
        #   weight:
        #       correspondence confidence
        #   damping:
        #       BA damping 값
        #   upmask:
        #       disparity upsampling mask
        update_out = self.update_op(
            self.net,
            self.inp,
            corr,
            motn,
            self.ii,
            self.jj,
            return_imu_confidence=self.use_learned_imu_prediction,
            return_component_imu_confidence=self.use_component_imu_confidence,
            use_dvi_imu_confidence=self.use_dvi_imu_confidence,
            use_dvi_imu_quality_fusion=self.use_dvi_imu_quality_fusion,
            use_learned_imu_precision=self.use_learned_imu_precision,
            imu_quality=self._edge_imu_quality(self.ii, self.jj),
        )
        if self.use_learned_imu_prediction:
            self.net, delta, weight, damping, upmask, imu_edge_confidence = update_out
            self.last_imu_confidence, confidence_coverage = (
                self._edge_confidence_to_frame_confidence(
                    imu_edge_confidence, return_coverage=True
                )
            )
            self._persist_imu_confidence(
                self.last_imu_confidence, confidence_coverage
            )
        else:
            self.net, delta, weight, damping, upmask = update_out

        ############################################################
        # BA window 시작점 설정
        ############################################################

        # t0가 없으면 source frame의 최소 index + 1부터 최적화
        # 보통 첫 frame은 gauge freedom 때문에 고정하기 위해 1부터 시작한다.
        if t0 is None:
            t0 = max(1, self.ii.min().item() + 1)

        with autocast(enabled=False):

            ########################################################
            # target / weight 갱신
            ########################################################

            # update network가 예측한 delta를 현재 reprojection coords1에 더해
            # 새로운 correspondence target으로 설정
            self.target = coords1 + delta.to(dtype=torch.float)

            # confidence weight 저장
            self.weight = weight.to(dtype=torch.float)

            ht, wd = self.coords0.shape[0:2]

            # source frame별 damping 값 저장
            self.damping[torch.unique(self.ii)] = damping

            ########################################################
            # inactive factor 포함 여부
            ########################################################

            if use_inactive:

                # 최근 t0-3 이후의 inactive factor만 사용
                # 너무 오래된 inactive factor까지 넣으면 계산량이 커지고 불안정할 수 있음
                m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)

                # inactive + active edge 결합
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)

                # inactive + active target/weight 결합
                target = torch.cat([self.target_inac[:, m], self.target], 1)
                weight = torch.cat([self.weight_inac[:, m], self.weight], 1)

            else:
                # active factor만 사용
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight

            ########################################################
            # BA 입력 shape 변환
            ########################################################

            # BA damping 계산
            # damping에 0.2를 곱하고 EP를 더해 수치 안정성 확보
            damping = 0.2 * self.damping[torch.unique(ii)].contiguous() + EP

            # target:
            #   [1, edges, ht, wd, 2]
            # -> [edges, 2, ht, wd]
            target = target.view(-1, ht, wd, 2).permute(0, 3, 1, 2).contiguous()

            # weight:
            #   [1, edges, ht, wd, 2]
            # -> [edges, 2, ht, wd]
            weight = weight.view(-1, ht, wd, 2).permute(0, 3, 1, 2).contiguous()

            ########################################################
            # Dense Bundle Adjustment
            ########################################################

            # 여기서 실제 pose와 disparity가 최적화된다.
            #
            # self.video.ba() 내부에서 droid_backends.ba()가 호출되고,
            # self.video.poses, self.video.disps가 in-place로 업데이트된다.
            ba_t1 = t1
            if ba_t1 is None:
                ba_t1 = max(ii.max().item(), jj.max().item()) + 1
            ba_t0 = t0
            aligned_for_imu = (
                not self.require_imu_alignment or self._has_imu_alignment()
            )
            joint_bias_active = bool(
                self.use_full_imu_ba
                and self._joint_shared_bias_enabled()
                and self._joint_shared_bias_due()
                and aligned_for_imu
            )
            outer_control_active = bool(
                self.use_full_imu_ba
                and self._outer_imu_ba_relinearization_due()
                and aligned_for_imu
            )
            outer_relinearize = joint_bias_active or outer_control_active
            if joint_bias_active:
                self._last_joint_bias_attempt_frame = int(self.video.counter.value)
            if outer_control_active:
                self._last_outer_relinearization_frame = int(self.video.counter.value)

            expanded_window = 0
            if joint_bias_active:
                expanded_window = self.imu_joint_bias_window
            if outer_control_active:
                expanded_window = max(expanded_window, self.imu_ba_outer_window)
            expanded_dba_window = expanded_window > 0
            if expanded_dba_window:
                # The normal frontend visual graph often starts at only the
                # latest five source frames.  A shared-bias factor needs the
                # pose/velocity states throughout its own window, otherwise
                # it degenerates to a six-frame correction.  The fixed-bias
                # outer-relinearization control can request the same range.
                ba_t0 = max(1, min(int(t0), int(ba_t1) - expanded_window))

            if expanded_dba_window and ba_t0 < int(t0):
                # CUDA forms its dense-depth diagonal over
                # unique([t0:t1), ii]).  The usual frontend range happens to
                # match unique(ii), but an expanded IMU window does not.
                # Build eta against the exact CUDA index set to keep the
                # depth Schur system and the state window consistent.
                ba_ts = torch.arange(ba_t0, ba_t1, device=ii.device, dtype=ii.dtype)
                ba_kx = torch.unique(torch.cat([ba_ts, ii], dim=0))
                damping = 0.2 * self.damping[ba_kx].contiguous() + EP

            self._run_dba_with_imu_prior(
                target=target,
                weight=weight,
                damping=damping,
                ii=ii,
                jj=jj,
                ba_t0=ba_t0,
                ba_t1=ba_t1,
                itrs=itrs,
                lm=1e-4,
                ep=0.1,
                motion_only=motion_only,
                stage="frontend",
                joint_bias_requested=joint_bias_active,
                outer_relinearize=outer_relinearize,
            )

            self._advance_tuned_confidence_schedule()

            self._apply_imu_residual("frontend")

            ########################################################
            # disparity upsampling
            ########################################################

            if self.upsample:
                self.video.upsample(torch.unique(self.ii), upmask)

        # factor age 증가
        self.age += 1

    @autocast(enabled=False)
    def update_lowmem(
        self,
        t0=None,
        t1=None,
        itrs=2,
        use_inactive=False,
        EP=1e-7,
        steps=8
    ):
        """
        low-memory 방식의 factor graph update.

        주로 DroidBackend / DroidAsyncBackend에서 사용된다.

        update()와의 차이:
            update():
                CorrBlock을 factor마다 저장해두고 사용.
                frontend local graph처럼 factor 수가 적을 때 적합.

            update_lowmem():
                AltCorrBlock을 사용해 필요한 구간의 correlation만 계산.
                backend처럼 factor 수가 많을 때 메모리 절약에 유리.

        처리 순서:
            1. 전체 feature map으로 AltCorrBlock 생성
            2. steps만큼 반복
            3. 현재 pose/depth로 reproject
            4. factor를 source index 기준으로 작은 chunk로 나눔
            5. chunk별 correlation 계산 및 update_op 수행
            6. target/weight/damping 갱신
            7. inactive 포함 여부 결정
            8. video.ba() 호출
            9. dirty flag 갱신
        """

        # 현재 video에 저장된 frame 수
        t = self.video.counter.value

        if self.ii.numel() == 0:
            return

        # 전체 feature map shape
        # num: frame 수
        # rig: monocular이면 1, stereo이면 2
        # ch: channel 수
        # ht, wd: feature map 해상도
        num, rig, ch, ht, wd = self.video.fmaps.shape

        # AltCorrBlock 생성
        # 모든 frame feature를 하나로 펴서 low-memory 방식 correlation을 계산한다.
        corr_op = AltCorrBlock(
            self.video.fmaps.view(1, num * rig, ch, ht, wd)
        )

        ############################################################
        # backend update 반복
        ############################################################

        for step in range(steps):

            with CudaTimer("backend", enabled=False):

                ####################################################
                # 현재 reprojection과 motion feature
                ####################################################

                with autocast(enabled=False):

                    # 현재 pose/depth 기준 reprojection
                    coords1, mask = self.video.reproject(self.ii, self.jj)

                    # motion feature 구성
                    motn = torch.cat(
                        [coords1 - self.coords0, self.target - coords1],
                        dim=-1
                    )

                    # [1, edges, ht, wd, 4] -> [1, edges, 4, ht, wd]
                    motn = motn.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

                ####################################################
                # factor chunk 단위 update
                ####################################################

                # 한 번에 모든 factor를 update하지 않고,
                # source frame index 기준으로 s개씩 나눠 처리한다.
                # backend factor가 많을 때 메모리 절약 목적.
                s = 8
                edge_confidence = None
                if self.use_learned_imu_prediction:
                    confidence_channels = 3 if (
                        self.use_component_imu_confidence
                        or self.use_learned_imu_precision
                    ) else 1
                    edge_confidence = torch.zeros(
                        (1, self.ii.numel(), confidence_channels),
                        device=self.device,
                        dtype=torch.float,
                    )

                for i in range(self.ii.min(), self.jj.max() + 1, s):

                    # source index가 [i, i+s) 범위에 있는 factor 선택
                    v = (self.ii >= i) & (self.ii < i + s)

                    iis = self.ii[v]
                    jjs = self.jj[v]

                    # 해당 chunk에 factor가 없으면 skip
                    if v.count_nonzero().item() == 0:
                        continue

                    ht, wd = self.coords0.shape[0:2]

                    with autocast(enabled=True):

                        # 선택된 factor chunk에 대한 correlation 계산
                        #
                        # rig * iis:
                        #   source frame feature index
                        #
                        # rig * jjs + (iis == jjs).long():
                        #   target frame feature index
                        #   stereo self-edge인 경우 right image feature 선택 가능
                        corr1 = corr_op(
                            coords1[:, v],
                            rig * iis,
                            rig * jjs + (iis == jjs).long()
                        )

                        # update network 수행
                        update_out = self.update_op(
                            self.net[:, v],
                            self.video.inps[None, iis],
                            corr1,
                            motn[:, v],
                            iis,
                            jjs,
                            return_imu_confidence=self.use_learned_imu_prediction,
                            return_component_imu_confidence=self.use_component_imu_confidence,
                            use_dvi_imu_confidence=self.use_dvi_imu_confidence,
                            use_dvi_imu_quality_fusion=self.use_dvi_imu_quality_fusion,
                            use_learned_imu_precision=self.use_learned_imu_precision,
                            imu_quality=self._edge_imu_quality(iis, jjs),
                        )
                        if self.use_learned_imu_prediction:
                            net, delta, weight, damping, upmask, imu_edge_confidence = update_out
                        else:
                            net, delta, weight, damping, upmask = update_out

                        # 필요 시 해당 source frame disparity upsample
                        if self.upsample:
                            self.video.upsample(torch.unique(iis), upmask)

                    ################################################
                    # chunk update 결과 저장
                    ################################################

                    self.net[:, v] = net
                    self.target[:, v] = coords1[:, v] + delta.float()
                    self.weight[:, v] = weight.float()
                    self.damping[torch.unique(iis)] = damping

                    if self.use_learned_imu_prediction:
                        pooled_confidence = imu_edge_confidence.float().mean(dim=(2, 3))
                        if not (
                            self.use_component_imu_confidence
                            or self.use_learned_imu_precision
                        ):
                            pooled_confidence = pooled_confidence[..., :1]
                        edge_confidence[:, v] = pooled_confidence

                if self.use_learned_imu_prediction:
                    self.last_imu_confidence, confidence_coverage = (
                        self._edge_confidence_to_frame_confidence(
                            edge_confidence, return_coverage=True
                        )
                    )
                    self._persist_imu_confidence(
                        self.last_imu_confidence, confidence_coverage
                    )

                ####################################################
                # BA 입력 구성
                ####################################################

                # active factor 기준 damping
                damping = 0.2 * self.damping[torch.unique(self.ii)].contiguous() + EP

                if use_inactive:
                    # inactive + active factor 모두 사용
                    # backend에서는 오래된 inactive factor까지 같이 사용해서
                    # 장기적인 trajectory consistency를 유지하려는 목적이 있다.
                    ii = torch.cat([self.ii_inac, self.ii], 0)
                    jj = torch.cat([self.jj_inac, self.jj], 0)
                    target = torch.cat([self.target_inac, self.target], 1)
                    weight = torch.cat([self.weight_inac, self.weight], 1)

                else:
                    # active factor만 사용
                    ii, jj, target, weight = self.ii, self.jj, self.target, self.weight

                # inactive 포함 후 damping 다시 계산
                damping = 0.2 * self.damping[torch.unique(ii)].contiguous() + EP

                # target shape:
                # [1, edges, ht, wd, 2] -> [edges, 2, ht, wd]
                target = target.view(-1, ht, wd, 2).permute(0, 3, 1, 2).contiguous()

                # weight shape:
                # [1, edges, ht, wd, 2] -> [edges, 2, ht, wd]
                weight = weight.view(-1, ht, wd, 2).permute(0, 3, 1, 2).contiguous()

                # factor age 증가
                self.age += 1

                ####################################################
                # Dense Bundle Adjustment
                ####################################################

                # backend에서는 전체 구간 [1, t]에 대해 BA 수행
                #
                # lm=1e-5, ep=1e-2:
                #   frontend update보다 약간 다른 damping/epsilon 설정
                aligned_for_imu = (
                    not self.require_imu_alignment or self._has_imu_alignment()
                )
                joint_bias_active = bool(
                    self.use_full_imu_ba
                    and self._joint_shared_bias_enabled()
                    and self._joint_shared_bias_due()
                    and aligned_for_imu
                )
                outer_control_active = bool(
                    self.use_full_imu_ba
                    and self._outer_imu_ba_relinearization_due()
                    and aligned_for_imu
                )
                outer_relinearize = joint_bias_active or outer_control_active
                if joint_bias_active:
                    self._last_joint_bias_attempt_frame = int(self.video.counter.value)
                if outer_control_active:
                    self._last_outer_relinearization_frame = int(self.video.counter.value)

                # The termination backend already optimizes [1, t), so the
                # fixed-bias control does not need a separate range expansion.
                self._run_dba_with_imu_prior(
                    target=target,
                    weight=weight,
                    damping=damping,
                    ii=ii,
                    jj=jj,
                    ba_t0=1,
                    ba_t1=t,
                    itrs=itrs,
                    lm=1e-5,
                    ep=1e-2,
                    motion_only=False,
                    stage="backend",
                    joint_bias_requested=joint_bias_active,
                    outer_relinearize=outer_relinearize,
                )

                self._advance_tuned_confidence_schedule()

                self._apply_imu_residual("backend")

                # 전체 frame이 backend update로 변경되었으므로 dirty=True
                self.video.dirty[:t] = True

    def add_neighborhood_factors(self, t0, t1, r=3):
        """
        시간적으로 가까운 frame들 사이에 edge를 추가한다.

        사용 위치:
            DroidFrontend._initialize()

        역할:
            초기 warmup frame들 사이에 인접 factor를 만들어
            초기 pose/depth를 안정화한다.

        t0, t1:
            frame index 범위 [t0, t1)

        r:
            연결할 최대 frame index 거리.
            예: r=3이면 |i-j| <= 3인 frame pair를 연결.
        """

        # [t0, t1) 범위의 모든 frame pair 생성
        ii, jj = torch.meshgrid(
            torch.arange(t0, t1, device=self.device),
            torch.arange(t0, t1, device=self.device),
            indexing="ij",
        )

        # stereo이면 같은 frame의 left-right edge를 허용하기 위해 c=1
        # monocular이면 self-edge는 제외하기 위해 c=0
        c = 1 if self.video.stereo else 0

        # 자기 자신 또는 너무 먼 frame pair 제외
        keep = ((ii - jj).abs() > c) & ((ii - jj).abs() <= r)

        # 조건을 만족하는 edge 추가
        self.add_factors(ii[keep], jj[keep])

    def add_proximity_factors(
        self,
        t0=0,
        t1=0,
        rad=2,
        nms=2,
        beta=0.25,
        thresh=16.0,
        remove=False
    ):
        """
        geometric distance 기반으로 가까운 frame pair를 찾아 edge로 추가한다.

        사용 위치:
            Frontend:
                local window 내에서 가까운 frame pair 추가.

            Backend:
                전체 또는 넓은 범위에서 가까운 frame pair 추가.

        핵심 역할:
            단순히 시간적으로 가까운 frame만 연결하는 것이 아니라,
            video.distance()로 계산한 기하학적 거리를 기준으로
            다시 관측될 가능성이 있는 frame pair를 edge로 연결한다.

        t0:
            source frame 후보 시작 index.

        t1:
            target frame 후보 시작 index.

        rad:
            기본적으로 시간상 가까운 frame들을 강제로 연결할 때 사용하는 radius.

        nms:
            edge 후보 주변을 suppression해서 비슷한 edge가 중복 추가되지 않게 하는 값.

        beta:
            video.distance() 계산에 사용되는 weighting parameter.

        thresh:
            distance가 이 값보다 작은 frame pair만 proximity edge로 선택.

        remove:
            max_factors 초과 시 오래된 factor 제거 여부.
        """

        # 현재 저장된 frame 수
        t = self.video.counter.value

        # source 후보 index: [t0, t)
        ix = torch.arange(t0, t)

        # target 후보 index: [t1, t)
        jx = torch.arange(t1, t)

        # 가능한 모든 source-target pair 생성
        ii, jj = torch.meshgrid(ix, jx, indexing="ij")
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        ############################################################
        # frame distance 계산
        ############################################################

        # pose, disparity, intrinsics 기반 geometric distance 계산
        d = self.video.distance(ii, jj, beta=beta).cpu()

        # 너무 가까운 시간 index 조합은 proximity 후보에서 제외
        # 이들은 아래에서 neighborhood edge로 강제로 추가됨
        d[ii - rad < jj] = np.inf

        # 너무 큰 distance는 후보에서 제외
        d[d > 100] = np.inf

        ############################################################
        # 이미 존재하거나 나쁜 edge 주변 suppression
        ############################################################

        # active, bad, inactive edge를 모두 모음
        # 같은 edge 또는 주변 edge가 다시 추가되지 않게 하기 위함
        ii1 = torch.cat([self.ii, self.ii_bad, self.ii_inac], 0)
        jj1 = torch.cat([self.jj, self.jj_bad, self.jj_inac], 0)

        # 기존 edge 주변의 candidate distance를 inf로 만들어 선택되지 않게 함
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            for di in range(-nms, nms + 1):
                for dj in range(-nms, nms + 1):

                    # edge 주변 suppression 범위
                    if abs(di) + abs(dj) <= max(min(abs(i - j) - 2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        # 후보 범위 안이면 distance를 inf로 만들어 제외
                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1 - t0) * (t - t1) + (j1 - t1)] = np.inf

        ############################################################
        # 기본 temporal edge 추가
        ############################################################

        es = []

        for i in range(t0, t):

            # stereo인 경우 같은 timestamp의 left-right pair 추가
            if self.video.stereo:
                es.append((i, i))
                d[(i - t0) * (t - t1) + (i - t1)] = np.inf

            # 시간적으로 가까운 frame들은 기본적으로 양방향 edge 추가
            #
            # 예:
            #   i와 i-1, i-2 정도를 연결
            #
            # 이 부분은 tracking 안정성을 위한 local temporal constraint 역할
            for j in range(max(i - rad - 1, 0), i):
                es.append((i, j))
                es.append((j, i))

                # 이미 추가했으므로 proximity 후보에서 제외
                d[(i - t0) * (t - t1) + (j - t1)] = np.inf

        ############################################################
        # proximity edge 선택
        ############################################################

        # distance가 작은 순서대로 후보 정렬
        ix = torch.argsort(d)

        for k in ix:

            # threshold보다 크면 추가하지 않음
            if d[k] > thresh:
                continue

            # max_factors 제한이 있으면 그 이상 추가하지 않음
            if self.max_factors > 0:
                if len(es) > self.max_factors:
                    break

            # 선택된 source-target frame index
            i = ii[k]
            j = jj[k]

            # 양방향 edge 추가
            # DROID는 i->j와 j->i를 모두 graph에 넣어 consistency를 높인다.
            es.append((i, j))
            es.append((j, i))

            ########################################################
            # 선택된 edge 주변 NMS
            ########################################################

            # 방금 선택한 edge 주변 후보를 suppression해서
            # 너무 비슷한 edge들이 중복으로 들어가지 않게 함
            for di in range(-nms, nms + 1):
                for dj in range(-nms, nms + 1):

                    if abs(di) + abs(dj) <= max(min(abs(i - j) - 2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1 - t0) * (t - t1) + (j1 - t1)] = np.inf

        ############################################################
        # 선택된 edge들을 factor graph에 추가
        ############################################################

        # es: [(i, j), (j, i), ...]
        # torch tensor로 바꾼 뒤 ii, jj로 분리
        if len(es) == 0:
            return

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)

        # 실제 factor 추가
        self.add_factors(ii, jj, remove)

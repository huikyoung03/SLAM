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
    quat_rotate,
    quat_to_rotvec,
    rotvec_to_quat,
)

from cuda_timer import CudaTimer
from functools import partial


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
        imu_confidence_floor=0.0,
        use_imu_ba_prior=False,
        imu_ba_prior_weight=0.0,
        imu_ba_prior_max_deg=45.0,
        use_full_imu_ba=False,
        imu_full_pos_weight=0.05,
        imu_full_vel_weight=0.05,
        imu_full_bias_weight=0.001,
        imu_motion_prior_weight=0.0,
        imu_local_bias_prior_weight=0.0,
        imu_gravity=None,
        imu_full_max_dt=0.5,
        imu_full_max_dv=5.0,
        imu_full_max_dp=1.0,
        use_imu_info_weighting=False,
        imu_info_weight_clip=4.0,
        imu_info_weight_eps=1e-12,
        imu_gyro_bias=None,
        imu_acc_bias=None,
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
        self.use_learned_imu_confidence = bool(use_learned_imu_confidence)
        self.imu_confidence_floor = float(imu_confidence_floor)
        self.use_imu_ba_prior = bool(use_imu_ba_prior)
        self.imu_ba_prior_weight = float(imu_ba_prior_weight)
        self.imu_ba_prior_max_deg = float(imu_ba_prior_max_deg)
        self.use_full_imu_ba = bool(use_full_imu_ba)
        self.imu_full_pos_weight = float(imu_full_pos_weight)
        self.imu_full_vel_weight = float(imu_full_vel_weight)
        self.imu_full_bias_weight = float(imu_full_bias_weight)
        self.imu_motion_prior_weight = float(imu_motion_prior_weight)
        self.imu_local_bias_prior_weight = float(imu_local_bias_prior_weight)
        self.imu_gravity = imu_gravity
        self.imu_full_max_dt = float(imu_full_max_dt)
        self.imu_full_max_dv = float(imu_full_max_dv)
        self.imu_full_max_dp = float(imu_full_max_dp)
        self.use_imu_info_weighting = bool(use_imu_info_weighting)
        self.imu_info_weight_clip = float(imu_info_weight_clip)
        self.imu_info_weight_eps = float(imu_info_weight_eps)
        self.imu_gyro_bias = imu_gyro_bias
        self.imu_acc_bias = imu_acc_bias
        self._reported_imu_ba_prior = False
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

    def _edge_confidence_to_frame_confidence(self, edge_confidence, ii=None, jj=None):
        """
        Convert update-head edge confidence into a per-video-frame IMU confidence.

        The learned IMU head predicts confidence per graph edge and feature cell.
        The rotation residual is temporal and applies to frame states, so we
        average each edge map and scatter those values back to the two frames
        touched by the edge.
        """

        if edge_confidence is None:
            return None

        ii = self.ii if ii is None else ii
        jj = self.jj if jj is None else jj

        if ii.numel() == 0 or jj.numel() == 0:
            return None

        confidence = edge_confidence.to(device=self.device, dtype=torch.float)
        if confidence.ndim == 1:
            confidence = confidence[None]
        elif confidence.ndim > 2:
            confidence = confidence.mean(dim=tuple(range(2, confidence.ndim)))

        frame_index = torch.cat([ii, jj], dim=0).to(device=self.device, dtype=torch.long)
        frame_values = torch.cat([confidence, confidence], dim=1)

        num_frames = max(
            int(self.video.counter.value),
            int(frame_index.max().item()) + 1,
        )
        frame_confidence = torch.ones(
            (confidence.shape[0], num_frames),
            device=self.device,
            dtype=torch.float,
        )
        sums = torch.zeros_like(frame_confidence)
        counts = torch.zeros_like(frame_confidence)
        scatter_index = frame_index[None].expand(confidence.shape[0], -1)

        sums.scatter_add_(1, scatter_index, frame_values)
        counts.scatter_add_(1, scatter_index, torch.ones_like(frame_values))
        observed = counts > 0
        frame_confidence[observed] = sums[observed] / counts[observed].clamp_min(1.0)

        if self.imu_confidence_floor > 0.0:
            floor = float(self.imu_confidence_floor)
            frame_confidence = floor + (1.0 - floor) * frame_confidence

        return frame_confidence.clamp(0.0, 1.0)

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

    def _confidence_for_pose_index(self, pose_ix):
        if not self.use_learned_imu_confidence or self.last_imu_confidence is None:
            return 1.0

        try:
            conf = self.last_imu_confidence
            if conf.ndim == 2:
                value = conf[0, pose_ix]
            else:
                value = conf.reshape(-1)[pose_ix]
            out = float(value.detach().cpu().item())
        except Exception:
            return 1.0

        if not np.isfinite(out):
            return 1.0

        return float(np.clip(out, 0.0, 1.0))

    def _record_imu_ba_debug(self, row):
        if not self.imu_ba_debug:
            return

        self._imu_ba_debug_seen += 1
        if len(self._imu_ba_debug_rows) >= self.imu_ba_debug_max_rows:
            return

        clean = {"stage": self.imu_ba_debug_stage}
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
            H[3:6, 3:6] = value * eye3
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
            if residual_deg > self.imu_ba_prior_max_deg:
                continue

            confidence = self._confidence_for_pose_index(curr_ix)
            weight = float(self.imu_ba_prior_weight) * float(row_scale) * confidence
            if weight <= 0.0 or not np.isfinite(weight):
                continue

            w = torch.as_tensor(weight, device=device, dtype=dtype)

            if 0 <= prev_local < P:
                add_rot_block(prev_local, prev_local, w)
                add_rot_block(prev_local, curr_local, -w)
                add_rot_block(curr_local, prev_local, -w)
                v_prior[prev_local, 3:6] += w * residual

            add_rot_block(curr_local, curr_local, w)
            v_prior[curr_local, 3:6] += -w * residual
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

        return delta_ac, valid, weight, used_steps, info

    def _initialize_runtime_velocity(self, t0, t1):
        if (
            getattr(self.video, "velocities", None) is None
            or getattr(self.video, "imu_delta", None) is None
        ):
            return

        n = int(self.video.counter.value)
        max_pose_ix = min(int(t1), n)
        poses = self.video.poses
        velocities = self.video.velocities
        gravity = self._global_imu_bias(self.imu_gravity, poses.device, poses.dtype)
        gravity = gravity * self._imu_metric_scale(poses.device, poses.dtype)

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
            dp = poses[pose_ix, 0:3] - poses[pose_ix - 1, 0:3]
            v = (dp - 0.5 * gravity * dt_t * dt_t) / dt_t
            if torch.linalg.norm(velocities[pose_ix - 1]).item() < 1e-9:
                velocities[pose_ix - 1] = v
            if torch.linalg.norm(velocities[pose_ix]).item() < 1e-9:
                velocities[pose_ix] = v

    def _build_full_imu_ba_prior(self, t0, t1):
        """
        Build a 15D inertial prior for CUDA DBA.

        State layout per frame:
            [pose(6), velocity(3), accel_bias(3), gyro_bias(3)]

        Residual layout:
            [r_p, r_v, r_R, r_ba, r_bg]
        """

        if (
            not self.use_full_imu_ba
            or not self.use_imu_ba_prior
            or self.imu_ba_prior_weight <= 0.0
            or getattr(self.video, "imu_delta", None) is None
        ):
            return None, None, None, None

        t0 = int(t0)
        t1 = int(t1)
        P = t1 - t0
        if P <= 0:
            return None, None, None, None

        n = int(self.video.counter.value)
        if n < 2:
            return None, None, None, None

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
        state_dim = 15

        H_blocks = []
        H_ii = []
        H_jj = []
        v_prior = torch.zeros((P, state_dim), device=device, dtype=dtype)
        eye3 = torch.eye(3, device=device, dtype=dtype)
        metric_scale = self._imu_metric_scale(device, dtype)
        ba_global = self._global_imu_bias(self.imu_acc_bias, device, dtype) * metric_scale
        bg_global = self._global_imu_bias(self.imu_gyro_bias, device, dtype)
        gravity = self._global_imu_bias(self.imu_gravity, device, dtype) * metric_scale
        base_scale = torch.as_tensor(
            [self.imu_full_pos_weight] * 3
            + [self.imu_full_vel_weight] * 3
            + [1.0] * 3
            + [self.imu_full_bias_weight] * 3
            + [self.imu_full_bias_weight] * 3,
            device=device,
            dtype=dtype,
        ).clamp_min(0.0).sqrt()
        info_ref = self._imu_info_reference(t0, max_pose_ix, device, dtype)
        rad_to_deg = 180.0 / np.pi
        used = 0
        confidence_sum = 0.0
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
                "weight": float(weight),
                "metric_scale": float(metric_scale.detach().cpu().item()),
                "gravity_x": float(gravity[0].detach().cpu().item()),
                "gravity_y": float(gravity[1].detach().cpu().item()),
                "gravity_z": float(gravity[2].detach().cpu().item()),
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

            p_i = poses[prev_ix, 0:3].detach()
            p_j = poses[curr_ix, 0:3].detach()
            q_i = poses[prev_ix, 3:7].detach()
            q_j = poses[curr_ix, 3:7].detach()

            v_i = velocities[prev_ix].detach()
            v_j = velocities[curr_ix].detach()
            ba_i = bias_acc[prev_ix].detach()
            ba_j = bias_acc[curr_ix].detach()
            bg_i = bias_gyro[prev_ix].detach()
            bg_j = bias_gyro[curr_ix].detach()

            ba_total = ba_i + ba_global
            bg_total = bg_i + bg_global

            rotvec = row[1:4] - bg_total * dt
            dv_imu = row[4:7] - ba_total * dt
            dp_imu = row[7:10] - 0.5 * ba_total * dt * dt

            q_imu = rotvec_to_quat(rotvec)
            q_pred = quat_multiply(q_i, q_imu)
            r_R = quat_to_rotvec(quat_multiply(quat_inverse(q_pred), q_j))

            q_i_inv = quat_inverse(q_i)
            r_p = quat_rotate(q_i_inv, p_j - p_i - v_i * dt - 0.5 * gravity * dt * dt) - dp_imu
            r_v = quat_rotate(q_i_inv, v_j - v_i - gravity * dt) - dv_imu
            r_ba = ba_j - ba_i
            r_bg = bg_j - bg_i
            residual = torch.cat([r_p, r_v, r_R, r_ba, r_bg], dim=0)

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
            if residual_deg > self.imu_ba_prior_max_deg:
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

            confidence = self._confidence_for_pose_index(curr_ix)
            weight = float(self.imu_ba_prior_weight) * row_scale * confidence
            if weight <= 0.0 or not np.isfinite(weight):
                record_edge(
                    "skip_weight",
                    prev_ix,
                    curr_ix,
                    dt_value=dt_value,
                    row_weight=row_weight,
                    row_scale=row_scale,
                    confidence=confidence,
                    weight=weight,
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

            info_term_scale, info_scales = self._imu_info_term_scale(
                curr_ix,
                info_ref,
                device,
                dtype,
            )

            Ji = torch.zeros((15, state_dim), device=device, dtype=dtype)
            Jj = torch.zeros((15, state_dim), device=device, dtype=dtype)

            Ji[0:3, 0:3] = eye3
            Jj[0:3, 0:3] = -eye3
            Ji[0:3, 6:9] = dt * eye3
            Ji[0:3, 9:12] = -0.5 * dt * dt * eye3

            Ji[3:6, 6:9] = eye3
            Jj[3:6, 6:9] = -eye3
            Ji[3:6, 9:12] = -dt * eye3

            Ji[6:9, 3:6] = eye3
            Jj[6:9, 3:6] = -eye3
            Ji[6:9, 12:15] = -dt * eye3

            Ji[9:12, 9:12] = eye3
            Jj[9:12, 9:12] = -eye3

            Ji[12:15, 12:15] = eye3
            Jj[12:15, 12:15] = -eye3

            scale = base_scale * info_term_scale
            residual = residual * scale
            Ji = Ji * scale.view(15, 1)
            Jj = Jj * scale.view(15, 1)
            w = torch.as_tensor(weight, device=device, dtype=dtype)

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
            record_edge(
                "used",
                prev_ix,
                curr_ix,
                dt_value=dt_value,
                row_weight=row_weight,
                row_scale=row_scale,
                confidence=confidence,
                weight=weight,
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
                info_rot_scale=info_scales[0],
                info_vel_scale=info_scales[1],
                info_pos_scale=info_scales[2],
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
                                poses[pose_ix, 0:3].detach()
                                - poses[pose_ix - 1, 0:3].detach()
                            ) / dt
                        else:
                            v_ref = torch.zeros(3, device=device, dtype=dtype)
                    else:
                        v_ref = torch.zeros(3, device=device, dtype=dtype)

                    residual_v = velocities[pose_ix].detach() - v_ref
                    wv = torch.as_tensor(self.imu_motion_prior_weight, device=device, dtype=dtype)
                    H[6:9, 6:9] += wv * eye3
                    v[6:9] += -wv * residual_v

                if self.imu_local_bias_prior_weight > 0.0:
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

        if used == 0 or len(H_blocks) == 0:
            return None, None, None, None

        if not self._reported_imu_ba_prior:
            conf_mean = confidence_sum / max(used, 1)
            print(
                f"[IMU-BA] full prior enabled: window=[{t0},{t1}), "
                f"edges={used}, blocks={len(H_blocks)}, "
                f"weight={self.imu_ba_prior_weight}, conf_mean={conf_mean:.4f}, "
                f"pos_w={self.imu_full_pos_weight}, vel_w={self.imu_full_vel_weight}, "
                f"bias_w={self.imu_full_bias_weight}, max_dt={self.imu_full_max_dt}, "
                f"info_weighting={self.use_imu_info_weighting}, "
                f"info_clip={self.imu_info_weight_clip}, "
                f"gravity={tuple(float(x) for x in gravity.detach().cpu().tolist())}, "
                f"skipped={skipped}"
            )
            self._reported_imu_ba_prior = True

        state_prior_H = torch.stack(H_blocks, dim=0).contiguous()
        state_prior_ii = torch.as_tensor(H_ii, device=device, dtype=torch.long).contiguous()
        state_prior_jj = torch.as_tensor(H_jj, device=device, dtype=torch.long).contiguous()
        return state_prior_H, v_prior.contiguous(), state_prior_ii, state_prior_jj

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
        if composed_imu is not None:
            delta_ac, valid_ac, weight_ac, used_steps_ac, info_ac = composed_imu
            self.video.imu_delta[ix] = delta_ac
            self.video.imu_valid[ix] = valid_ac
            self.video.imu_weight[ix] = weight_ac
            self.video.imu_used_steps[ix] = used_steps_ac
            self.video.imu_info[ix] = info_ac
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
            return_imu_confidence=self.use_learned_imu_confidence,
        )
        if self.use_learned_imu_confidence:
            self.net, delta, weight, damping, upmask, imu_edge_confidence = update_out
            self.last_imu_confidence = self._edge_confidence_to_frame_confidence(
                imu_edge_confidence
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

            if self.use_full_imu_ba:
                pose_prior_H, pose_prior_v, pose_prior_ii, pose_prior_jj = (
                    self._build_full_imu_ba_prior(t0, ba_t1)
                )
            else:
                pose_prior_H, pose_prior_v, pose_prior_ii, pose_prior_jj = (
                    self._build_imu_ba_pose_prior(t0, ba_t1)
                )
            self.video.ba(
                target,
                weight,
                damping,
                ii,
                jj,
                t0,
                t1,
                itrs=itrs,
                lm=1e-4,
                ep=0.1,
                motion_only=motion_only,
                pose_prior_H=pose_prior_H,
                pose_prior_v=pose_prior_v,
                pose_prior_ii=pose_prior_ii,
                pose_prior_jj=pose_prior_jj,
            )

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
                if self.use_learned_imu_confidence:
                    edge_confidence = torch.zeros(
                        (1, self.ii.numel()),
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
                            return_imu_confidence=self.use_learned_imu_confidence,
                        )
                        if self.use_learned_imu_confidence:
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

                    if self.use_learned_imu_confidence:
                        edge_confidence[:, v] = imu_edge_confidence.float().mean(
                            dim=(2, 3, 4)
                        )

                if self.use_learned_imu_confidence:
                    self.last_imu_confidence = self._edge_confidence_to_frame_confidence(
                        edge_confidence
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
                if self.use_full_imu_ba:
                    pose_prior_H, pose_prior_v, pose_prior_ii, pose_prior_jj = (
                        self._build_full_imu_ba_prior(1, t)
                    )
                else:
                    pose_prior_H, pose_prior_v, pose_prior_ii, pose_prior_jj = (
                        self._build_imu_ba_pose_prior(1, t)
                    )
                self.video.ba(
                    target,
                    weight,
                    damping,
                    ii,
                    jj,
                    1,
                    t,
                    itrs=itrs,
                    lm=1e-5,
                    ep=1e-2,
                    motion_only=False,
                    pose_prior_H=pose_prior_H,
                    pose_prior_v=pose_prior_v,
                    pose_prior_ii=pose_prior_ii,
                    pose_prior_jj=pose_prior_jj,
                )

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

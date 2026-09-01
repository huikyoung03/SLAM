import cv2
import torch
import lietorch
import math

from collections import OrderedDict
from droid_net import DroidNet

import geom.projective_ops as pops
from modules.corr import CorrBlock

from functools import partial


"""
MotionFilter의 원래 역할:
    - 모든 입력 프레임을 DepthVideo에 저장하지 않는다.
    - 이전 selected frame과 현재 frame 사이의 visual motion을 추정한다.
    - visual motion이 threshold보다 크면 현재 frame을 DepthVideo에 append한다.
    - motion이 작으면 중복 프레임으로 판단하고 skip한다.

이번 수정본에서 추가된 IMU 역할:
    1. IMU 회전량을 이용해 visual motion threshold를 동적으로 낮춘다.
    2. IMU 회전량이 충분히 크면 visual motion이 작아도 frame을 유지한다.
    3. gyro preintegration 결과를 이용해 다음 frame의 pose 초기값을 만든다.
    4. 선택적으로 IMU translation prior도 사용할 수 있게 되어 있다.

즉, 기존:
    visual motion > threshold

수정:
    visual motion > imu_adjusted_threshold
    또는 IMU rotation > imu_rot_thresh
    또는 force_all_frames=True
"""


# PyTorch 버전에 따라 autocast 사용 방식 분기
# torch 2.x 이상에서는 torch.autocast(device_type="cuda") 사용
# 그 이전 버전에서는 torch.cuda.amp.autocast 사용
if torch.__version__.startswith("2"):
    autocast = partial(torch.autocast, device_type="cuda")
else:
    autocast = torch.cuda.amp.autocast


class MotionFilter:
    """
    incoming frame을 필터링하고 feature를 추출하는 클래스.

    원본 DROID-SLAM에서의 역할:
        - 새 이미지가 들어오면 feature map을 추출한다.
        - 이전 selected frame과 현재 frame의 correlation을 계산한다.
        - update network를 1회 실행하여 대략적인 2D motion delta를 얻는다.
        - delta 평균 크기가 threshold보다 크면 DepthVideo에 추가한다.

    IMU 결합 후 역할:
        - visual motion뿐 아니라 IMU gyro 적분 회전량도 함께 본다.
        - IMU 회전이 크면 threshold를 낮추거나 frame을 강제로 유지한다.
        - 선택된 frame에 대해 IMU 기반 pose prior를 생성하여 video.append에 넘긴다.
    """

    def __init__(self, net, video, thresh=2.5, device="cuda", args=None):
        """
        net:
            DROID-SLAM의 DroidNet.
            cnet, fnet, update module을 포함한다.

        video:
            DepthVideo 객체.
            selected frame, pose, disparity, intrinsics, feature map 등이 저장된다.

        thresh:
            visual motion 기준 frame append threshold.
            원본 DROID에서는 delta.norm().mean() > thresh이면 frame을 append한다.

        args:
            이번 수정본에서는 IMU 관련 옵션을 args에서 읽어온다.
            args가 없거나 해당 값이 없으면 getattr(..., default)를 통해 기본값을 사용한다.
        """

        ############################################################
        # 1. DROID 네트워크 모듈 분리
        ############################################################

        # context feature encoder
        # update block의 hidden state와 context input을 만든다.
        self.cnet = net.cnet

        # correlation feature encoder
        # frame 간 CorrBlock을 만들 때 사용할 feature map을 추출한다.
        self.fnet = net.fnet

        # recurrent update module
        # MotionFilter에서는 1회만 호출해서 대략적인 2D motion delta를 추정한다.
        self.update = net.update

        ############################################################
        # 2. 기본 MotionFilter 상태
        ############################################################

        # SLAM 상태 저장 buffer
        self.video = video

        # visual motion threshold
        self.thresh = thresh

        # device 설정
        self.device = device

        # 연속 skip frame count
        # 현재 코드에서는 증가만 하고 별도 trigger에는 사용하지 않는다.
        self.count = 0

        ############################################################
        # 3. IMU frame filtering 관련 옵션
        ############################################################

        # IMU를 frame selection에 사용할지 여부
        #
        # False:
        #   원본 DROID처럼 visual motion만 보고 frame append 판단
        #
        # True:
        #   IMU rotation norm을 이용해 threshold를 낮추거나 frame을 강제로 유지
        self.use_imu_filter = getattr(args, "use_imu_filter", False)

        # 모든 frame을 강제로 유지하는 debug 옵션
        # frame filtering 효과를 제거하고 전체 frame을 넣어보고 싶을 때 사용
        self.force_all_frames = getattr(args, "force_all_frames", False)

        # IMU rotation norm이 이 값 이상이면 frame을 강제로 유지
        #
        # 단위:
        #   dr_x, dr_y, dr_z가 radian axis-angle 누적값이라면 rad 단위
        #
        # 기본값 0.035 rad는 약 2도 정도
        self.imu_rot_thresh = getattr(args, "imu_rot_thresh", 0.035)

        # IMU 회전량에 따라 visual threshold를 얼마나 낮출지 결정하는 gain
        #
        # adjusted_threshold = self.thresh - imu_filter_gain * imu_rot_norm
        #
        # 값이 클수록 IMU 회전이 조금만 있어도 threshold가 크게 낮아진다.
        self.imu_filter_gain = getattr(args, "imu_filter_gain", 20.0)

        # IMU 때문에 threshold가 너무 낮아지는 것을 방지하는 하한값
        #
        # 예:
        #   self.thresh=2.5이고 IMU 회전이 크면 adjusted threshold가 0 이하가 될 수 있음.
        #   이때 최소 0.5는 유지하도록 제한한다.
        self.imu_min_filter_thresh = getattr(args, "imu_min_filter_thresh", 0.5)

        ############################################################
        # 4. IMU pose prior 관련 옵션
        ############################################################

        # gyro preintegration 기반 rotation prior를 pose 초기값에 사용할지 여부
        #
        # True이면 build_imu_pose_prior()에서 이전 pose에 IMU delta rotation을 합성한다.
        self.use_imu_pose_prior = getattr(args, "use_imu_pose_prior", False)

        # IMU rotation prior 강도
        #
        # 1.0:
        #   IMU dr_x, dr_y, dr_z를 그대로 사용
        #
        # 0.1:
        #   IMU rotation을 10%만 반영
        #
        # 이 값은 너무 크게 두면 visual BA가 수렴하기 전에 pose 초기값이 과하게 틀어질 수 있다.
        self.imu_pose_rot_weight = getattr(args, "imu_pose_rot_weight", 1.0)

        # 한 frame에 반영할 IMU 회전량의 최대 norm
        #
        # IMU spike나 timestamp mismatch 때문에 너무 큰 회전이 들어가는 것을 방지한다.
        # 기본값 0.35 rad는 약 20도 정도다.
        self.imu_pose_max_rot = getattr(args, "imu_pose_max_rot", 0.35)

        # IMU 좌표계와 카메라/DROID 좌표계 축 방향이 다를 수 있으므로
        # x, y, z 축별 sign을 조정할 수 있게 한 옵션
        #
        # 예:
        #   IMU x축이 카메라 x축과 반대라면 imu_axis_sign_x = -1
        self.imu_axis_sign_x = getattr(args, "imu_axis_sign_x", 1.0)
        self.imu_axis_sign_y = getattr(args, "imu_axis_sign_y", 1.0)
        self.imu_axis_sign_z = getattr(args, "imu_axis_sign_z", 1.0)

        # IMU delta rotation을 inverse해서 사용할지 여부
        #
        # IMU preintegration 방향이 DROID pose convention과 반대일 경우 사용한다.
        # 예:
        #   dq가 prev->curr이 아니라 curr->prev 기준이면 inverse가 필요할 수 있다.
        self.imu_pose_inverse = getattr(args, "imu_pose_inverse", False)

        ############################################################
        # 5. IMU translation prior 관련 옵션
        ############################################################

        # IMU translation prior 사용 여부
        #
        # 주의:
        #   가속도 기반 translation은 gravity, bias, scale 문제가 있기 때문에
        #   rotation prior보다 훨씬 위험하다.
        #
        # 권장:
        #   초기 실험에서는 False 또는 weight=0.0으로 두는 것이 안전하다.
        self.use_imu_translation_prior = getattr(args, "use_imu_translation_prior", False)

        # IMU dp_x, dp_y, dp_z를 pose translation에 얼마나 반영할지 결정
        self.imu_translation_weight = getattr(args, "imu_translation_weight", 0.0)

        # 한 frame에 반영할 translation prior의 최대 크기
        # 단안 DROID의 scale과 IMU dp scale이 맞지 않을 수 있으므로 작게 제한한다.
        self.imu_translation_max = getattr(args, "imu_translation_max", 0.05)

        # 전체 frame_index -> imu_prior row lookup.
        # MotionFilter는 입력 frame 일부만 DepthVideo에 append하므로, 선택된 keyframe
        # 사이가 여러 입력 frame을 건너뛸 수 있다. 이 dict가 있으면 마지막 selected
        # frame부터 현재 frame까지의 IMU prior를 합성해서 저장/pose prior에 사용한다.
        self.imu_priors = getattr(args, "imu_priors", None)

        ############################################################
        # 6. 이미지 정규화 값
        ############################################################

        # DROID 네트워크는 ImageNet normalization된 RGB 입력을 기대한다.
        self.MEAN = torch.as_tensor(
            [0.485, 0.456, 0.406],
            device=self.device
        )[:, None, None]

        self.STDV = torch.as_tensor(
            [0.229, 0.224, 0.225],
            device=self.device
        )[:, None, None]

    @autocast(enabled=True)
    def __context_encoder(self, image):
        """
        context feature 추출 함수.

        cnet 출력:
            256 channel

        이를 128 / 128로 나누어:
            net:
                recurrent update block의 hidden state

            inp:
                update block에 계속 주입되는 context input

        MotionFilter에서는 selected frame이 바뀔 때마다
        이 net, inp를 self.net, self.inp로 저장한다.
        """

        net, inp = self.cnet(image).split([128, 128], dim=2)

        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @autocast(enabled=True)
    def __feature_encoder(self, image):
        """
        correlation feature map 추출 함수.

        fnet 출력은 이전 selected frame과 현재 frame 사이의
        CorrBlock을 만들 때 사용된다.

        MotionFilter에서는 이 feature를 이용해
        현재 frame을 사용할지 말지 판단한다.
        """

        return self.fnet(image).squeeze(0)

    ############################################################
    # IMU helper functions
    ############################################################

    def _imu_is_usable(self, imu_prior):
        """
        현재 frame에 들어온 imu_prior가 사용할 수 있는 상태인지 검사한다.

        imu_prior는 보통 imu_prior.csv의 한 row를 dict 형태로 넘긴 값이라고 보면 된다.

        사용 가능 조건:
            1. imu_prior가 None이 아니어야 함
            2. imu_valid가 0이 아니어야 함
            3. dt가 0보다 커야 함
            4. imu_count가 최소 2 이상이어야 함
            5. 실제 사용된 IMU step 수가 1 이상이어야 함

        이 검사를 통과하지 못하면 IMU filter나 pose prior에 사용하지 않는다.
        """

        if imu_prior is None:
            return False

        # imu_valid=0이면 preintegration 실패 또는 유효하지 않은 구간
        if int(imu_prior.get("imu_valid", 1)) == 0:
            return False

        # dt가 0 이하이면 시간 간격이 비정상적이므로 사용 불가
        if float(imu_prior.get("dt", 0.0)) <= 0.0:
            return False

        # IMU sample 수가 너무 적으면 적분값 신뢰도가 낮음
        if int(imu_prior.get("imu_count", 0)) < 2:
            return False

        # imu_used_steps가 있으면 이를 사용하고,
        # 없으면 imu_count를 fallback으로 사용
        if int(imu_prior.get("imu_used_steps", imu_prior.get("imu_count", 0))) < 1:
            return False

        return True

    def compute_imu_rotation_norm(self, imu_prior):
        """
        IMU gyro preintegration 결과인 dr_x, dr_y, dr_z의 norm을 계산한다.

        dr_x, dr_y, dr_z:
            frame 사이의 누적 회전량을 axis-angle vector 형태로 표현한 값으로 해석한다.

        반환값:
            sqrt(dr_x^2 + dr_y^2 + dr_z^2)

        의미:
            이전 frame에서 현재 frame까지 IMU가 측정한 회전량 크기.
            MotionFilter에서는 이 값을 이용해:
                - visual threshold를 낮추거나
                - frame을 강제로 유지한다.
        """

        if not self._imu_is_usable(imu_prior):
            return 0.0

        try:
            dr_x = float(imu_prior.get("dr_x", 0.0))
            dr_y = float(imu_prior.get("dr_y", 0.0))
            dr_z = float(imu_prior.get("dr_z", 0.0))

            return math.sqrt(dr_x * dr_x + dr_y * dr_y + dr_z * dr_z)

        except Exception:
            # 값이 없거나 문자열 변환 실패 시 IMU를 사용하지 않는 것으로 처리
            return 0.0

    def get_imu_adjusted_threshold(self, imu_rot_norm):
        """
        IMU 회전량에 따라 visual motion threshold를 조정한다.

        원본 DROID:
            threshold = self.thresh

        수정 후:
            threshold = self.thresh - gain * imu_rotation_norm

        의미:
            IMU 회전량이 클수록 현재 frame이 실제로 움직였을 가능성이 높으므로,
            visual motion이 작게 나와도 frame을 유지하기 쉽게 threshold를 낮춘다.

        단, threshold가 너무 낮아지면 거의 모든 frame이 추가될 수 있으므로
        imu_min_filter_thresh로 하한을 둔다.
        """

        if not self.use_imu_filter or self.imu_filter_gain == 0.0:
            return self.thresh

        adjusted = self.thresh - self.imu_filter_gain * imu_rot_norm

        return max(adjusted, self.imu_min_filter_thresh)

    def axis_angle_to_quat(self, rx, ry, rz):
        """
        axis-angle 회전 벡터를 quaternion으로 변환한다.

        입력:
            rx, ry, rz:
                axis-angle vector.
                방향은 회전축, norm은 회전각이다.

        출력:
            quaternion [x, y, z, w]

        사용 위치:
            build_imu_pose_prior()에서 IMU gyro 적분 회전량을
            DROID pose의 quaternion에 합성하기 위해 사용한다.
        """

        angle = math.sqrt(rx * rx + ry * ry + rz * rz)

        # 회전량이 거의 없으면 identity quaternion 반환
        if angle < 1e-12:
            return [0.0, 0.0, 0.0, 1.0]

        # 회전축 normalize
        ax = rx / angle
        ay = ry / angle
        az = rz / angle

        half = 0.5 * angle
        s = math.sin(half)
        c = math.cos(half)

        return [ax * s, ay * s, az * s, c]

    def quat_multiply(self, q1, q2):
        """
        quaternion 곱셈.

        입력:
            q1, q2:
                [x, y, z, w] 순서의 quaternion

        출력:
            q1 * q2

        현재 코드에서는:
            q_new = quat_multiply(q_prev, q_delta)

        즉, 이전 pose 회전에 IMU delta rotation을 합성한다.

        주의:
            q_prev * q_delta가 맞는지,
            q_delta * q_prev가 맞는지는 좌표계 convention에 따라 달라질 수 있다.
            결과가 반대로 회전하면 곱셈 순서 또는 imu_pose_inverse 옵션을 확인해야 한다.
        """

        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

        # quaternion normalize
        norm = math.sqrt(x * x + y * y + z * z + w * w)

        if norm < 1e-12:
            return [0.0, 0.0, 0.0, 1.0]

        return [x / norm, y / norm, z / norm, w / norm]

    def quat_inverse(self, q):
        """
        unit quaternion inverse.

        q = [x, y, z, w]일 때 inverse는 [-x, -y, -z, w].

        IMU preintegration 방향이 DROID pose update 방향과 반대일 때 사용한다.
        """

        x, y, z, w = q
        return [-x, -y, -z, w]

    def _clamp_vec3(self, x, y, z, max_norm):
        """
        3D vector의 norm을 max_norm 이하로 제한한다.

        사용 이유:
            IMU spike, timestamp mismatch, preintegration 오류 때문에
            비정상적으로 큰 회전/이동 prior가 들어가면 DROID pose가 망가질 수 있다.

        rotation prior:
            imu_pose_max_rot으로 제한

        translation prior:
            imu_translation_max로 제한
        """

        norm = math.sqrt(x * x + y * y + z * z)

        if norm > max_norm and norm > 1e-12:
            scale = max_norm / norm
            return x * scale, y * scale, z * scale

        return x, y, z

    def _frame_index_from_tstamp(self, tstamp):
        try:
            value = float(tstamp)
            rounded = int(round(value))
            if abs(value - rounded) < 1e-3:
                return rounded
        except Exception:
            pass

        return None

    def _prior_delta_quat(self, imu_prior):
        if imu_prior is None:
            return [0.0, 0.0, 0.0, 1.0]

        has_dq = all(
            imu_prior.get(k) not in (None, "")
            for k in ("dq_x", "dq_y", "dq_z", "dq_w")
        )

        if has_dq:
            q = [
                float(imu_prior.get("dq_x", 0.0)),
                float(imu_prior.get("dq_y", 0.0)),
                float(imu_prior.get("dq_z", 0.0)),
                float(imu_prior.get("dq_w", 1.0)),
            ]
        else:
            q = self.axis_angle_to_quat(
                float(imu_prior.get("dr_x", 0.0)),
                float(imu_prior.get("dr_y", 0.0)),
                float(imu_prior.get("dr_z", 0.0)),
            )

        norm = math.sqrt(sum(v * v for v in q))
        if norm < 1e-12:
            return [0.0, 0.0, 0.0, 1.0]

        return [v / norm for v in q]

    def _quat_to_rotvec(self, q):
        x, y, z, w = q

        if w < 0.0:
            x, y, z, w = -x, -y, -z, -w

        s = math.sqrt(x * x + y * y + z * z)
        if s < 1e-12:
            return 2.0 * x, 2.0 * y, 2.0 * z

        angle = 2.0 * math.atan2(s, w)
        scale = angle / s
        return x * scale, y * scale, z * scale

    def compose_imu_prior_for_selected_interval(self, curr_tstamp, fallback_imu_prior):
        """
        마지막 selected frame부터 현재 입력 frame까지의 IMU prior를 합성한다.

        demo.py는 frame마다 한 row짜리 prior(i-1 -> i)를 넘긴다. 하지만 MotionFilter가
        중간 frame을 skip하면 DepthVideo의 새 keyframe은 마지막 selected frame에서
        현재 selected frame까지의 누적 IMU 측정값을 가져야 한다. 이 함수는 전체
        imu_prior lookup이 있을 때 그 누적 row를 만든다.
        """

        if not self.imu_priors or self.video.counter.value <= 0:
            return fallback_imu_prior

        prev_tstamp = self.video.tstamp[self.video.counter.value - 1].detach().cpu().item()
        prev_index = self._frame_index_from_tstamp(prev_tstamp)
        curr_index = self._frame_index_from_tstamp(curr_tstamp)

        if prev_index is None or curr_index is None or curr_index <= prev_index:
            return fallback_imu_prior

        rows = []
        for frame_index in range(prev_index + 1, curr_index + 1):
            row = self.imu_priors.get(frame_index)
            if row is None:
                return fallback_imu_prior
            rows.append(row)

        q_total = [0.0, 0.0, 0.0, 1.0]
        dt = 0.0
        imu_count = 0
        imu_used_steps = 0
        valid = 1
        imu_weight = 1.0
        dv = [0.0, 0.0, 0.0]
        dp = [0.0, 0.0, 0.0]
        reasons = []

        for row in rows:
            q_total = self.quat_multiply(q_total, self._prior_delta_quat(row))
            dt += float(row.get("dt", 0.0))
            imu_count += int(float(row.get("imu_count", 0)))
            imu_used_steps += int(float(row.get("imu_used_steps", row.get("imu_count", 0))))
            valid = valid and int(row.get("imu_valid", 1)) != 0
            imu_weight = min(imu_weight, float(row.get("imu_weight", 1.0)))
            reasons.append(str(row.get("imu_reason", "ok")))

            for i, key in enumerate(("dv_x", "dv_y", "dv_z")):
                dv[i] += float(row.get(key, 0.0))

            for i, key in enumerate(("dp_x", "dp_y", "dp_z")):
                dp[i] += float(row.get(key, 0.0))

        dr_x, dr_y, dr_z = self._quat_to_rotvec(q_total)
        out = dict(rows[-1])
        out.update({
            "prev_frame_index": prev_index,
            "frame_index": curr_index,
            "dt": dt,
            "imu_count": imu_count,
            "imu_used_steps": imu_used_steps,
            "imu_valid": 1 if valid else 0,
            "imu_weight": imu_weight,
            "imu_reason": "ok" if all(r == "ok" for r in reasons) else "composed_with_invalid_reason",
            "dr_x": dr_x,
            "dr_y": dr_y,
            "dr_z": dr_z,
            "dq_x": q_total[0],
            "dq_y": q_total[1],
            "dq_z": q_total[2],
            "dq_w": q_total[3],
            "dv_x": dv[0],
            "dv_y": dv[1],
            "dv_z": dv[2],
            "dp_x": dp[0],
            "dp_y": dp[1],
            "dp_z": dp[2],
        })

        return out

    def build_imu_pose_prior(self, imu_prior):
        """
        IMU preintegration 결과를 이용해 현재 frame의 pose 초기값을 생성한다.

        원본 DROID:
            두 번째 frame 이후 append 시 pose=None을 넘긴다.
            그러면 DepthVideo/Frontend 쪽에서 이전 pose 등을 기반으로 초기화한다.

        수정 후:
            IMU prior가 유효하고 옵션이 켜져 있으면,
            마지막 selected frame의 pose에 IMU delta rotation 또는 translation을 더해서
            pose_prior를 만든 뒤 video.append에 넘긴다.

        반환:
            pose_prior tensor [tx, ty, tz, qx, qy, qz, qw]
            또는 None

        핵심:
            이 함수는 정식 IMU residual을 BA에 넣는 것이 아니다.
            BA가 시작되기 전 pose 초기값을 IMU 기반으로 보정하는 방식이다.
        """

        # IMU 데이터가 유효하지 않으면 prior 생성 안 함
        if not self._imu_is_usable(imu_prior):
            return None

        # rotation prior와 translation prior 모두 꺼져 있으면 생성할 필요 없음
        if not self.use_imu_pose_prior and not self.use_imu_translation_prior:
            return None

        # 이전 pose가 없으면 prior를 만들 수 없음
        if self.video.counter.value <= 0:
            return None

        try:
            ########################################################
            # 1. 마지막 selected frame의 pose 가져오기
            ########################################################

            last_index = self.video.counter.value - 1

            # DepthVideo에 저장된 마지막 pose를 CPU로 복사
            # pose layout은 [tx, ty, tz, qx, qy, qz, qw]로 사용하고 있음
            prev_pose = self.video.poses[last_index].detach().clone().cpu()

            tx = float(prev_pose[0])
            ty = float(prev_pose[1])
            tz = float(prev_pose[2])

            q_prev = [
                float(prev_pose[3]),
                float(prev_pose[4]),
                float(prev_pose[5]),
                float(prev_pose[6]),
            ]

            ########################################################
            # 2. IMU rotation prior 적용
            ########################################################

            if self.use_imu_pose_prior:
                # IMU axis-angle 누적 회전량 읽기
                # 축 방향 보정을 위해 axis_sign을 곱함
                dr_x = float(imu_prior.get("dr_x", 0.0)) * float(self.imu_axis_sign_x)
                dr_y = float(imu_prior.get("dr_y", 0.0)) * float(self.imu_axis_sign_y)
                dr_z = float(imu_prior.get("dr_z", 0.0)) * float(self.imu_axis_sign_z)

                # IMU rotation prior 반영 강도 조절
                dr_x *= float(self.imu_pose_rot_weight)
                dr_y *= float(self.imu_pose_rot_weight)
                dr_z *= float(self.imu_pose_rot_weight)

                # 과도한 회전량 제한
                dr_x, dr_y, dr_z = self._clamp_vec3(
                    dr_x,
                    dr_y,
                    dr_z,
                    float(self.imu_pose_max_rot),
                )

                # axis-angle -> quaternion
                q_delta = self.axis_angle_to_quat(dr_x, dr_y, dr_z)

                # 필요하면 inverse 적용
                if self.imu_pose_inverse:
                    q_delta = self.quat_inverse(q_delta)

                # 이전 pose quaternion에 IMU delta rotation 합성
                q_new = self.quat_multiply(q_prev, q_delta)

            else:
                # rotation prior를 쓰지 않으면 이전 quaternion 유지
                q_new = q_prev

            ########################################################
            # 3. IMU translation prior 적용
            ########################################################

            if self.use_imu_translation_prior and self.imu_translation_weight != 0.0:
                # IMU preintegration translation 읽기
                dp_x = float(imu_prior.get("dp_x", 0.0)) * float(self.imu_axis_sign_x)
                dp_y = float(imu_prior.get("dp_y", 0.0)) * float(self.imu_axis_sign_y)
                dp_z = float(imu_prior.get("dp_z", 0.0)) * float(self.imu_axis_sign_z)

                # translation prior 반영 강도 조절
                dp_x *= float(self.imu_translation_weight)
                dp_y *= float(self.imu_translation_weight)
                dp_z *= float(self.imu_translation_weight)

                # 과도한 translation 제한
                dp_x, dp_y, dp_z = self._clamp_vec3(
                    dp_x,
                    dp_y,
                    dp_z,
                    float(self.imu_translation_max),
                )

                # 이전 translation에 IMU translation prior 추가
                tx += dp_x
                ty += dp_y
                tz += dp_z

            ########################################################
            # 4. DROID pose tensor 형식으로 반환
            ########################################################

            return torch.tensor(
                [tx, ty, tz, q_new[0], q_new[1], q_new[2], q_new[3]],
                dtype=self.video.poses.dtype,
                device=self.video.poses.device,
            )

        except Exception as e:
            # IMU prior 생성 중 오류가 나더라도 SLAM 전체가 죽지 않도록 None 반환
            print(f"[IMU PosePrior WARNING] failed to build pose prior: {e}")
            return None

    @autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None, imu_prior=None):
        """
        입력 frame마다 호출되는 MotionFilter main update.

        핵심 판단:
            1. visual_motion > IMU-adjusted threshold 이면 frame append
            2. use_imu_filter=True이고 IMU 회전량이 imu_rot_thresh 이상이면 frame append
            3. force_all_frames=True이면 frame append

        IMU prior는 두 가지 경로로 쓰인다.
            1. MotionFilter의 frame selection 판단
            2. 옵션이 켜진 경우 pose 초기값 생성

        그리고 append 시 DepthVideo에도 그대로 저장해둔다. 이 값은 아직 CUDA DBA에서
        직접 최적화되지는 않지만, 이후 E^u(T, M)를 실제 BA에 넣을 때 입력 측정값으로
        사용할 수 있다.
        """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        image = image.cuda()

        # normalize images
        inputs = image[None, :, [2, 1, 0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)

        # always add first frame to the depth video
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:, [0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            self.video.append(
                tstamp,
                image[0],
                Id,
                1.0,
                depth,
                intrinsics / 8.0,
                gmap,
                net[0, 0],
                inp[0, 0],
                imu_prior,
            )
            return

        # index correlation volume
        coords0 = pops.coords_grid(ht, wd, device=self.device)[None, None]
        corr = CorrBlock(self.fmap[None, [0]], gmap[None, [0]])(coords0)

        # approximate flow magnitude using 1 update iteration
        _, delta, weight = self.update(self.net[None], self.inp[None], corr)
        visual_motion = delta.norm(dim=-1).mean().item()

        selected_imu_prior = self.compose_imu_prior_for_selected_interval(
            tstamp,
            imu_prior,
        )
        imu_rot_norm = self.compute_imu_rotation_norm(selected_imu_prior)
        adjusted_thresh = self.get_imu_adjusted_threshold(imu_rot_norm)
        imu_forces_frame = (
            self.use_imu_filter
            and imu_rot_norm >= float(self.imu_rot_thresh)
        )

        should_append = (
            visual_motion > adjusted_thresh
            or imu_forces_frame
            or self.force_all_frames
        )

        if should_append:
            self.count = 0
            net, inp = self.__context_encoder(inputs[:, [0]])
            self.net, self.inp, self.fmap = net, inp, gmap

            pose_prior = self.build_imu_pose_prior(selected_imu_prior)

            self.video.append(
                tstamp,
                image[0],
                pose_prior,
                None,
                depth,
                intrinsics / 8.0,
                gmap,
                net[0],
                inp[0],
                selected_imu_prior,
            )

        else:
            self.count += 1

import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller
from vi_initialization import (
    apply_visual_inertial_alignment,
    recenter_full_imu_biases,
    record_window_bias_refinement,
    record_imu_state_diagnostics,
    refine_global_bias_15dof,
    try_visual_inertial_initialization,
)
from imu_runtime import (
    apply_learned_imu_ba_weight,
    apply_learned_imu_residual_weights,
    configure_confidence_only_imu_weighting,
    configure_learned_imu_precision_weighting,
    configure_imu_noise_model,
)

from collections import OrderedDict
from torch.multiprocessing import Process


'''
droid_async.py와 달리,
하나의 DepthVideo를 기준으로
MotionFilter → Frontend → Backend → TrajectoryFiller를 순차적으로 실행하는 기본 DROID-SLAM 실행 클래스야.


demo.py
  ↓
Droid(args)
  ↓
DepthVideo 생성
  ↓
MotionFilter 생성
  ↓
DroidFrontend 생성
  ↓
DroidBackend 생성
  ↓
입력 frame마다 Droid.track()
  ↓
MotionFilter.track()
  ↓
DroidFrontend()
  ↓
모든 frame 처리 후 Droid.terminate()
  ↓
DroidBackend 2회 실행
  ↓
PoseTrajectoryFiller
  ↓
최종 trajectory 반환



'''

class Droid:
    def __init__(self, args):
        """
        DROID-SLAM 기본 동기식 실행 클래스.

        역할:
            1. pretrained DROID network weight를 로드한다.
            2. SLAM 상태를 저장할 DepthVideo buffer를 만든다.
            3. MotionFilter를 생성해 입력 frame을 선별한다.
            4. DroidFrontend를 생성해 실시간 local tracking / local BA를 수행한다.
            5. DroidBackend를 생성해 종료 시 global refinement를 수행한다.
            6. Visualizer process를 선택적으로 실행한다.
            7. PoseTrajectoryFiller를 통해 non-keyframe pose까지 채운다.

        DroidAsync와의 차이:
            - Droid:
                video buffer가 하나만 있음.
                frontend와 backend가 같은 video를 사용.
                backend는 terminate() 시점에 순차적으로 실행됨.

            - DroidAsync:
                video1, video2를 나눔.
                backend가 별도 process에서 비동기적으로 실행됨.

        args:
            demo.py에서 argparse로 받은 실행 옵션.
        """

        # 부모 클래스 초기화
        super(Droid, self).__init__()

        ############################################################
        # 1. 네트워크 weight 로드
        ############################################################

        # self.net에 DroidNet을 만들고 pretrained weight를 로드한다.
        self.load_weights(args.weights)

        # 실행 옵션 저장
        self.args = args
        if getattr(self.args, "use_component_imu_confidence", False):
            loaded_keys = getattr(self.net, "_loaded_state_keys", set())
            has_component_head = (
                any(
                    key.startswith("update.imu_component_confidence.")
                    for key in loaded_keys
                )
                and any(
                    key.startswith("update.imu_quality_fusion.")
                    for key in loaded_keys
                )
            )
            if not has_component_head:
                self.net.initialize_component_imu_confidence_from_scalar()
                print(
                    "[IMU] initialized component confidence from scalar checkpoint; "
                    "quality fusion is neutral until fine-tuned"
                )
        if getattr(self.args, "use_dvi_imu_quality_fusion", False):
            loaded_keys = getattr(self.net, "_loaded_state_keys", set())
            has_dvi_quality_head = any(
                key.startswith("update.dvi_imu_quality_fusion.")
                for key in loaded_keys
            )
            if not has_dvi_quality_head:
                self.net.initialize_dvi_imu_quality_fusion()
                print(
                    "[IMU] initialized scalar DVI IMU-quality fusion from "
                    "an older checkpoint; quality contribution is neutral"
                )
        configure_imu_noise_model(self.args, source="droid")
        configure_learned_imu_precision_weighting(self.args, source="droid")
        configure_confidence_only_imu_weighting(self.args, source="droid")
        apply_learned_imu_ba_weight(self.args, self.net, source="droid")
        apply_learned_imu_residual_weights(self.args, self.net, source="droid")

        # visualization 비활성화 여부
        self.disable_vis = args.disable_vis

        ############################################################
        # 2. DepthVideo 생성
        ############################################################

        # store images, depth, poses, intrinsics (shared between processes)
        #
        # DepthVideo는 DROID-SLAM의 핵심 상태 저장소이다.
        # 여기에 선택된 frame, pose, disparity, intrinsics, feature map 등이 저장된다.
        #
        # args.image_size:
        #   입력 이미지 크기 [H, W]
        #
        # args.buffer:
        #   최대 저장 frame 수
        #
        # args.stereo:
        #   monocular이면 False
        self.video = DepthVideo(
            args.image_size,
            args.buffer,
            stereo=args.stereo
        )
        lever_arm = getattr(args, "imu_camera_translation", (0.0, 0.0, 0.0))
        lever_source = getattr(
            args,
            "imu_camera_translation_source",
            "zero_lever_arm_fallback",
        )
        self.video.set_imu_camera_translation(lever_arm, source=lever_source)

        ############################################################
        # 3. MotionFilter 생성
        ############################################################

        # filter incoming frames so that there is enough motion
        #
        # MotionFilter는 모든 입력 frame을 video에 저장하지 않고,
        # 충분한 움직임이 있는 frame만 keyframe 후보로 추가한다.
        #
        # args.filter_thresh:
        #   frame 추가 여부를 판단하는 motion threshold.
        self.filterx = MotionFilter(
            self.net,
            self.video,
            thresh=args.filter_thresh,
            args=args,
        )

        ############################################################
        # 4. Frontend 생성
        ############################################################

        # frontend process
        #
        # 실제로 별도 process는 아니고, main thread에서 self.frontend()로 호출된다.
        #
        # 역할:
        #   local factor graph 구성
        #   최근 window에 대한 local BA
        #   pose/disparity 업데이트
        self.frontend = DroidFrontend(
            self.net,
            self.video,
            self.args
        )
        
        ############################################################
        # 5. Backend 생성
        ############################################################

        # backend process
        #
        # Droid 기본 버전에서는 backend도 별도 process가 아니라
        # terminate() 시점에 self.backend(...)로 동기 실행된다.
        #
        # 역할:
        #   전체 또는 넓은 구간에 대해 proximity factor를 추가하고
        #   global refinement / backend BA를 수행한다.
        self.backend = DroidBackend(
            self.net,
            self.video,
            self.args
        )

        ############################################################
        # 6. Visualizer process 생성
        ############################################################

        # visualization을 끄지 않았다면 별도 process로 visualizer 실행
        if not self.disable_vis:
            from visualizer.droid_visualizer import visualization_fn

            # Droid 기본 버전은 video가 하나뿐이므로 두 번째 인자는 None
            self.visualizer = Process(
                target=visualization_fn,
                args=(self.video, None)
            )

            # visualizer process 시작
            self.visualizer.start()

        ############################################################
        # 7. Trajectory filler 생성
        ############################################################

        # post processor - fill in poses for non-keyframes
        #
        # MotionFilter가 일부 frame만 keyframe으로 넣기 때문에,
        # 원본 stream의 모든 frame에 대한 pose가 바로 존재하지는 않는다.
        #
        # PoseTrajectoryFiller는 skipped frame의 pose를 추정/보간해서
        # 최종 camera trajectory를 만든다.
        self.traj_filler = PoseTrajectoryFiller(
            self.net,
            self.video
        )


    def load_weights(self, weights):
        """
        load trained model weights

        pretrained DROID-SLAM model을 로드하는 함수.

        처리 순서:
            1. DroidNet 객체 생성
            2. checkpoint 로드
            3. DataParallel prefix인 "module." 제거
            4. update head 일부 weight shape 조정
            5. state_dict 로드
            6. cuda:0으로 이동
            7. eval mode 설정
        """

        # 현재 로드하는 weight path 출력
        print(weights)

        # DROID network 생성
        self.net = DroidNet()

        # checkpoint 로드
        #
        # 학습 시 torch.nn.DataParallel을 사용하면
        # state_dict key 앞에 "module."이 붙는 경우가 있다.
        # 단일 GPU 추론에서는 이 prefix가 있으면 key mismatch가 나므로 제거한다.
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()
        ])

        model_state = self.net.state_dict()
        incompatible = [
            key for key, value in state_dict.items()
            if key in model_state and value.shape != model_state[key].shape
        ]
        for key in incompatible:
            print(
                f"[CKPT WARNING] skip incompatible weight {key}: "
                f"checkpoint={tuple(state_dict[key].shape)}, model={tuple(model_state[key].shape)}"
            )
            state_dict.pop(key)

        # weight 로드
        self.net.load_state_dict(state_dict, strict=False)
        self.net._loaded_state_keys = set(state_dict.keys())

        # cuda:0으로 이동하고 evaluation mode 설정
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None, imu_prior=None):
        """
        main thread - update map

        demo.py에서 매 입력 frame마다 호출되는 함수.

        입력:
            tstamp:
                frame timestamp 또는 frame index

            image:
                현재 입력 이미지 tensor
                shape 예: [1, 3, H, W]

            depth:
                optional depth map.
                monocular에서는 보통 None.

            intrinsics:
                camera intrinsics [fx, fy, cx, cy]

        처리 순서:
            1. MotionFilter로 현재 frame이 충분한 motion을 갖는지 검사
            2. 조건을 만족하면 DepthVideo에 frame append
            3. DroidFrontend를 호출해 local bundle adjustment 수행
        """

        # 추론 과정이므로 gradient 계산 불필요
        with torch.no_grad():

            ########################################################
            # 1. MotionFilter
            ########################################################

            # check there is enough motion
            #
            # MotionFilter 내부에서:
            #   feature extraction
            #   이전 selected frame과 correlation 계산
            #   update network로 approximate flow delta 계산
            #   delta가 threshold보다 크면 self.video.append(...)
            was_selected = self.filterx.track(
                tstamp,
                image,
                depth,
                intrinsics,
                imu_prior=imu_prior,
            )

            ########################################################
            # 2. Frontend local BA
            ########################################################

            # local bundle adjustment
            #
            # MotionFilter가 새 frame을 추가했다면
            # frontend가 factor graph를 갱신하고 pose/disparity를 최적화한다.
            self.frontend()

            # Existing callers ignore this return value. The offline manifest
            # exporter consumes the pre-frontend MotionFilter decision.
            return bool(was_selected)

    def _has_vi_alignment(self):
        ready = getattr(self.video, "imu_alignment_ready", None)
        return bool(ready.value) if hasattr(ready, "value") else False

    @torch.no_grad()
    def _maybe_backend_visual_inertial_initialization(self):
        """Recover VI metricization after the first visual-only backend pass.

        This is deliberately an opt-in offline fallback.  It does not replace
        the online frontend initializer used by live phone capture; it lets a
        completed sequence use its globally refined visual trajectory when the
        early online window was not yet observable.
        """

        if not (
            getattr(self.args, "use_vi_initialization", False)
            and getattr(self.args, "vi_init_backend_fallback", False)
            and not self._has_vi_alignment()
        ):
            return False

        count = int(self.video.counter.value)
        if count < int(getattr(self.args, "vi_init_min_frames", 30)):
            print(
                "[VI-INIT] backend fallback skipped: "
                f"insufficient_frames, count={count}"
            )
            return False

        window = max(
            int(getattr(self.args, "vi_init_window", 45)),
            int(getattr(self.args, "vi_init_backend_window", 120)),
        )
        result = try_visual_inertial_initialization(
            self.video,
            initial_gyro_bias=getattr(self.net, "imu_gyro_bias", None),
            initial_accel_bias=getattr(self.net, "imu_acc_bias", None),
            compose_order=getattr(self.args, "imu_full_compose_order", "dq_prev"),
            inverse_imu=getattr(self.args, "imu_full_inverse", True),
            min_frames=getattr(self.args, "vi_init_min_frames", 30),
            window=window,
            max_dt=getattr(self.args, "vi_init_max_dt", 0.5),
            min_edges=getattr(self.args, "vi_init_min_edges", 20),
            min_duration=getattr(self.args, "vi_init_min_duration", 1.5),
            min_rotation_deg=getattr(self.args, "vi_init_min_rotation_deg", 5.0),
            min_visual_span_ratio=getattr(
                self.args, "vi_init_min_visual_span_ratio", 3.0
            ),
            max_rms_m=getattr(self.args, "vi_init_max_rms_m", 0.5),
            min_scale=getattr(self.args, "vi_init_min_scale", 0.05),
            max_scale=getattr(self.args, "vi_init_max_scale", 20.0),
            max_condition=getattr(self.args, "vi_init_max_condition", 1e9),
            max_accel_bias_norm=getattr(
                self.args, "vi_init_max_accel_bias_norm", 1.0
            ),
            max_gyro_bias_norm=getattr(
                self.args, "vi_init_max_gyro_bias_norm", 0.2
            ),
        )
        if not result.success:
            metrics = result.metrics or {}
            print(
                "[VI-INIT] backend fallback deferred: "
                f"{result.reason}, edges={int(metrics.get('candidate_edges', 0.0))}, "
                f"duration={metrics.get('candidate_duration_sec', 0.0):.3f}s"
            )
            return False

        result_metrics = dict(result.metrics or {})
        result_metrics["backend_fallback"] = 1.0
        result_metrics["backend_fallback_window"] = float(window)
        result.metrics = result_metrics
        if not apply_visual_inertial_alignment(self.video, result):
            print("[VI-INIT] backend fallback could not commit alignment")
            return False

        metrics = result.metrics or {}
        print(
            "[VI-INIT] backend fallback aligned "
            f"scale={result.scale:.6f}, "
            f"gravity_norm={metrics.get('gravity_norm_internal', 0.0) / max(result.scale, 1e-12):.6f}, "
            f"edges={int(metrics.get('edges', 0.0))}, "
            f"rms={metrics.get('residual_rms_internal', float('nan')) / max(result.scale, 1e-12):.6f}"
        )
        return True

    @torch.no_grad()
    def _run_final_global_bias_refinement(self):
        """Refine the common bias centre after the first global 15DoF pass.

        The main CUDA DBA already optimizes every keyframe's local 15DoF
        state. This explicit six-variable block is an offline finalization
        step, followed by a short ordinary full-state backend pass. Keeping it
        here makes the online frontend schedule unchanged and records a clear
        before/after boundary in the reconstruction diagnostics.
        """

        if not getattr(self.args, "use_global_bias_15dof_refinement", False):
            return False

        # Global and local bias states have a gauge freedom: moving the same
        # offset between them leaves every physical total bias unchanged.
        # Normalize that gauge before the shared six-variable solve instead of
        # injecting a new local-bias prior only in this final backend pass.
        precenter = recenter_full_imu_biases(
            self.video,
            max_dt=getattr(self.args, "imu_full_max_dt", 0.5),
            max_accel_norm=getattr(
                self.args, "imu_global_bias_refine_max_accel_norm", 2.0
            ),
            max_gyro_norm=getattr(
                self.args, "imu_global_bias_refine_max_gyro_norm", 0.2
            ),
        )
        record_window_bias_refinement(self.video, precenter, "global_15dof_precenter")
        if precenter.get("updated", 0.0) > 0.0:
            print(
                "[VI-BIAS-GLOBAL] precentered local 15DoF biases "
                f"|dba|={precenter.get('accel_center_shift_norm', 0.0):.6g}, "
                f"|dbg|={precenter.get('gyro_center_shift_norm', 0.0):.6g}, "
                f"total_preservation={precenter.get('total_bias_preservation_max_abs', 0.0):.3g}"
            )

        record_imu_state_diagnostics(self.video, "pre_global_bias_15dof")
        learned_confidence = None
        confidence_stats = {
            "learned_confidence_enabled": 0.0,
            "learned_confidence_cached_rows": 0.0,
            "learned_confidence_cached_mean": 1.0,
        }
        if getattr(self.args, "use_learned_imu_confidence", False):
            count = int(self.video.counter.value)
            cache = getattr(self.video, "imu_learned_confidence", None)
            coverage = getattr(self.video, "imu_learned_confidence_valid", None)
            if cache is not None and cache.shape[0] >= count:
                learned_confidence = cache[:count].detach().clone()
                if coverage is not None and coverage.numel() >= count:
                    coverage = coverage[:count].detach().to(dtype=torch.bool)
                    learned_confidence = torch.where(
                        coverage[:, None] if learned_confidence.ndim == 2 else coverage,
                        learned_confidence,
                        torch.ones_like(learned_confidence),
                    )
                    cached_rows = int(coverage.sum().detach().cpu().item())
                    cached_mean = (
                        float(learned_confidence[coverage].mean().detach().cpu().item())
                        if cached_rows > 0
                        else 1.0
                    )
                else:
                    cached_rows = count
                    cached_mean = float(learned_confidence.mean().detach().cpu().item())
                confidence_stats = {
                    "learned_confidence_enabled": 1.0,
                    "learned_confidence_cached_rows": float(cached_rows),
                    "learned_confidence_cached_mean": cached_mean,
                }
                print(
                    "[IMU-CONFIDENCE] global shared-bias uses cached direct "
                    f"temporal confidence: rows={cached_rows}/{count}, "
                    f"mean={cached_mean:.4f}"
                )
            else:
                print(
                    "[IMU-CONFIDENCE WARNING] no complete cached temporal confidence; "
                    "using neutral global shared-bias weights"
                )

        # `refine_global_bias_15dof()` predates component confidence and has
        # a scalar per-factor likelihood.  Full local DBA uses all three
        # channels; global bias-only refinement receives their neutral mean.
        if learned_confidence is not None and learned_confidence.ndim == 2:
            learned_confidence = learned_confidence.mean(dim=-1)

        result = refine_global_bias_15dof(
            self.video,
            compose_order=getattr(self.args, "imu_full_compose_order", "dq_prev"),
            inverse_imu=getattr(self.args, "imu_full_inverse", True),
            max_dt=getattr(self.args, "imu_full_max_dt", 0.5),
            max_residual_deg=getattr(self.args, "imu_ba_prior_max_deg", 45.0),
            pos_weight=getattr(self.args, "imu_full_pos_weight", 0.05),
            vel_weight=getattr(self.args, "imu_full_vel_weight", 0.05),
            damping=getattr(self.args, "imu_global_bias_refine_damping", 1e-4),
            step_scale=getattr(self.args, "imu_global_bias_refine_step_scale", 0.5),
            max_accel_step=getattr(
                self.args, "imu_global_bias_refine_max_accel_step", 0.15
            ),
            max_gyro_step=getattr(
                self.args, "imu_global_bias_refine_max_gyro_step", 0.015
            ),
            max_accel_norm=getattr(
                self.args, "imu_global_bias_refine_max_accel_norm", 2.0
            ),
            max_gyro_norm=getattr(
                self.args, "imu_global_bias_refine_max_gyro_norm", 0.2
            ),
            # This block conditions on the completed pose/velocity trajectory.
            # Its likelihood scale must therefore not inherit the small
            # visual-versus-IMU DBA trade-off weight; covariance, row quality,
            # and robustification provide the relevant relative weighting here.
            global_weight=getattr(
                self.args, "imu_global_bias_refine_weight", 1.0
            ),
            # The cache contains only direct temporal visual edge predictions.
            # Rows without such an edge are explicitly neutral rather than
            # inheriting confidence from an unrelated graph factor.
            imu_confidence=learned_confidence,
            use_imu_info_weighting=getattr(
                self.args, "use_imu_info_weighting", False
            ),
            imu_info_reference=getattr(self.args, "imu_info_reference", None),
            imu_info_weight_clip=getattr(self.args, "imu_info_weight_clip", 4.0),
            imu_info_weight_eps=getattr(self.args, "imu_info_weight_eps", 1e-12),
            robust_delta=getattr(self.args, "imu_global_bias_refine_robust_delta", 3.0),
            min_edges=getattr(self.args, "imu_global_bias_refine_min_edges", 20),
            min_duration=getattr(
                self.args, "imu_global_bias_refine_min_duration", 2.0
            ),
            min_rotation_deg=getattr(
                self.args, "imu_global_bias_refine_min_rotation_deg", 5.0
            ),
            max_condition=getattr(
                self.args, "imu_global_bias_refine_max_condition", 1e8
            ),
            min_relative_improvement=getattr(
                self.args,
                "imu_global_bias_refine_min_relative_improvement",
                1e-4,
            ),
            accel_random_walk=getattr(
                self.args, "imu_global_bias_refine_accel_random_walk", -1.0
            ),
            gyro_random_walk=getattr(
                self.args, "imu_global_bias_refine_gyro_random_walk", -1.0
            ),
            accel_prior_std=getattr(
                self.args, "imu_global_bias_refine_accel_prior_std", 0.5
            ),
            gyro_prior_std=getattr(
                self.args, "imu_global_bias_refine_gyro_prior_std", 0.05
            ),
            relinearize_accel_threshold=getattr(
                self.args, "imu_global_bias_relinearize_accel", 0.05
            ),
            relinearize_gyro_threshold=getattr(
                self.args, "imu_global_bias_relinearize_gyro", 0.005
            ),
        )
        result.update(confidence_stats)
        record_window_bias_refinement(self.video, result, "global_15dof_block")
        if result.get("updated", 0.0) <= 0.0:
            print(
                "[VI-BIAS-GLOBAL] skipped: "
                f"{result.get('reason', 'unknown')}, "
                f"edges={int(result.get('edges', 0.0))}"
            )
            record_imu_state_diagnostics(self.video, "post_global_bias_15dof_rejected")
            return False

        print(
            "[VI-BIAS-GLOBAL] accepted "
            f"cost={result.get('cost_before', 0.0):.6g}->{result.get('cost_after', 0.0):.6g}, "
            f"|dba|={result.get('accel_step_norm', 0.0):.6g}, "
            f"|dbg|={result.get('gyro_step_norm', 0.0):.6g}, "
            f"cond={result.get('condition', float('nan')):.3g}"
        )
        torch.cuda.empty_cache()
        post_steps = max(
            1, int(getattr(self.args, "imu_global_bias_refine_post_steps", 2))
        )
        print(
            "[IMU] rerunning global 15DoF DBA after shared-bias update "
            f"(steps={post_steps})"
        )
        self.backend(post_steps)
        record_imu_state_diagnostics(self.video, "post_global_bias_15dof_backend")

        recenter = recenter_full_imu_biases(
            self.video,
            max_dt=getattr(self.args, "imu_full_max_dt", 0.5),
            max_accel_norm=getattr(
                self.args, "imu_global_bias_refine_max_accel_norm", 2.0
            ),
            max_gyro_norm=getattr(
                self.args, "imu_global_bias_refine_max_gyro_norm", 0.2
            ),
        )
        record_window_bias_refinement(self.video, recenter, "global_15dof_recenter")
        if recenter.get("updated", 0.0) > 0.0:
            print(
                "[VI-BIAS-GLOBAL] recentered local 15DoF biases "
                f"|dba|={recenter.get('accel_center_shift_norm', 0.0):.6g}, "
                f"|dbg|={recenter.get('gyro_center_shift_norm', 0.0):.6g}, "
                f"total_preservation={recenter.get('total_bias_preservation_max_abs', 0.0):.3g}"
            )
        else:
            print(
                "[VI-BIAS-GLOBAL] recenter skipped: "
                f"{recenter.get('reason', 'unknown')}"
            )
        record_imu_state_diagnostics(self.video, "post_global_bias_15dof")
        return True

    def terminate(self, stream=None):
        """
        terminate the visualization process, return poses [t, q]

        모든 frame 입력이 끝난 뒤 호출되는 종료/후처리 함수.

        역할:
            1. frontend 객체 제거
            2. CUDA cache 정리
            3. backend optimization 1차 수행
            4. backend optimization 2차 수행
            5. PoseTrajectoryFiller로 skipped frame pose까지 채움
            6. 최종 camera trajectory 반환

        stream:
            원본 image stream.
            PoseTrajectoryFiller가 keyframe이 아닌 frame들의 pose를 채울 때 사용한다.

        반환:
            camera_trajectory:
                최종 카메라 trajectory.
                shape은 대략 [N, 7].
                마지막에 inv()를 적용해 camera pose convention으로 변환한 뒤 numpy로 반환한다.
        """

        ############################################################
        # 1. frontend 제거
        ############################################################

        if hasattr(self.frontend, "flush_imu_debug"):
            self.frontend.flush_imu_debug()

        # Preserve scale/bias state before either global backend pass. This
        # makes it possible to tell a bad VI initialization from later drift.
        record_imu_state_diagnostics(self.video, "post_frontend")

        # 더 이상 실시간 local tracking이 필요 없으므로 frontend 삭제
        del self.frontend

        ############################################################
        # 2. backend refinement 1차
        ############################################################

        # CUDA cache 정리
        torch.cuda.empty_cache()

        print("#" * 32)

        if getattr(self.args, "use_two_stage_imu_ba", False):
            # Local updates already ran visual pose-only followed by full
            # state DBA. The global-bias path must start from exactly the
            # same state objective as its no-global-bias control.
            if getattr(self.args, "imu_two_stage_skip_global_backend", False):
                # This is an ablation aid, not a normal VI-SLAM termination
                # mode. It saves the state produced by local pose-only ->
                # local full-15DoF updates so a later run can isolate what
                # the final global graph changes.
                print(
                    "[IMU-DIAGNOSTIC] skipping final global 15DoF backend; "
                    "saving local two-stage state"
                )
                record_imu_state_diagnostics(self.video, "post_backend_skipped")
            else:
                global_steps = max(
                    1, int(getattr(self.args, "imu_two_stage_global_steps", 12))
                )
                visual_pre_steps = max(
                    0,
                    int(
                        getattr(
                            self.args,
                            "imu_two_stage_global_visual_pre_steps",
                            0,
                        )
                    ),
                )
                if visual_pre_steps > 0:
                    # Stock DROID performs an initial global visual BA pass
                    # before its final global refinement.  The historical
                    # two-stage VI path replaced that pass with the full IMU
                    # objective, so it did not start its global solve from
                    # the same visual optimum as the baseline.  Keep the
                    # visual pass explicit, then construct a fresh full-IMU
                    # graph for the actual VI refinement below.
                    print(
                        "[IMU] running DROID-compatible visual-only global "
                        f"pre-pass (steps={visual_pre_steps}) before 15DoF BA"
                    )
                    self.backend(visual_pre_steps, visual_only=True)
                    record_imu_state_diagnostics(
                        self.video, "post_backend_visual_pre_15dof"
                    )
                print("[IMU] running one global joint 15DoF backend refinement")
                self.backend(global_steps)
                record_imu_state_diagnostics(self.video, "post_backend_final")
                refined = self._run_final_global_bias_refinement()

                # A no-global-bias run can use this to match the short post-update
                # backend pass of a shared-bias experiment without changing any
                # bias state. It is intentionally explicit for ablation reports.
                if not refined:
                    matched_steps = max(
                        0,
                        int(
                            getattr(self.args, "imu_two_stage_matched_post_steps", 0)
                        ),
                    )
                    if matched_steps > 0:
                        print(
                            "[IMU] running matched no-bias extra global 15DoF DBA "
                            f"(steps={matched_steps})"
                        )
                        self.backend(matched_steps)
                        record_imu_state_diagnostics(
                            self.video, "post_backend_matched_control"
                        )
        else:
            # backend optimization 수행
            # steps=7
            #
            # 첫 번째 backend pass로 전체 trajectory를 한 번 정리한다.
            self.backend(7)
            record_imu_state_diagnostics(self.video, "post_backend_visual")

            # A difficult sequence can lack enough early parallax for online VI
            # initialization even though the first visual global pass makes it
            # observable.  This optional fallback keeps the next backend pass a
            # real 9DoF visual-inertial refinement rather than silently visual-only.
            self._maybe_backend_visual_inertial_initialization()
            record_imu_state_diagnostics(self.video, "post_backend_vi_init")

            ############################################################
            # 3. backend refinement 2차
            ############################################################

            # 다시 CUDA cache 정리
            torch.cuda.empty_cache()

            print("#" * 32)

            # 두 번째 backend pass
            # steps=12
            #
            # 더 많은 iteration으로 추가 refinement를 수행한다.
            self.backend(12)
            record_imu_state_diagnostics(self.video, "post_backend_final")

        ############################################################
        # 4. trajectory filling
        ############################################################

        # MotionFilter 때문에 DepthVideo에는 selected frame만 들어 있다.
        # 따라서 원본 stream 기준 모든 frame의 pose를 얻기 위해
        # PoseTrajectoryFiller를 실행한다.
        camera_trajectory = self.traj_filler(stream)

        ############################################################
        # 5. 최종 trajectory 반환
        ############################################################

        # DROID 내부 pose convention을 camera trajectory convention으로 맞추기 위해 inverse 적용
        # GPU tensor -> CPU numpy 변환
        return camera_trajectory.inv().data.cpu().numpy()

    def terminate_local(self, stream=None):
        """Return a dense local VO/VIO trajectory without global BA.

        The normal :meth:`terminate` executes the termination-time global
        backend.  This method deliberately does not: it preserves the online
        frontend's sliding-window local-BA state and only invokes
        ``PoseTrajectoryFiller`` to recover poses for input timestamps that
        MotionFilter skipped.  The filler uses temporary, motion-only query
        poses and restores ``video.counter`` afterwards.

        It is therefore suitable for a DVI-style *local* VO/VIO ATE-scale
        evaluation on the full camera timeline.  It must not be reported as a
        global SLAM result.
        """

        if stream is None:
            raise ValueError("terminate_local requires an image stream for pose filling")

        if hasattr(self.frontend, "flush_imu_debug"):
            self.frontend.flush_imu_debug()

        record_imu_state_diagnostics(self.video, "post_local_frontend_before_fill")
        local_keyframes = int(self.video.counter.value)
        if local_keyframes < 5:
            raise RuntimeError(
                "local tracking produced fewer than five selected keyframes; "
                "cannot construct a local trajectory"
            )

        print(
            "[TRAJECTORY] local VIO export: "
            f"keyframes={local_keyframes}, global_backend=skipped, "
            "pose_filler=enabled"
        )
        torch.cuda.empty_cache()
        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()

from collections import deque

import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph
from imu_residual import build_imu_rotation_residual_from_args
from vi_initialization import (
    apply_visual_inertial_alignment,
    capture_vi_initialization_snapshot,
    record_window_bias_refinement,
    rebase_vi_alignment_result,
    try_visual_inertial_initialization,
    vi_initialization_observability,
)

from cuda_timer import CudaTimer


ENABLE_TIMING = False

class DroidFrontend:
    def __init__(self, net, video, args):
        self.video = video
        self.net = net
        self.args = args
        self.update_op = net.update
        self.imu_regularizer = build_imu_rotation_residual_from_args(
            args,
            stage="frontend",
        )
        self.graph = FactorGraph(
            video,
            net.update,
            max_factors=48,
            upsample=args.upsample,
            imu_regularizer=self.imu_regularizer,
            apply_imu_residual=getattr(args, "use_imu_residual", False),
            use_learned_imu_confidence=getattr(args, "use_learned_imu_confidence", False),
            imu_confidence_floor=getattr(args, "imu_confidence_floor", 0.0),
            use_imu_ba_prior=getattr(args, "use_imu_ba_prior", False),
            imu_ba_prior_weight=getattr(args, "imu_ba_prior_weight", 0.0),
            imu_ba_prior_max_deg=getattr(args, "imu_ba_prior_max_deg", 45.0),
            use_full_imu_ba=(
                getattr(args, "use_full_imu_ba", False)
                and getattr(args, "imu_full_frontend", False)
            ),
            require_imu_alignment=getattr(args, "use_vi_initialization", False),
            imu_full_pos_weight=getattr(args, "imu_full_pos_weight", 0.05),
            imu_full_vel_weight=getattr(args, "imu_full_vel_weight", 0.05),
            imu_full_bias_weight=getattr(args, "imu_full_bias_weight", 0.001),
            imu_motion_prior_weight=getattr(args, "imu_motion_prior_weight", 0.0),
            imu_local_bias_prior_weight=getattr(args, "imu_local_bias_prior_weight", 0.0),
            imu_gravity=getattr(args, "imu_gravity", None),
            imu_full_compose_order=getattr(args, "imu_full_compose_order", "dq_prev"),
            imu_full_inverse=getattr(args, "imu_full_inverse", True),
            imu_full_max_dt=getattr(args, "imu_full_max_dt", 0.5),
            imu_full_max_dv=getattr(args, "imu_full_max_dv", 5.0),
            imu_full_max_dp=getattr(args, "imu_full_max_dp", 1.0),
            use_imu_info_weighting=getattr(args, "use_imu_info_weighting", False),
            imu_info_weight_clip=getattr(args, "imu_info_weight_clip", 4.0),
            imu_info_weight_eps=getattr(args, "imu_info_weight_eps", 1e-12),
            imu_info_reference=getattr(args, "imu_info_reference", None),
            use_adaptive_imu_gate=getattr(args, "use_adaptive_imu_gate", False),
            imu_gate_min=getattr(args, "imu_gate_min", 0.2),
            imu_gate_max=getattr(args, "imu_gate_max", 1.25),
            imu_gate_visual_thresh=getattr(args, "imu_gate_visual_thresh", 0.2),
            imu_gate_visual_boost=getattr(args, "imu_gate_visual_boost", 0.25),
            imu_gate_motion_thresh_deg=getattr(args, "imu_gate_motion_thresh_deg", 5.0),
            imu_gate_motion_boost=getattr(args, "imu_gate_motion_boost", 0.15),
            imu_gate_residual_soft_deg=getattr(args, "imu_gate_residual_soft_deg", 2.5),
            imu_gate_residual_hard_deg=getattr(args, "imu_gate_residual_hard_deg", 8.0),
            imu_gate_residual_pos_soft=getattr(args, "imu_gate_residual_pos_soft", 0.0),
            imu_gate_residual_vel_soft=getattr(args, "imu_gate_residual_vel_soft", 0.0),
            imu_optimize_bias=not getattr(args, "use_pose_velocity_imu_ba", False),
            imu_gyro_bias=getattr(net, "imu_gyro_bias", None),
            imu_acc_bias=getattr(net, "imu_acc_bias", None),
            use_joint_shared_bias_ba=getattr(args, "use_joint_shared_bias_ba", False),
            imu_joint_bias_interval=getattr(args, "imu_joint_bias_interval", 10),
            imu_joint_bias_window=getattr(args, "imu_joint_bias_window", 30),
            imu_joint_bias_damping=getattr(args, "imu_joint_bias_damping", 1e-4),
            imu_joint_bias_step_scale=getattr(args, "imu_joint_bias_step_scale", 1.0),
            imu_joint_bias_max_accel_step=getattr(args, "imu_joint_bias_max_accel_step", 0.15),
            imu_joint_bias_max_gyro_step=getattr(args, "imu_joint_bias_max_gyro_step", 0.015),
            imu_joint_bias_max_accel_norm=getattr(args, "imu_joint_bias_max_accel_norm", 0.5),
            imu_joint_bias_max_gyro_norm=getattr(args, "imu_joint_bias_max_gyro_norm", 0.1),
            imu_joint_bias_min_edges=getattr(args, "imu_joint_bias_min_edges", 8),
            imu_joint_bias_min_duration=getattr(args, "imu_joint_bias_min_duration", 0.5),
            imu_joint_bias_min_rotation_deg=getattr(args, "imu_joint_bias_min_rotation_deg", 2.0),
            imu_joint_bias_max_condition=getattr(args, "imu_joint_bias_max_condition", 1e8),
            imu_joint_bias_accel_random_walk=getattr(args, "imu_joint_bias_accel_random_walk", -1.0),
            imu_joint_bias_gyro_random_walk=getattr(args, "imu_joint_bias_gyro_random_walk", -1.0),
            imu_joint_bias_accel_prior_std=getattr(args, "imu_joint_bias_accel_prior_std", 0.05),
            imu_joint_bias_gyro_prior_std=getattr(args, "imu_joint_bias_gyro_prior_std", 0.005),
            imu_joint_bias_min_improvement=getattr(args, "imu_joint_bias_min_improvement", 1e-10),
            imu_joint_bias_relinearize_accel=getattr(args, "imu_joint_bias_relinearize_accel", 0.05),
            imu_joint_bias_relinearize_gyro=getattr(args, "imu_joint_bias_relinearize_gyro", 0.005),
            imu_ba_outer_relinearize=getattr(args, "imu_ba_outer_relinearize", False),
            imu_ba_outer_relinearize_interval=getattr(args, "imu_ba_outer_relinearize_interval", 10),
            imu_ba_outer_window=getattr(args, "imu_ba_outer_window", 0),
            imu_ba_debug=getattr(args, "imu_ba_debug", False),
            imu_ba_debug_path=getattr(args, "imu_ba_debug_path", None),
            imu_ba_debug_max_rows=getattr(args, "imu_ba_debug_max_rows", 20000),
            imu_ba_debug_stage="frontend",
        )

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 20
        self.iters1 = 3
        self.iters2 = 2

        self.keyframe_removal_index = 3

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

        self.depth_window = 3

        self.motion_damping = 0.0
        if hasattr(args, "motion_damping"):
            self.motion_damping = args.motion_damping

        self.use_vi_initialization = bool(
            getattr(args, "use_vi_initialization", False)
        )
        self.vi_init_bootstrap = bool(
            getattr(args, "vi_init_bootstrap", True)
        )
        self.vi_init_bootstrap_frames = max(
            0, int(getattr(args, "vi_init_bootstrap_frames", 60))
        )
        self._reported_vi_init_keyframe_retention = False
        self.vi_init_retry_interval = max(
            1, int(getattr(args, "vi_init_retry_interval", 5))
        )
        self._last_vi_init_attempt = -self.vi_init_retry_interval
        self._last_vi_init_reason = None
        self.vi_init_snapshot_history_size = max(
            0, int(getattr(args, "vi_init_snapshot_history_size", 24))
        )
        self.vi_init_snapshot_stride = max(
            1, int(getattr(args, "vi_init_snapshot_stride", 5))
        )
        self.vi_init_snapshot_max_candidates = max(
            0, int(getattr(args, "vi_init_snapshot_max_candidates", 6))
        )
        self.vi_init_snapshots = deque(maxlen=self.vi_init_snapshot_history_size)
        self._reported_vi_init_snapshot_history = False
        self.use_window_bias_refinement = bool(
            not getattr(args, "use_joint_shared_bias_ba", False)
            and (
                getattr(args, "use_low_rate_bias_refinement", False)
                or getattr(args, "use_shared_bias_local_ba", False)
                or getattr(args, "use_window_bias_refinement", False)
            )
        )
        self._last_bias_refinement_frame = 0

    def _has_vi_alignment(self):
        ready = getattr(self.video, "imu_alignment_ready", None)
        return bool(ready.value) if hasattr(ready, "value") else False

    def _retain_startup_keyframes_for_vi(self):
        """Keep a bounded visual window intact until VI alignment is observable."""

        return bool(
            self.use_vi_initialization
            and self.vi_init_bootstrap
            and not self._has_vi_alignment()
            and int(self.video.counter.value) < self.vi_init_bootstrap_frames
        )

    def _vi_initialization_kwargs(self):
        """Keep live and retained-window alignment calls numerically identical."""

        return {
            "initial_gyro_bias": getattr(self.net, "imu_gyro_bias", None),
            "initial_accel_bias": getattr(self.net, "imu_acc_bias", None),
            "compose_order": getattr(self.args, "imu_full_compose_order", "dq_prev"),
            "inverse_imu": getattr(self.args, "imu_full_inverse", True),
            "min_frames": getattr(self.args, "vi_init_min_frames", 20),
            "window": getattr(self.args, "vi_init_window", 30),
            "max_dt": getattr(self.args, "vi_init_max_dt", 0.5),
            "min_edges": getattr(self.args, "vi_init_min_edges", 20),
            "min_duration": getattr(self.args, "vi_init_min_duration", 1.5),
            "min_rotation_deg": getattr(self.args, "vi_init_min_rotation_deg", 5.0),
            "min_visual_span_ratio": getattr(
                self.args, "vi_init_min_visual_span_ratio", 3.0
            ),
            "max_rms_m": getattr(self.args, "vi_init_max_rms_m", 0.5),
            "min_scale": getattr(self.args, "vi_init_min_scale", 0.05),
            "max_scale": getattr(self.args, "vi_init_max_scale", 20.0),
            "max_condition": getattr(self.args, "vi_init_max_condition", 1e9),
            "max_accel_bias_norm": getattr(
                self.args, "vi_init_max_accel_bias_norm", 1.0
            ),
            "max_gyro_bias_norm": getattr(
                self.args, "vi_init_max_gyro_bias_norm", 0.2
            ),
        }

    def _record_vi_initialization_snapshot(self):
        """Retain a compact pre-removal window for later VI alignment retries."""

        if (
            not self.use_vi_initialization
            or self._has_vi_alignment()
            or self.vi_init_snapshot_history_size <= 0
            or self.count % self.vi_init_snapshot_stride != 0
        ):
            return

        window = max(
            int(getattr(self.args, "vi_init_window", 30)),
            int(getattr(self.args, "vi_init_min_frames", 20)),
        )
        snapshot = capture_vi_initialization_snapshot(self.video, window)
        if snapshot is None:
            return

        if self.vi_init_snapshots:
            previous = self.vi_init_snapshots[-1]
            if (
                previous.tstamp.numel() > 0
                and snapshot.tstamp.numel() > 0
                and float(previous.tstamp[-1].detach().cpu().item())
                == float(snapshot.tstamp[-1].detach().cpu().item())
            ):
                self.vi_init_snapshots[-1] = snapshot
                return

        self.vi_init_snapshots.append(snapshot)
        if not self._reported_vi_init_snapshot_history:
            print(
                "[VI-INIT] retaining pre-removal alignment snapshots: "
                f"window={window}, stride={self.vi_init_snapshot_stride}, "
                f"capacity={self.vi_init_snapshot_history_size}"
            )
            self._reported_vi_init_snapshot_history = True

    @staticmethod
    def _snapshot_terminal_timestamp(snapshot):
        if snapshot.tstamp.numel() == 0:
            return None
        return int(round(float(snapshot.tstamp[-1].detach().cpu().item())))

    def _try_visual_inertial_candidates(self):
        """Try live state first, then retained windows that pass pre-screening."""

        kwargs = self._vi_initialization_kwargs()
        observability_kwargs = {
            key: kwargs[key]
            for key in (
                "min_frames",
                "window",
                "max_dt",
                "min_edges",
                "min_duration",
                "min_rotation_deg",
                "min_visual_span_ratio",
            )
        }
        live_metrics = vi_initialization_observability(
            self.video, **observability_kwargs
        )
        last_result = None
        live_terminal = None
        count = int(self.video.counter.value)
        if count > 0:
            live_terminal = int(
                round(float(self.video.tstamp[count - 1].detach().cpu().item()))
            )

        # Preserve the conventional path when the newest local window is
        # observable. History is only a fallback after that solve is rejected.
        if live_metrics.get("eligible", 0.0) > 0.0:
            last_result = try_visual_inertial_initialization(self.video, **kwargs)
            if last_result.success:
                return last_result

        candidates = []
        for age, snapshot in enumerate(reversed(self.vi_init_snapshots), start=1):
            if self._snapshot_terminal_timestamp(snapshot) == live_terminal:
                continue
            metrics = vi_initialization_observability(
                snapshot, **observability_kwargs
            )
            if metrics.get("eligible", 0.0) <= 0.0:
                continue
            candidates.append((float(metrics.get("score", 0.0)), -age, snapshot, metrics))

        candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
        for score, negative_age, snapshot, metrics in candidates[
            : self.vi_init_snapshot_max_candidates
        ]:
            result = try_visual_inertial_initialization(snapshot, **kwargs)
            if not result.success:
                if last_result is None:
                    last_result = result
                continue

            result = rebase_vi_alignment_result(result, snapshot, self.video)
            result_metrics = dict(result.metrics or {})
            result_metrics.update(
                {
                    "history_candidate_age": float(-negative_age),
                    "history_candidate_score": score,
                    "history_candidate_edges": float(metrics.get("candidate_edges", 0.0)),
                }
            )
            result.metrics = result_metrics
            return result

        # Retain the existing diagnostic reason if neither current nor history
        # had an observable candidate.
        if last_result is not None:
            return last_result
        return try_visual_inertial_initialization(self.video, **kwargs)

    def _maybe_initialize_visual_inertial(self):
        if not self.use_vi_initialization or self._has_vi_alignment():
            return False
        if self.count - self._last_vi_init_attempt < self.vi_init_retry_interval:
            return False

        self._last_vi_init_attempt = self.count
        result = self._try_visual_inertial_candidates()
        if not result.success:
            metrics = result.metrics or {}
            if result.reason != self._last_vi_init_reason:
                print(
                    "[VI-INIT] alignment deferred: "
                    f"{result.reason}, edges={int(metrics.get('candidate_edges', 0.0))}, "
                    f"duration={metrics.get('candidate_duration_sec', 0.0):.3f}s, "
                    f"rotation={metrics.get('imu_rotation_excitation_deg', 0.0):.2f}deg, "
                    f"span_ratio={metrics.get('visual_span_ratio', 0.0):.2f}"
                )
                self._last_vi_init_reason = result.reason
            return False

        if not apply_visual_inertial_alignment(self.video, result):
            print("[VI-INIT] alignment result could not be committed")
            return False

        metrics = result.metrics or {}
        self._last_vi_init_reason = None
        print(
            "[VI-INIT] aligned "
            f"scale={result.scale:.6f}, "
            f"gravity_norm={metrics.get('gravity_norm_internal', 0.0) / max(result.scale, 1e-12):.6f}, "
            f"edges={int(metrics.get('edges', 0.0))}, "
            f"rms={metrics.get('residual_rms_internal', float('nan')) / max(result.scale, 1e-12):.6f}, "
            f"source={'history' if metrics.get('history_snapshot', 0.0) > 0.0 else 'live'}"
        )
        return True

    def _maybe_refine_window_biases(self):
        if not self.use_window_bias_refinement or not self._has_vi_alignment():
            return False
        interval = max(1, int(getattr(self.args, "imu_bias_refine_interval", 20)))
        if self.count % interval != 0:
            return False

        count = int(self.video.counter.value)
        start = max(1, min(self._last_bias_refinement_frame, count - 1))
        prior_elapsed_sec = float(
            self.video.imu_delta[start:count, 0].clamp_min(0.0).sum().item()
        )

        result = self.graph.refine_shared_bias_block(
            window=getattr(self.args, "imu_bias_refine_window", 30),
            max_dt=getattr(self.args, "imu_full_max_dt", 0.5),
            damping=getattr(self.args, "imu_bias_refine_damping", 1e-4),
            step_scale=getattr(self.args, "imu_bias_refine_step_scale", 0.5),
            max_accel_step=getattr(self.args, "imu_bias_refine_max_accel_step", 0.15),
            max_gyro_step=getattr(self.args, "imu_bias_refine_max_gyro_step", 0.015),
            max_accel_norm=getattr(self.args, "imu_bias_refine_max_accel_norm", 0.5),
            max_gyro_norm=getattr(self.args, "imu_bias_refine_max_gyro_norm", 0.1),
            robust_delta=getattr(self.args, "imu_bias_refine_robust_delta", 3.0),
            min_edges=getattr(self.args, "imu_bias_refine_min_edges", 8),
            min_duration=getattr(self.args, "imu_bias_refine_min_duration", 0.5),
            min_rotation_deg=getattr(self.args, "imu_bias_refine_min_rotation_deg", 2.0),
            max_condition=getattr(self.args, "imu_bias_refine_max_condition", 1e8),
            min_relative_improvement=getattr(
                self.args, "imu_bias_refine_min_relative_improvement", 1e-7
            ),
            prior_elapsed_sec=prior_elapsed_sec,
            accel_random_walk=getattr(self.args, "imu_bias_refine_accel_random_walk", -1.0),
            gyro_random_walk=getattr(self.args, "imu_bias_refine_gyro_random_walk", -1.0),
            accel_prior_std=getattr(self.args, "imu_bias_refine_accel_prior_std", 0.05),
            gyro_prior_std=getattr(self.args, "imu_bias_refine_gyro_prior_std", 0.005),
        )
        record_window_bias_refinement(self.video, result, "frontend")
        if result.get("updated", 0.0) > 0.0:
            self._last_bias_refinement_frame = count
            print(
                "[VI-BIAS] shared block accepted "
                f"edges={int(result.get('edges', 0.0))}, "
                f"cost={result.get('cost_before', 0.0):.6g}->{result.get('cost_after', 0.0):.6g}, "
                f"|dba|={result.get('accel_step_norm', 0.0):.6g}, "
                f"|dbg|={result.get('gyro_step_norm', 0.0):.6g}, "
                f"cond={result.get('condition', float('nan')):.3g}"
            )
            return True
        return False

    def _init_next_state(self):
        # set pose / depth for next iteration
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1]

        self.video.disps[self.t1] = torch.quantile(
            self.video.disps[self.t1 - 3 : self.t1 - 1], 0.5
        )

        # damped linear velocity model
        if self.motion_damping >= 0:
            poses = SE3(self.video.poses)
            vel = (poses[self.t1 - 1] * poses[self.t1 - 2].inv()).log()
            damped_vel = self.motion_damping * vel
            next_pose = SE3.exp(damped_vel) * poses[self.t1 - 1]
            self.video.poses[self.t1] = next_pose.data

    def _update(self):
        """add edges, perform update"""

        self.count += 1
        self.t1 += 1

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(
            self.t1 - 5,
            max(self.t1 - self.frontend_window, 0),
            rad=self.frontend_radius,
            nms=self.frontend_nms,
            thresh=self.frontend_thresh,
            beta=self.beta,
            remove=True,
        )

        self.video.disps[self.t1 - 1] = torch.where(
            self.video.disps_sens[self.t1 - 1] > 0,
            self.video.disps_sens[self.t1 - 1],
            self.video.disps[self.t1 - 1],
        )

        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True)

        # Capture the visual/IMU state before a redundancy decision can remove
        # one of its short temporal edges.
        self._record_vi_initialization_snapshot()

        # A small, bounded startup set is required to make VI scale/gravity
        # observable.  MotionFilter retains input frames, and this matching
        # guard prevents the regular frontend redundancy test from immediately
        # deleting them again.
        if self._retain_startup_keyframes_for_vi():
            removed_keyframe = False
            if not self._reported_vi_init_keyframe_retention:
                print(
                    "[VI-INIT] temporarily disabling frontend keyframe removal "
                    f"until {self.vi_init_bootstrap_frames} startup keyframes"
                )
                self._reported_vi_init_keyframe_retention = True
        else:
            d = self.video.distance(
                [self.t1 - 4], [self.t1 - 2], beta=self.beta, bidirectional=True
            )
            removed_keyframe = d.item() < 2 * self.keyframe_thresh
        if removed_keyframe:
            self.graph.rm_keyframe(self.t1 - 3)

            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        self._maybe_initialize_visual_inertial()
        bias_updated = self._maybe_refine_window_biases()

        # A newly accepted shared-bias step must be followed by at least one
        # regular 9DoF CUDA DBA pass so T, v and dense depth react to the new
        # preintegration correction in the same local update.
        post_bias_iterations = self.iters2 if not removed_keyframe else (1 if bias_updated else 0)
        if post_bias_iterations > 0:
            for itr in range(post_bias_iterations):
                self.graph.update(None, None, use_inactive=True)


        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1]
        self.video.disps[self.t1] = torch.quantile(
            self.video.disps[self.t1 - self.depth_window - 1 : self.t1 - 1], 0.7
        )

        # update visualization
        self.video.dirty[self.graph.ii.min() : self.t1] = True

    def _initialize(self):
        """initialize the SLAM system"""

        self.t0 = 0
        self.t1 = self.video.counter.value

        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        self.graph.add_proximity_factors(
            0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False
        )

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1 - 4 : self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1 - 1].clone()
        self.last_disp = self.video.disps[self.t1 - 1].clone()
        self.last_time = self.video.tstamp[self.t1 - 1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[: self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup - 4, store=True)

    def flush_imu_debug(self):
        if hasattr(self, "graph"):
            self.graph.flush_imu_debug()

    def __call__(self):
        """main update"""

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self._initialize()
            self._init_next_state()

        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self._update()
            self._init_next_state()

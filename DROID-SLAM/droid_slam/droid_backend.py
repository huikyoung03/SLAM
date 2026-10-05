import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph
from imu_residual import build_imu_rotation_residual_from_args
from vi_initialization import record_window_bias_refinement


def _maybe_refine_runtime_bias(graph, args, stage):
    """Optionally refine bias after a termination-time backend pass.

    Online low-rate refinement intentionally lives in ``DroidFrontend``. A
    full backend pass spans a much larger graph and can change scale/trajectory
    behavior, so it is opt-in for an offline diagnostic only.
    """

    if not getattr(args, "imu_bias_refine_backend", False):
        return False
    if not (
        getattr(args, "use_low_rate_bias_refinement", False)
        or getattr(args, "use_window_bias_refinement", False)
        or getattr(args, "use_shared_bias_local_ba", False)
    ) or getattr(args, "use_joint_shared_bias_ba", False):
        return False
    video = graph.video
    ready = getattr(video, "imu_alignment_ready", None)
    if not hasattr(ready, "value") or not bool(ready.value):
        return False

    result = graph.refine_shared_bias_block(
        window=getattr(args, "imu_bias_refine_window", 30),
        max_dt=getattr(args, "imu_full_max_dt", 0.5),
        damping=getattr(args, "imu_bias_refine_damping", 1e-4),
        step_scale=getattr(args, "imu_bias_refine_step_scale", 0.5),
        max_accel_step=getattr(args, "imu_bias_refine_max_accel_step", 0.15),
        max_gyro_step=getattr(args, "imu_bias_refine_max_gyro_step", 0.015),
        max_accel_norm=getattr(args, "imu_bias_refine_max_accel_norm", 0.5),
        max_gyro_norm=getattr(args, "imu_bias_refine_max_gyro_norm", 0.1),
        robust_delta=getattr(args, "imu_bias_refine_robust_delta", 3.0),
        min_edges=getattr(args, "imu_bias_refine_min_edges", 8),
        min_duration=getattr(args, "imu_bias_refine_min_duration", 0.5),
        min_rotation_deg=getattr(args, "imu_bias_refine_min_rotation_deg", 2.0),
        max_condition=getattr(args, "imu_bias_refine_max_condition", 1e8),
        min_relative_improvement=getattr(
            args, "imu_bias_refine_min_relative_improvement", 1e-7
        ),
        accel_random_walk=getattr(args, "imu_bias_refine_accel_random_walk", -1.0),
        gyro_random_walk=getattr(args, "imu_bias_refine_gyro_random_walk", -1.0),
        accel_prior_std=getattr(args, "imu_bias_refine_accel_prior_std", 0.05),
        gyro_prior_std=getattr(args, "imu_bias_refine_gyro_prior_std", 0.005),
    )
    record_window_bias_refinement(video, result, stage)
    if result.get("updated", 0.0) > 0.0:
        print(
            f"[VI-BIAS] {stage} shared block accepted "
            f"cost={result.get('cost_before', 0.0):.6g}->{result.get('cost_after', 0.0):.6g}, "
            f"cond={result.get('condition', float('nan')):.3g}"
        )
        return True
    return False


class DroidBackend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update
        self.net = net
        self.args = args

        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.upsample = args.upsample
        self.beta = args.beta
        self.backend_thresh = args.backend_thresh
        self.backend_radius = args.backend_radius
        self.backend_nms = args.backend_nms
        self.imu_regularizer = build_imu_rotation_residual_from_args(
            args,
            stage="backend",
        )

    @torch.no_grad()
    def __call__(self, steps=12, normalize=True):
        """ main update """

        t = self.video.counter.value
        if normalize:
            if not self.video.stereo and not torch.any(self.video.disps_sens):
                self.video.normalize()

        graph = FactorGraph(
            self.video,
            self.update_op,
            corr_impl="alt",
            max_factors=16 * t,
            upsample=self.upsample,
            imu_regularizer=self.imu_regularizer,
            apply_imu_residual=getattr(self.args, "use_imu_residual", False),
            use_learned_imu_confidence=getattr(self.args, "use_learned_imu_confidence", False),
            imu_confidence_floor=getattr(self.args, "imu_confidence_floor", 0.0),
            use_imu_ba_prior=getattr(self.args, "use_imu_ba_prior", False),
            imu_ba_prior_weight=getattr(self.args, "imu_ba_prior_weight", 0.0),
            imu_ba_prior_max_deg=getattr(self.args, "imu_ba_prior_max_deg", 45.0),
            use_full_imu_ba=getattr(self.args, "use_full_imu_ba", False),
            require_imu_alignment=getattr(self.args, "use_vi_initialization", False),
            imu_full_pos_weight=getattr(self.args, "imu_full_pos_weight", 0.05),
            imu_full_vel_weight=getattr(self.args, "imu_full_vel_weight", 0.05),
            imu_full_bias_weight=getattr(self.args, "imu_full_bias_weight", 0.001),
            imu_motion_prior_weight=getattr(self.args, "imu_motion_prior_weight", 0.0),
            imu_local_bias_prior_weight=getattr(self.args, "imu_local_bias_prior_weight", 0.0),
            imu_gravity=getattr(self.args, "imu_gravity", None),
            imu_full_compose_order=getattr(self.args, "imu_full_compose_order", "dq_prev"),
            imu_full_inverse=getattr(self.args, "imu_full_inverse", True),
            imu_full_max_dt=getattr(self.args, "imu_full_max_dt", 0.5),
            imu_full_max_dv=getattr(self.args, "imu_full_max_dv", 5.0),
            imu_full_max_dp=getattr(self.args, "imu_full_max_dp", 1.0),
            use_imu_info_weighting=getattr(self.args, "use_imu_info_weighting", False),
            imu_info_weight_clip=getattr(self.args, "imu_info_weight_clip", 4.0),
            imu_info_weight_eps=getattr(self.args, "imu_info_weight_eps", 1e-12),
            imu_info_reference=getattr(self.args, "imu_info_reference", None),
            use_adaptive_imu_gate=getattr(self.args, "use_adaptive_imu_gate", False),
            imu_gate_min=getattr(self.args, "imu_gate_min", 0.2),
            imu_gate_max=getattr(self.args, "imu_gate_max", 1.25),
            imu_gate_visual_thresh=getattr(self.args, "imu_gate_visual_thresh", 0.2),
            imu_gate_visual_boost=getattr(self.args, "imu_gate_visual_boost", 0.25),
            imu_gate_motion_thresh_deg=getattr(self.args, "imu_gate_motion_thresh_deg", 5.0),
            imu_gate_motion_boost=getattr(self.args, "imu_gate_motion_boost", 0.15),
            imu_gate_residual_soft_deg=getattr(self.args, "imu_gate_residual_soft_deg", 2.5),
            imu_gate_residual_hard_deg=getattr(self.args, "imu_gate_residual_hard_deg", 8.0),
            imu_gate_residual_pos_soft=getattr(self.args, "imu_gate_residual_pos_soft", 0.0),
            imu_gate_residual_vel_soft=getattr(self.args, "imu_gate_residual_vel_soft", 0.0),
            imu_optimize_bias=not getattr(self.args, "use_pose_velocity_imu_ba", False),
            imu_gyro_bias=getattr(self.net, "imu_gyro_bias", None),
            imu_acc_bias=getattr(self.net, "imu_acc_bias", None),
            use_joint_shared_bias_ba=getattr(self.args, "use_joint_shared_bias_ba", False),
            imu_joint_bias_interval=getattr(self.args, "imu_joint_bias_interval", 10),
            imu_joint_bias_window=getattr(self.args, "imu_joint_bias_window", 30),
            imu_joint_bias_damping=getattr(self.args, "imu_joint_bias_damping", 1e-4),
            imu_joint_bias_step_scale=getattr(self.args, "imu_joint_bias_step_scale", 1.0),
            imu_joint_bias_max_accel_step=getattr(self.args, "imu_joint_bias_max_accel_step", 0.15),
            imu_joint_bias_max_gyro_step=getattr(self.args, "imu_joint_bias_max_gyro_step", 0.015),
            imu_joint_bias_max_accel_norm=getattr(self.args, "imu_joint_bias_max_accel_norm", 0.5),
            imu_joint_bias_max_gyro_norm=getattr(self.args, "imu_joint_bias_max_gyro_norm", 0.1),
            imu_joint_bias_min_edges=getattr(self.args, "imu_joint_bias_min_edges", 8),
            imu_joint_bias_min_duration=getattr(self.args, "imu_joint_bias_min_duration", 0.5),
            imu_joint_bias_min_rotation_deg=getattr(self.args, "imu_joint_bias_min_rotation_deg", 2.0),
            imu_joint_bias_max_condition=getattr(self.args, "imu_joint_bias_max_condition", 1e8),
            imu_joint_bias_accel_random_walk=getattr(self.args, "imu_joint_bias_accel_random_walk", -1.0),
            imu_joint_bias_gyro_random_walk=getattr(self.args, "imu_joint_bias_gyro_random_walk", -1.0),
            imu_joint_bias_accel_prior_std=getattr(self.args, "imu_joint_bias_accel_prior_std", 0.05),
            imu_joint_bias_gyro_prior_std=getattr(self.args, "imu_joint_bias_gyro_prior_std", 0.005),
            imu_joint_bias_min_improvement=getattr(self.args, "imu_joint_bias_min_improvement", 1e-10),
            imu_joint_bias_relinearize_accel=getattr(self.args, "imu_joint_bias_relinearize_accel", 0.05),
            imu_joint_bias_relinearize_gyro=getattr(self.args, "imu_joint_bias_relinearize_gyro", 0.005),
            imu_ba_outer_relinearize=getattr(self.args, "imu_ba_outer_relinearize", False),
            imu_ba_outer_relinearize_interval=getattr(self.args, "imu_ba_outer_relinearize_interval", 10),
            imu_ba_outer_window=getattr(self.args, "imu_ba_outer_window", 0),
            imu_ba_debug=getattr(self.args, "imu_ba_debug", False),
            imu_ba_debug_path=getattr(self.args, "imu_ba_debug_path", None),
            imu_ba_debug_max_rows=getattr(self.args, "imu_ba_debug_max_rows", 20000),
            imu_ba_debug_stage="backend",
        )

        graph.add_proximity_factors(rad=self.backend_radius,
                                    nms=self.backend_nms,
                                    thresh=self.backend_thresh,
                                    beta=self.beta)

        graph.update_lowmem(steps=steps)
        if _maybe_refine_runtime_bias(graph, self.args, "backend"):
            # Re-linearize the local 9DoF CUDA problem around the accepted
            # shared bias before this termination-time graph is discarded.
            graph.update_lowmem(steps=1)
        graph.clear_edges()
        graph.flush_imu_debug()
        self.video.dirty[:t] = True


class DroidAsyncBackend:
    def __init__(self, net, video, args, max_age = 7):
        self.video = video
        self.update_op = net.update
        self.net = net
        self.args = args
        self.max_age = max_age

        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.upsample = args.upsample
        self.beta = args.beta
        self.backend_thresh = args.backend_thresh
        self.backend_radius = args.backend_radius
        self.backend_nms = args.backend_nms
        self.imu_regularizer = build_imu_rotation_residual_from_args(
            args,
            stage="async_backend",
        )

        self.graph = FactorGraph(
            self.video,
            self.update_op,
            corr_impl="alt",
            max_factors=-1,
            upsample=self.upsample,
            imu_regularizer=self.imu_regularizer,
            apply_imu_residual=getattr(args, "use_imu_residual", False),
            use_learned_imu_confidence=getattr(args, "use_learned_imu_confidence", False),
            imu_confidence_floor=getattr(args, "imu_confidence_floor", 0.0),
            use_imu_ba_prior=getattr(args, "use_imu_ba_prior", False),
            imu_ba_prior_weight=getattr(args, "imu_ba_prior_weight", 0.0),
            imu_ba_prior_max_deg=getattr(args, "imu_ba_prior_max_deg", 45.0),
            use_full_imu_ba=getattr(args, "use_full_imu_ba", False),
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
            imu_ba_debug_stage="async_backend",
        )

    @torch.no_grad()
    def __call__(self, steps=12, normalize=True):
        """main update"""

        t = self.video.counter.value
        if normalize:
            if not self.video.stereo and not torch.any(self.video.disps_sens):
                self.video.normalize()

        self.graph.add_proximity_factors(
            rad=self.backend_radius,
            nms=self.backend_nms,
            thresh=self.backend_thresh,
            beta=self.beta,
        )

        self.graph.update_lowmem(steps=steps, use_inactive=True)
        if _maybe_refine_runtime_bias(self.graph, self.args, "async-backend"):
            self.graph.update_lowmem(steps=1, use_inactive=True)
        self.graph.rm_factors(self.graph.age > self.max_age, store=True)
        self.graph.flush_imu_debug()

        self.video.dirty[:t] = True

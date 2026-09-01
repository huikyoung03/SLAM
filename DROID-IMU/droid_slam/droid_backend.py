import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph
from imu_residual import build_imu_rotation_residual_from_args


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
            imu_full_pos_weight=getattr(self.args, "imu_full_pos_weight", 0.05),
            imu_full_vel_weight=getattr(self.args, "imu_full_vel_weight", 0.05),
            imu_full_bias_weight=getattr(self.args, "imu_full_bias_weight", 0.001),
            imu_motion_prior_weight=getattr(self.args, "imu_motion_prior_weight", 0.0),
            imu_local_bias_prior_weight=getattr(self.args, "imu_local_bias_prior_weight", 0.0),
            imu_gravity=getattr(self.args, "imu_gravity", None),
            imu_full_max_dt=getattr(self.args, "imu_full_max_dt", 0.5),
            imu_full_max_dv=getattr(self.args, "imu_full_max_dv", 5.0),
            imu_full_max_dp=getattr(self.args, "imu_full_max_dp", 1.0),
            imu_gyro_bias=getattr(self.net, "imu_gyro_bias", None),
            imu_acc_bias=getattr(self.net, "imu_acc_bias", None),
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
        graph.clear_edges()
        graph.flush_imu_debug()
        self.video.dirty[:t] = True


class DroidAsyncBackend:
    def __init__(self, net, video, args, max_age = 7):
        self.video = video
        self.update_op = net.update
        self.net = net
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
            imu_full_pos_weight=getattr(args, "imu_full_pos_weight", 0.05),
            imu_full_vel_weight=getattr(args, "imu_full_vel_weight", 0.05),
            imu_full_bias_weight=getattr(args, "imu_full_bias_weight", 0.001),
            imu_motion_prior_weight=getattr(args, "imu_motion_prior_weight", 0.0),
            imu_local_bias_prior_weight=getattr(args, "imu_local_bias_prior_weight", 0.0),
            imu_gravity=getattr(args, "imu_gravity", None),
            imu_full_max_dt=getattr(args, "imu_full_max_dt", 0.5),
            imu_full_max_dv=getattr(args, "imu_full_max_dv", 5.0),
            imu_full_max_dp=getattr(args, "imu_full_max_dp", 1.0),
            imu_gyro_bias=getattr(net, "imu_gyro_bias", None),
            imu_acc_bias=getattr(net, "imu_acc_bias", None),
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
        self.graph.rm_factors(self.graph.age > self.max_age, store=True)
        self.graph.flush_imu_debug()

        self.video.dirty[:t] = True

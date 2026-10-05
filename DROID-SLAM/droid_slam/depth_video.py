import numpy as np
import torch
import lietorch
import droid_backends

from torch.multiprocessing import Process, Queue, Lock, Value
from collections import OrderedDict

from droid_net import cvx_upsample
import geom.projective_ops as pops


IMU_BIAS_JACOBIAN_ORDER = ("J_R_bg", "J_v_ba", "J_v_bg", "J_p_ba", "J_p_bg")

class DepthVideo:
    def __init__(self, image_size=[480, 640], buffer=1024, stereo=False, device="cuda:0"):
                
        # current keyframe count
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]

        ### state attributes ###
        self.tstamp = torch.zeros(buffer, device=device, dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device=device, dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        self.red = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()

        # IMU motion state for full inertial BA:
        # M_i = (v_i, ba_i, bg_i). Velocity uses the current DROID length unit.
        # Biases remain in physical IMU units (m/s^2 and rad/s), even when DROID
        # normalizes its monocular scale.
        self.velocities = torch.zeros(buffer, 3, device=device, dtype=torch.float).share_memory_()
        self.bias_acc = torch.zeros(buffer, 3, device=device, dtype=torch.float).share_memory_()
        self.bias_gyro = torch.zeros(buffer, 3, device=device, dtype=torch.float).share_memory_()
        self.imu_unit_scale = torch.ones(1, device=device, dtype=torch.float).share_memory_()
        self.imu_gravity = torch.zeros(3, device=device, dtype=torch.float).share_memory_()
        self.imu_global_bias_acc = torch.zeros(3, device=device, dtype=torch.float).share_memory_()
        self.imu_global_bias_gyro = torch.zeros(3, device=device, dtype=torch.float).share_memory_()
        # Static lever arm in physical metres: p_C = R_CI p_I + t_CI.  It is
        # deliberately kept metric; callers convert it through imu_unit_scale
        # while the monocular state is still in DROID's internal length unit.
        self.imu_camera_translation = torch.zeros(
            3, device=device, dtype=torch.float
        ).share_memory_()
        self.imu_camera_translation_source = "zero_lever_arm_fallback"
        self.imu_alignment_scale = torch.zeros(1, device=device, dtype=torch.float).share_memory_()
        self.imu_alignment_ready = Value('i', 0)
        self.imu_alignment_metrics = {}
        # Scalar records from accepted/rejected low-rate shared-bias updates.
        # Kept on the Python side because it is diagnostic metadata, not a
        # tensor needed by the CUDA DBA process.
        self.imu_bias_refinement_history = []
        self.imu_bias_relinearization_required = False

        # Per-frame preintegrated IMU measurement from previous selected/input
        # frame to this frame.
        #
        # imu_delta columns:
        #   [dt, dr_x, dr_y, dr_z, dv_x, dv_y, dv_z, dp_x, dp_y, dp_z]
        self.imu_delta = torch.zeros(buffer, 10, device=device, dtype=torch.float).share_memory_()
        self.imu_valid = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        self.imu_weight = torch.zeros(buffer, device=device, dtype=torch.float).share_memory_()
        self.imu_used_steps = torch.zeros(buffer, device=device, dtype=torch.float).share_memory_()
        self.imu_info = torch.zeros(buffer, 3, device=device, dtype=torch.float).share_memory_()
        # Per-row sensor bias random-walk densities in physical IMU units:
        # [accelerometer (m/s^2/sqrt(s)), gyroscope (rad/s/sqrt(s))].
        # They are calibration metadata, not keyframe optimization states.
        self.imu_bias_random_walk = torch.zeros(
            buffer, 2, device=device, dtype=torch.float
        ).share_memory_()
        self.imu_bias_jacobian = torch.zeros(
            buffer,
            len(IMU_BIAS_JACOBIAN_ORDER),
            3,
            3,
            device=device,
            dtype=torch.float,
        ).share_memory_()

        self.disps = torch.ones(buffer, ht//8, wd//8, device=device, dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht//8, wd//8, device=device, dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device=device, dtype=torch.float).share_memory_()

        self.stereo = stereo
        c = 1 if not self.stereo else 2

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht//8, wd//8, dtype=torch.half, device=device).share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device=device).share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device=device).share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        
    def to(self, device="cuda"):
        self.tstamp = self.tstamp.to(device=device)
        self.images = self.images.to(device=device)
        self.dirty = self.dirty.to(device=device)
        self.red = self.red.to(device=device)
        self.poses = self.poses.to(device=device)
        self.velocities = self.velocities.to(device=device)
        self.bias_acc = self.bias_acc.to(device=device)
        self.bias_gyro = self.bias_gyro.to(device=device)
        self.imu_unit_scale = self.imu_unit_scale.to(device=device)
        self.imu_gravity = self.imu_gravity.to(device=device)
        self.imu_global_bias_acc = self.imu_global_bias_acc.to(device=device)
        self.imu_global_bias_gyro = self.imu_global_bias_gyro.to(device=device)
        self.imu_camera_translation = self.imu_camera_translation.to(device=device)
        self.imu_alignment_scale = self.imu_alignment_scale.to(device=device)
        self.imu_delta = self.imu_delta.to(device=device)
        self.imu_valid = self.imu_valid.to(device=device)
        self.imu_weight = self.imu_weight.to(device=device)
        self.imu_used_steps = self.imu_used_steps.to(device=device)
        self.imu_info = self.imu_info.to(device=device)
        self.imu_bias_random_walk = self.imu_bias_random_walk.to(device=device)
        self.imu_bias_jacobian = self.imu_bias_jacobian.to(device=device)
        self.disps = self.disps.to(device=device)
        self.disps_sens = self.disps_sens.to(device=device)
        self.disps_up = self.disps_up.to(device=device)
        self.intrinsics = self.intrinsics.to(device=device)

        self.fmaps = self.fmaps.to(device=device)
        self.nets = self.nets.to(device=device)
        self.inps = self.inps.to(device=device)

        return self

    def __del__(self):
        # delete all tensors
        del self.tstamp
        del self.images
        del self.dirty
        del self.red
        del self.poses
        del self.velocities
        del self.bias_acc
        del self.bias_gyro
        del self.imu_unit_scale
        del self.imu_gravity
        del self.imu_global_bias_acc
        del self.imu_global_bias_gyro
        del self.imu_camera_translation
        del self.imu_alignment_scale
        del self.imu_delta
        del self.imu_valid
        del self.imu_weight
        del self.imu_used_steps
        del self.imu_info
        del self.imu_bias_random_walk
        del self.imu_bias_jacobian
        del self.disps
        del self.disps_sens
        del self.disps_up
        del self.intrinsics
        del self.fmaps
        del self.nets
        del self.inps

    def get_lock(self):
        return self.counter.get_lock()

    @torch.no_grad()
    def set_imu_camera_translation(self, translation, source="unknown"):
        """Set the metric IMU-origin lever arm shared by all keyframes."""

        value = torch.as_tensor(
            translation,
            device=self.imu_camera_translation.device,
            dtype=self.imu_camera_translation.dtype,
        ).reshape(-1)
        if value.numel() != 3 or not bool(torch.isfinite(value).all().item()):
            raise ValueError("imu_camera_translation must be three finite values")
        self.imu_camera_translation.copy_(value)
        self.imu_camera_translation_source = str(source)

    def imu_camera_translation_in_current_units(self):
        """Return ``t_CI`` in the same length unit as current pose centers."""

        return self.imu_camera_translation * self.imu_unit_scale[0]

    @staticmethod
    def __imu_value(imu_prior, key, default=0.0):
        if imu_prior is None:
            return default

        if hasattr(imu_prior, "get"):
            value = imu_prior.get(key, default)
        else:
            value = default

        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def __set_imu_prior(self, index, imu_prior):
        """Store one row from imu_prior.csv next to the selected frame state."""

        if imu_prior is None:
            return

        valid = int(self.__imu_value(imu_prior, "imu_valid", 1.0)) != 0
        used_steps = self.__imu_value(
            imu_prior,
            "imu_used_steps",
            self.__imu_value(imu_prior, "imu_count", 0.0),
        )

        delta = torch.as_tensor([
            self.__imu_value(imu_prior, "dt", 0.0),
            self.__imu_value(imu_prior, "dr_x", 0.0),
            self.__imu_value(imu_prior, "dr_y", 0.0),
            self.__imu_value(imu_prior, "dr_z", 0.0),
            self.__imu_value(imu_prior, "dv_x", 0.0),
            self.__imu_value(imu_prior, "dv_y", 0.0),
            self.__imu_value(imu_prior, "dv_z", 0.0),
            self.__imu_value(imu_prior, "dp_x", 0.0),
            self.__imu_value(imu_prior, "dp_y", 0.0),
            self.__imu_value(imu_prior, "dp_z", 0.0),
        ], device=self.imu_delta.device, dtype=self.imu_delta.dtype)

        # DROID normalizes monocular pose/depth scale during backend updates.
        # Keep IMU translation/velocity deltas in the same internal unit.
        delta[4:10] *= self.imu_unit_scale[0].to(device=delta.device, dtype=delta.dtype)

        self.imu_delta[index] = delta
        self.imu_valid[index] = valid
        self.imu_weight[index] = self.__imu_value(imu_prior, "imu_weight", 1.0)
        self.imu_used_steps[index] = used_steps
        self.imu_info[index] = torch.as_tensor([
            self.__imu_value(imu_prior, "rot_info", 0.0),
            self.__imu_value(imu_prior, "vel_info", 0.0),
            self.__imu_value(imu_prior, "pos_info", 0.0),
        ], device=self.imu_info.device, dtype=self.imu_info.dtype)
        self.imu_bias_random_walk[index] = torch.as_tensor([
            self.__imu_value(imu_prior, "accel_random_walk", 0.0),
            self.__imu_value(imu_prior, "gyro_random_walk", 0.0),
        ],
            device=self.imu_bias_random_walk.device,
            dtype=self.imu_bias_random_walk.dtype,
        )

        jacobian = torch.zeros_like(self.imu_bias_jacobian[index])
        for prefix_index, prefix in enumerate(IMU_BIAS_JACOBIAN_ORDER):
            for i in range(3):
                for j in range(3):
                    jacobian[prefix_index, i, j] = self.__imu_value(
                        imu_prior,
                        f"{prefix}_{i}{j}",
                        0.0,
                    )

        if torch.count_nonzero(jacobian).item() == 0 and delta[0] > 0.0:
            dt = delta[0]
            eye = torch.eye(3, device=jacobian.device, dtype=jacobian.dtype)
            jacobian[0] = -dt * eye
            jacobian[1] = -dt * eye
            jacobian[3] = -0.5 * dt * dt * eye

        jacobian[1:5] *= self.imu_unit_scale[0].to(
            device=jacobian.device,
            dtype=jacobian.dtype,
        )
        self.imu_bias_jacobian[index] = jacobian

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        # self.dirty[index] = True
        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if isinstance(index, int) and index > 0:
            self.velocities[index] = self.velocities[index - 1]
            self.bias_acc[index] = self.bias_acc[index - 1]
            self.bias_gyro[index] = self.bias_gyro[index - 1]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            depth = item[4][3::8,3::8].cuda()
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, depth)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

        if len(item) > 9:
            self.__set_imu_prior(index, item[9])

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.velocities[:self.counter.value] *= s
            # Accelerometer bias is a physical m/s^2 quantity. Only residual
            # terms expressed in DROID's visual length unit are rescaled.
            self.imu_gravity *= s
            self.imu_delta[:self.counter.value, 4:10] *= s
            self.imu_info[:self.counter.value, 1:3] /= (s * s).clamp_min(1e-12)
            self.imu_bias_jacobian[:self.counter.value, 1:5] *= s
            self.imu_unit_scale *= s
            self.dirty[:self.counter.value] = True

    @torch.no_grad()
    def set_imu_unit_scale(self, value, count=None):
        """Rescale stored translational IMU terms without changing physical bias."""

        if count is None:
            count = int(self.counter.value)
        count = max(0, min(int(count), self.imu_delta.shape[0]))
        target = torch.as_tensor(value, device=self.imu_unit_scale.device, dtype=self.imu_unit_scale.dtype)
        target = target.reshape(-1)[0]
        if not bool(torch.isfinite(target).detach().cpu().item()) or float(target.detach().cpu().item()) <= 0.0:
            raise ValueError("imu_unit_scale must be finite and positive")

        current = self.imu_unit_scale[0].clamp_min(1e-12)
        ratio = target / current
        if count > 0:
            self.imu_delta[:count, 4:10] *= ratio
            self.imu_info[:count, 1:3] /= (ratio * ratio).clamp_min(1e-12)
            self.imu_bias_jacobian[:count, 1:5] *= ratio
        self.imu_unit_scale.fill_(target)


    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d

    def ba(
        self,
        target,
        weight,
        eta,
        ii,
        jj,
        t0=1,
        t1=None,
        itrs=2,
        lm=1e-4,
        ep=0.1,
        motion_only=False,
        pose_prior_H=None,
        pose_prior_v=None,
        pose_prior_ii=None,
        pose_prior_jj=None,
    ):
        """ dense bundle adjustment (DBA) """

        with self.get_lock():
            ba_result = None

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            has_pose_prior = (
                pose_prior_H is not None
                and pose_prior_v is not None
                and pose_prior_ii is not None
                and pose_prior_jj is not None
                and pose_prior_H.numel() > 0
                and pose_prior_v.numel() > 0
            )

            if has_pose_prior:
                state_dim = int(pose_prior_H.shape[-1])
                if state_dim in (9, 15):
                    ba_result = droid_backends.ba_with_full_prior(
                        self.poses,
                        self.velocities,
                        self.bias_acc,
                        self.bias_gyro,
                        self.disps,
                        self.intrinsics[0],
                        self.disps_sens,
                        target,
                        weight,
                        eta,
                        ii,
                        jj,
                        t0,
                        t1,
                        itrs,
                        lm,
                        ep,
                        motion_only,
                        pose_prior_H.contiguous(),
                        pose_prior_v.contiguous(),
                        pose_prior_ii.contiguous(),
                        pose_prior_jj.contiguous(),
                    )
                else:
                    ba_result = droid_backends.ba_with_prior(
                        self.poses,
                        self.disps,
                        self.intrinsics[0],
                        self.disps_sens,
                        target,
                        weight,
                        eta,
                        ii,
                        jj,
                        t0,
                        t1,
                        itrs,
                        lm,
                        ep,
                        motion_only,
                        pose_prior_H.contiguous(),
                        pose_prior_v.contiguous(),
                        pose_prior_ii.contiguous(),
                        pose_prior_jj.contiguous(),
                    )
            else:
                ba_result = droid_backends.ba(
                    self.poses,
                    self.disps,
                    self.intrinsics[0],
                    self.disps_sens,
                    target,
                    weight,
                    eta,
                    ii,
                    jj,
                    t0,
                    t1,
                    itrs,
                    lm,
                    ep,
                    motion_only,
                )

            self.disps.clamp_(min=0.001)
            return ba_result

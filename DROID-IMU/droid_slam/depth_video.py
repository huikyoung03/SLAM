import numpy as np
import torch
import lietorch
import droid_backends

from torch.multiprocessing import Process, Queue, Lock, Value
from collections import OrderedDict

from droid_net import cvx_upsample
import geom.projective_ops as pops

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

        # IMU motion state placeholders for future inertial BA.
        #
        # DROID's CUDA BA does not optimize these tensors yet. They are kept here
        # because DepthVideo is the shared frame-state container used by frontend,
        # backend, and visualization code.
        #
        # M_i = (v_i, ba_i, bg_i)
        self.velocities = torch.zeros(buffer, 3, device=device, dtype=torch.float).share_memory_()
        self.bias_acc = torch.zeros(buffer, 3, device=device, dtype=torch.float).share_memory_()
        self.bias_gyro = torch.zeros(buffer, 3, device=device, dtype=torch.float).share_memory_()

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
        self.imu_delta = self.imu_delta.to(device=device)
        self.imu_valid = self.imu_valid.to(device=device)
        self.imu_weight = self.imu_weight.to(device=device)
        self.imu_used_steps = self.imu_used_steps.to(device=device)
        self.imu_info = self.imu_info.to(device=device)
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
        del self.imu_delta
        del self.imu_valid
        del self.imu_weight
        del self.imu_used_steps
        del self.imu_info
        del self.disps
        del self.disps_sens
        del self.disps_up
        del self.intrinsics
        del self.fmaps
        del self.nets
        del self.inps

    def get_lock(self):
        return self.counter.get_lock()

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

        self.imu_delta[index] = delta
        self.imu_valid[index] = valid
        self.imu_weight[index] = self.__imu_value(imu_prior, "imu_weight", 1.0)
        self.imu_used_steps[index] = used_steps
        self.imu_info[index] = torch.as_tensor([
            self.__imu_value(imu_prior, "rot_info", 0.0),
            self.__imu_value(imu_prior, "vel_info", 0.0),
            self.__imu_value(imu_prior, "pos_info", 0.0),
        ], device=self.imu_info.device, dtype=self.imu_info.dtype)

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
            self.dirty[:self.counter.value] = True


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
                if state_dim == 15:
                    droid_backends.ba_with_full_prior(
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
                    droid_backends.ba_with_prior(
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
                droid_backends.ba(
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

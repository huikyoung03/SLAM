
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import csv
import os
import cv2
import math
import random
import json
import pickle
import os.path as osp

from .augmentation import RGBDAugmentor
from .rgbd_utils import *

def _to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _to_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _quat_normalize_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_multiply_wxyz(q1, q2):
    w1, x1, y1, z1 = _quat_normalize_wxyz(q1)
    w2, x2, y2, z2 = _quat_normalize_wxyz(q2)
    return _quat_normalize_wxyz([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_to_rotvec_wxyz(q):
    w, x, y, z = _quat_normalize_wxyz(q)
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z

    sin_half = math.sqrt(x * x + y * y + z * z)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)

    angle = 2.0 * math.atan2(sin_half, w)
    scale = angle / sin_half
    return np.array([x * scale, y * scale, z * scale], dtype=np.float32)


class RGBDDataset(data.Dataset):
    def __init__(
        self,
        name,
        datapath,
        n_frames=4,
        crop_size=[384,512],
        fmin=8.0,
        fmax=75.0,
        do_aug=True,
        use_imu=False,
        imu_prior_name="imu_prior.csv",
        imu_require=False,
    ):
        """ Base class for RGBD dataset """
        self.aug = None
        self.root = datapath
        self.name = name

        self.n_frames = n_frames
        self.fmin = fmin # exclude very easy examples
        self.fmax = fmax # exclude very hard examples
        self.use_imu = use_imu
        self.imu_prior_name = imu_prior_name
        self.imu_require = imu_require
        
        if do_aug:
            self.aug = RGBDAugmentor(crop_size=crop_size)

        # building dataset is expensive, cache so only needs to be performed once
        cur_path = osp.dirname(osp.abspath(__file__))
        if not os.path.isdir(osp.join(cur_path, 'cache')):
            os.mkdir(osp.join(cur_path, 'cache'))
        
        cache_path = osp.join(cur_path, 'cache', '{}.pickle'.format(self.name))

        if osp.isfile(cache_path):
            scene_info = pickle.load(open(cache_path, 'rb'))[0]
        else:
            scene_info = self._build_dataset()
            with open(cache_path, 'wb') as cachefile:
                pickle.dump((scene_info,), cachefile)

        self.scene_info = scene_info
        if self.use_imu:
            self._attach_imu_priors()
        self._build_dataset_index()
                
    def _build_dataset_index(self):
        self.dataset_index = []
        for scene in self.scene_info:
            if not self.__class__.is_test_scene(scene):
                graph = self.scene_info[scene]['graph']
                for i in graph:
                    if len(graph[i][0]) > self.n_frames:
                        self.dataset_index.append((scene, i))
            else:
                print("Reserving {} for validation".format(scene))

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        return np.load(depth_file)

    @staticmethod
    def read_imu_prior_csv(path):
        priors = {}
        with open(path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                frame_index = _to_int(
                    row.get("frame_index"),
                    _to_int(row.get("frame_id"), len(priors)),
                )
                priors[frame_index] = {
                    "dt": _to_float(row.get("dt")),
                    "dr_x": _to_float(row.get("dr_x")),
                    "dr_y": _to_float(row.get("dr_y")),
                    "dr_z": _to_float(row.get("dr_z")),
                    "dq_w": _to_float(row.get("dq_w"), 1.0),
                    "dq_x": _to_float(row.get("dq_x")),
                    "dq_y": _to_float(row.get("dq_y")),
                    "dq_z": _to_float(row.get("dq_z")),
                    "dv_x": _to_float(row.get("dv_x")),
                    "dv_y": _to_float(row.get("dv_y")),
                    "dv_z": _to_float(row.get("dv_z")),
                    "dp_x": _to_float(row.get("dp_x")),
                    "dp_y": _to_float(row.get("dp_y")),
                    "dp_z": _to_float(row.get("dp_z")),
                    "rot_var": _to_float(row.get("rot_var")),
                    "vel_var": _to_float(row.get("vel_var")),
                    "pos_var": _to_float(row.get("pos_var")),
                    "rot_info": _to_float(row.get("rot_info")),
                    "vel_info": _to_float(row.get("vel_info")),
                    "pos_info": _to_float(row.get("pos_info")),
                    "imu_valid": _to_int(row.get("imu_valid"), 1),
                }

        return priors

    def _attach_imu_priors(self):
        loaded = 0
        missing = 0

        for scene, info in self.scene_info.items():
            prior_path = osp.join(scene, self.imu_prior_name)
            if not osp.isfile(prior_path):
                missing += 1
                if self.imu_require:
                    raise FileNotFoundError(f"imu prior not found: {prior_path}")
                continue

            info["imu_priors"] = self.read_imu_prior_csv(prior_path)
            loaded += 1

        print(f"[IMU DATA] loaded imu priors for {loaded} scenes, missing={missing}")

    @staticmethod
    def _empty_imu_sequence(num_frames):
        return (
            np.zeros((num_frames, 10), dtype=np.float32),
            np.zeros((num_frames,), dtype=np.float32),
            np.zeros((num_frames, 3), dtype=np.float32),
        )

    @staticmethod
    def _compose_imu_interval(priors, start, end):
        if priors is None or end <= start:
            return np.zeros(10, dtype=np.float32), 0.0, np.zeros(3, dtype=np.float32)

        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        delta = np.zeros(10, dtype=np.float32)
        valid = 1.0
        rot_var = 0.0
        vel_var = 0.0
        pos_var = 0.0

        for frame_index in range(start + 1, end + 1):
            row = priors.get(frame_index)
            if row is None:
                return np.zeros(10, dtype=np.float32), 0.0, np.zeros(3, dtype=np.float32)

            q_step = [
                row["dq_w"],
                row["dq_x"],
                row["dq_y"],
                row["dq_z"],
            ]
            q = _quat_multiply_wxyz(q, q_step)

            delta[0] += row["dt"]
            delta[4] += row["dv_x"]
            delta[5] += row["dv_y"]
            delta[6] += row["dv_z"]
            delta[7] += row["dp_x"]
            delta[8] += row["dp_y"]
            delta[9] += row["dp_z"]
            valid *= float(row["imu_valid"] != 0)

            rot_var += max(float(row["rot_var"]), 0.0)
            vel_var += max(float(row["vel_var"]), 0.0)
            pos_var += max(float(row["pos_var"]), 0.0)

        delta[1:4] = _quat_to_rotvec_wxyz(q)
        imu_info = np.array([
            1.0 / rot_var if rot_var > 1e-18 else 0.0,
            1.0 / vel_var if vel_var > 1e-18 else 0.0,
            1.0 / pos_var if pos_var > 1e-18 else 0.0,
        ], dtype=np.float32)

        return delta, valid, imu_info

    def build_imu_sequence(self, scene_id, inds):
        priors = self.scene_info[scene_id].get("imu_priors")
        imu_delta, imu_valid, imu_info = self._empty_imu_sequence(len(inds))

        if priors is None:
            return imu_delta, imu_valid, imu_info

        for k in range(1, len(inds)):
            delta, valid, info = self._compose_imu_interval(priors, int(inds[k - 1]), int(inds[k]))
            imu_delta[k] = delta
            imu_valid[k] = valid
            imu_info[k] = info

        return imu_delta, imu_valid, imu_info

    def build_frame_graph(self, poses, depths, intrinsics, f=16, max_flow=256):
        """ compute optical flow distance between all pairs of frames """
        def read_disp(fn):
            depth = self.__class__.depth_read(fn)[f//2::f, f//2::f]
            depth[depth < 0.01] = np.mean(depth)
            return 1.0 / depth

        poses = np.array(poses)
        intrinsics = np.array(intrinsics) / f
        
        disps = np.stack(list(map(read_disp, depths)), 0)
        d = f * compute_distance_matrix_flow(poses, disps, intrinsics)

        # uncomment for nice visualization
        # import matplotlib.pyplot as plt
        # plt.imshow(d)
        # plt.show()

        graph = {}
        for i in range(d.shape[0]):
            j, = np.where(d[i] < max_flow)
            graph[i] = (j, d[i,j])

        return graph

    def __getitem__(self, index):
        """ return training video """

        index = index % len(self.dataset_index)
        scene_id, ix = self.dataset_index[index]

        frame_graph = self.scene_info[scene_id]['graph']
        images_list = self.scene_info[scene_id]['images']
        depths_list = self.scene_info[scene_id]['depths']
        poses_list = self.scene_info[scene_id]['poses']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']

        inds = [ ix ]
        while len(inds) < self.n_frames:
            # get other frames within flow threshold
            k = (frame_graph[ix][1] > self.fmin) & (frame_graph[ix][1] < self.fmax)
            frames = frame_graph[ix][0][k]

            # prefer frames forward in time
            if np.count_nonzero(frames[frames > ix]):
                ix = np.random.choice(frames[frames > ix])
            
            elif np.count_nonzero(frames):
                ix = np.random.choice(frames)

            inds += [ ix ]

        images, depths, poses, intrinsics = [], [], [], []
        for i in inds:
            images.append(self.__class__.image_read(images_list[i]))
            depths.append(self.__class__.depth_read(depths_list[i]))
            poses.append(poses_list[i])
            intrinsics.append(intrinsics_list[i])

        if self.use_imu:
            imu_delta, imu_valid, imu_info = self.build_imu_sequence(scene_id, inds)

        images = np.stack(images).astype(np.float32)
        depths = np.stack(depths).astype(np.float32)
        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)

        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)

        disps = torch.from_numpy(1.0 / depths)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        if self.aug is not None:
            images, poses, disps, intrinsics = \
                self.aug(images, poses, disps, intrinsics)

        # scale scene
        if len(disps[disps>0.01]) > 0:
            s = disps[disps>0.01].mean()
            disps = disps / s
            poses[...,:3] *= s

        if self.use_imu:
            return (
                images,
                poses,
                disps,
                intrinsics,
                torch.from_numpy(imu_delta),
                torch.from_numpy(imu_valid),
                torch.from_numpy(imu_info),
            )

        return images, poses, disps, intrinsics 

    def __len__(self):
        return len(self.dataset_index)

    def __imul__(self, x):
        self.dataset_index *= x
        return self

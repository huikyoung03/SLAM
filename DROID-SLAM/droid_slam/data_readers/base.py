
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

try:
    from imu_prior import compose_imu_prior_rows
except ImportError:
    from droid_slam.imu_prior import compose_imu_prior_rows


IMU_BIAS_JACOBIAN_ORDER = ("J_R_bg", "J_v_ba", "J_v_bg", "J_p_ba", "J_p_bg")


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


def _eye_mat3(scale=1.0):
    return np.eye(3, dtype=np.float32) * float(scale)


def _zero_bias_jacobians(dt=0.0):
    dt = max(float(dt), 0.0)
    return {
        "J_R_bg": _eye_mat3(-dt),
        "J_v_ba": _eye_mat3(-dt),
        "J_v_bg": np.zeros((3, 3), dtype=np.float32),
        "J_p_ba": _eye_mat3(-0.5 * dt * dt),
        "J_p_bg": np.zeros((3, 3), dtype=np.float32),
    }


def _read_mat3(row, prefix, fallback):
    values = []
    for i in range(3):
        values.append([
            _to_float(row.get(f"{prefix}_{i}{j}"), None)
            for j in range(3)
        ])

    if any(value is None for line in values for value in line):
        return fallback.astype(np.float32)

    return np.asarray(values, dtype=np.float32)


def _bias_jacobian_array(row):
    fallback = _zero_bias_jacobians(_to_float(row.get("dt"), 0.0) if row else 0.0)
    return np.stack([
        _read_mat3(row or {}, prefix, fallback[prefix])
        for prefix in IMU_BIAS_JACOBIAN_ORDER
    ], axis=0).astype(np.float32)


def _copy_bias_jacobian_fields(target, source, scale=1.0):
    scale = float(scale)
    for prefix_index, prefix in enumerate(IMU_BIAS_JACOBIAN_ORDER):
        for i in range(3):
            for j in range(3):
                value = _to_float(source.get(f"{prefix}_{i}{j}"), 0.0)
                if prefix_index > 0:
                    value *= scale
                target[f"{prefix}_{i}{j}"] = float(value)


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
                parsed = {
                    "frame_id": _to_int(row.get("frame_id"), frame_index),
                    "frame_index": frame_index,
                    "imu_count": _to_float(row.get("imu_count")),
                    "imu_used_steps": _to_float(
                        row.get("imu_used_steps"),
                        _to_float(row.get("imu_count")),
                    ),
                    "imu_weight": _to_float(row.get("imu_weight"), 1.0),
                    "imu_reason": row.get("imu_reason", ""),
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
                fallback = _zero_bias_jacobians(parsed["dt"])
                has_jacobian = any(
                    row.get(f"{prefix}_{i}{j}") not in (None, "")
                    for prefix in IMU_BIAS_JACOBIAN_ORDER
                    for i in range(3)
                    for j in range(3)
                )
                if has_jacobian:
                    jacobian = _bias_jacobian_array(row)
                else:
                    jacobian = np.stack([
                        fallback[prefix]
                        for prefix in IMU_BIAS_JACOBIAN_ORDER
                    ], axis=0)

                for prefix_index, prefix in enumerate(IMU_BIAS_JACOBIAN_ORDER):
                    for i in range(3):
                        for j in range(3):
                            parsed[f"{prefix}_{i}{j}"] = float(jacobian[prefix_index, i, j])
                priors[frame_index] = parsed

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
            # Keep the nominal camera-rate information separate from individual
            # training windows. A sparse visual interval can be composed from
            # many rows, and its uncertainty must remain relative to this
            # original sensor reference.
            info["imu_info_reference"] = self._estimate_imu_info_reference(
                info["imu_priors"]
            )
            loaded += 1

        print(f"[IMU DATA] loaded imu priors for {loaded} scenes, missing={missing}")

    @staticmethod
    def _estimate_imu_info_reference(priors, eps=1e-12):
        """Return median raw-camera information `[rotation, velocity, position]`.

        These values are a relative sensor-noise reference only. The learned
        global factor scale still decides the visual-versus-IMU balance, while a
        composed interval is attenuated according to its covariance relative to
        the original camera-rate preintegration.
        """

        if not priors:
            return None

        reference = []
        for key in ("rot_info", "vel_info", "pos_info"):
            values = [
                _to_float(row.get(key), 0.0)
                for row in priors.values()
                if _to_int(row.get("imu_valid"), 1) != 0
                and _to_float(row.get(key), 0.0) > float(eps)
            ]
            if not values:
                return None
            reference.append(float(np.median(np.asarray(values, dtype=np.float64))))

        return np.asarray(reference, dtype=np.float32)

    @staticmethod
    def _empty_imu_sequence(num_frames):
        return (
            np.zeros((num_frames, 10), dtype=np.float32),
            np.zeros((num_frames,), dtype=np.float32),
            np.zeros((num_frames, 3), dtype=np.float32),
            np.zeros((num_frames, len(IMU_BIAS_JACOBIAN_ORDER), 3, 3), dtype=np.float32),
        )

    @staticmethod
    def _compose_imu_interval(priors, start, end):
        if priors is None or end <= start:
            return (
                np.zeros(10, dtype=np.float32),
                0.0,
                np.zeros(3, dtype=np.float32),
                np.zeros((len(IMU_BIAS_JACOBIAN_ORDER), 3, 3), dtype=np.float32),
            )

        rows = []
        for frame_index in range(start + 1, end + 1):
            row = priors.get(frame_index)
            if row is None:
                return (
                    np.zeros(10, dtype=np.float32),
                    0.0,
                    np.zeros(3, dtype=np.float32),
                    np.zeros((len(IMU_BIAS_JACOBIAN_ORDER), 3, 3), dtype=np.float32),
                )
            rows.append(row)

        composed = compose_imu_prior_rows(
            rows,
            prev_frame_index=start,
            frame_index=end,
        )
        if composed is None:
            return (
                np.zeros(10, dtype=np.float32),
                0.0,
                np.zeros(3, dtype=np.float32),
                np.zeros((len(IMU_BIAS_JACOBIAN_ORDER), 3, 3), dtype=np.float32),
            )

        delta = np.zeros(10, dtype=np.float32)
        delta[0] = _to_float(composed.get("dt"))
        delta[1] = _to_float(composed.get("dr_x"))
        delta[2] = _to_float(composed.get("dr_y"))
        delta[3] = _to_float(composed.get("dr_z"))
        delta[4] = _to_float(composed.get("dv_x"))
        delta[5] = _to_float(composed.get("dv_y"))
        delta[6] = _to_float(composed.get("dv_z"))
        delta[7] = _to_float(composed.get("dp_x"))
        delta[8] = _to_float(composed.get("dp_y"))
        delta[9] = _to_float(composed.get("dp_z"))
        valid = float(_to_int(composed.get("imu_valid"), 1) != 0)
        imu_info = np.array([
            _to_float(composed.get("rot_info")),
            _to_float(composed.get("vel_info")),
            _to_float(composed.get("pos_info")),
        ], dtype=np.float32)
        imu_bias_jac = _bias_jacobian_array(composed)

        return delta, valid, imu_info, imu_bias_jac

    def build_imu_sequence(self, scene_id, inds):
        priors = self.scene_info[scene_id].get("imu_priors")
        imu_delta, imu_valid, imu_info, imu_bias_jac = self._empty_imu_sequence(len(inds))

        if priors is None:
            return imu_delta, imu_valid, imu_info, imu_bias_jac

        for k in range(1, len(inds)):
            delta, valid, info, jac = self._compose_imu_interval(priors, int(inds[k - 1]), int(inds[k]))
            imu_delta[k] = delta
            imu_valid[k] = valid
            imu_info[k] = info
            imu_bias_jac[k] = jac

        return imu_delta, imu_valid, imu_info, imu_bias_jac

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
            imu_delta, imu_valid, imu_info, imu_bias_jacobian = self.build_imu_sequence(scene_id, inds)
            imu_info_reference = self.scene_info[scene_id].get("imu_info_reference")
            if imu_info_reference is None:
                imu_info_reference = np.zeros(3, dtype=np.float32)

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
                scale = float(s.detach().cpu().item())
                imu_delta = imu_delta.copy()
                imu_delta[:, 4:10] *= scale
                imu_bias_jacobian = imu_bias_jacobian.copy()
                imu_bias_jacobian[:, 1:5] *= scale

        if self.use_imu:
            return (
                images,
                poses,
                disps,
                intrinsics,
                torch.from_numpy(imu_delta),
                torch.from_numpy(imu_valid),
                torch.from_numpy(imu_info),
                torch.from_numpy(imu_bias_jacobian),
                torch.from_numpy(np.asarray(imu_info_reference, dtype=np.float32)),
            )

        return images, poses, disps, intrinsics 

    def __len__(self):
        return len(self.dataset_index)

    def __imul__(self, x):
        self.dataset_index *= x
        return self

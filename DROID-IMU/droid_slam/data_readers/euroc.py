import csv
import glob
import os
import os.path as osp
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data

from scipy.spatial.transform import Rotation, Slerp

from .augmentation import RGBDAugmentor
from .base import RGBDDataset


def _read_yaml_vector(path, key):
    text = open(path, "r", encoding="utf-8").read()
    match = re.search(rf"^\s*{re.escape(key)}\s*:\s*\[([^\]]+)\]", text, re.M)
    if match is None:
        return None
    return np.array(
        [float(v) for v in re.split(r"[,\s]+", match.group(1).strip()) if v],
        dtype=np.float64,
    )


def _read_yaml_matrix4(path, key="T_BS"):
    text = open(path, "r", encoding="utf-8").read()
    block = re.search(rf"{re.escape(key)}\s*:\s*(.*?)(?:\n\S|\Z)", text, re.S)
    search_text = block.group(1) if block else text
    match = re.search(r"data\s*:\s*\[([^\]]+)\]", search_text, re.S)
    if match is None:
        raise ValueError(f"cannot find {key}.data in {path}")

    values = [float(v) for v in re.split(r"[,\s]+", match.group(1).strip()) if v]
    if len(values) != 16:
        raise ValueError(f"{path} {key}.data must contain 16 values, got {len(values)}")
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def _timestamp_ns(row):
    for key, value in row.items():
        if value in (None, ""):
            continue
        if "timestamp" not in key.lower():
            continue
        stamp = float(value)
        if "[ns]" in key.lower() or stamp > 1.0e12:
            return int(stamp)
        return int(stamp * 1_000_000_000)
    raise ValueError(f"timestamp column not found in row keys={list(row.keys())}")


def _read_camera_rows(cam_csv):
    rows = []
    with open(cam_csv, "r", newline="", encoding="utf-8") as f:
        for index, row in enumerate(csv.DictReader(f)):
            filename = row.get("filename")
            if filename is None:
                raise ValueError(f"filename column not found in {cam_csv}")
            rows.append({
                "frame_index": index,
                "timestamp_ns": _timestamp_ns(row),
                "filename": filename,
            })
    rows.sort(key=lambda row: row["timestamp_ns"])
    return rows


def _read_groundtruth(gt_csv):
    data = np.loadtxt(gt_csv, delimiter=",", comments="#", dtype=np.float64)
    if data.ndim == 1:
        data = data[None]

    timestamps_ns = data[:, 0].astype(np.int64)
    positions = data[:, 1:4]
    quats_wxyz = data[:, 4:8]
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]

    order = np.argsort(timestamps_ns)
    timestamps_ns = timestamps_ns[order]
    positions = positions[order]
    quats_xyzw = quats_xyzw[order]

    return timestamps_ns, positions, quats_xyzw


def _interpolate_body_poses(target_timestamps_ns, gt_timestamps_ns, positions, quats_xyzw):
    t0 = float(gt_timestamps_ns[0])
    gt_times = (gt_timestamps_ns.astype(np.float64) - t0) / 1.0e9
    target_times = (np.asarray(target_timestamps_ns, dtype=np.float64) - t0) / 1.0e9

    interp_pos = np.stack(
        [np.interp(target_times, gt_times, positions[:, axis]) for axis in range(3)],
        axis=1,
    )

    rotations = Rotation.from_quat(quats_xyzw)
    interp_rot = Slerp(gt_times, rotations)(target_times)
    return interp_pos, interp_rot


def _matrix_to_se3_vec(T):
    quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.concatenate([T[:3, 3], quat_xyzw], axis=0).astype(np.float32)


def _discover_euroc_sequences(root):
    root = osp.abspath(root)
    if osp.isfile(osp.join(root, "cam0", "data.csv")):
        return [root]

    mav0 = osp.join(root, "mav0")
    if osp.isfile(osp.join(mav0, "cam0", "data.csv")):
        return [mav0]

    matches = sorted(glob.glob(osp.join(root, "**", "mav0", "cam0", "data.csv"), recursive=True))
    return [osp.dirname(osp.dirname(path)) for path in matches]


class EuRoC(data.Dataset):
    """EuRoC cam0 + IMU training reader.

    EuRoC does not provide dense depth in the format DROID-SLAM's RGBD trainer
    expects. This reader therefore emits a constant placeholder disparity so the
    recurrent DBA loop can run, while `train.py` disables GT-depth flow loss for
    this dataset and trains with pose + visual residual + IMU losses.
    """

    has_dense_depth = False

    def __init__(
        self,
        datapath,
        n_frames=7,
        crop_size=[384, 512],
        do_aug=True,
        use_imu=False,
        imu_prior_name="imu_prior.csv",
        imu_require=False,
        placeholder_depth=1.0,
        **kwargs,
    ):
        self.root = datapath
        self.n_frames = int(n_frames)
        self.use_imu = bool(use_imu)
        self.imu_prior_name = imu_prior_name
        self.imu_require = bool(imu_require)
        self.placeholder_depth = float(placeholder_depth)
        self.aug = RGBDAugmentor(crop_size=crop_size) if do_aug else None

        self.scenes = self._build_scenes()
        self.dataset_index = []
        for scene_id, scene in enumerate(self.scenes):
            count = len(scene["images"])
            for start in range(0, max(count - self.n_frames + 1, 0)):
                self.dataset_index.append((scene_id, start))

    def _build_scenes(self):
        sequence_roots = _discover_euroc_sequences(self.root)
        if not sequence_roots:
            raise FileNotFoundError(f"EuRoC sequence not found under {self.root}")

        scenes = []
        for sequence_root in sequence_roots:
            scene = self._load_scene(sequence_root)
            if len(scene["images"]) >= self.n_frames:
                scenes.append(scene)

        if not scenes:
            raise ValueError(f"no EuRoC scene has at least {self.n_frames} cam0 frames")

        total_frames = sum(len(scene["images"]) for scene in scenes)
        print(f"Building EuRoC dataset: scenes={len(scenes)}, frames={total_frames}")
        return scenes

    def _load_scene(self, mav0_root):
        cam_dir = osp.join(mav0_root, "cam0")
        imu_dir = osp.join(mav0_root, "imu0")
        gt_dir = osp.join(mav0_root, "state_groundtruth_estimate0")

        cam_csv = osp.join(cam_dir, "data.csv")
        cam_yaml = osp.join(cam_dir, "sensor.yaml")
        imu_csv = osp.join(imu_dir, "data.csv")
        imu_yaml = osp.join(imu_dir, "sensor.yaml")
        gt_csv = osp.join(gt_dir, "data.csv")

        cam_rows = _read_camera_rows(cam_csv)
        gt_ts, gt_pos, gt_quat = _read_groundtruth(gt_csv)

        valid_rows = [
            row for row in cam_rows
            if gt_ts[0] <= row["timestamp_ns"] <= gt_ts[-1]
        ]
        if len(valid_rows) < self.n_frames:
            raise ValueError(f"not enough cam0 frames with GT pose in {mav0_root}")

        timestamps = [row["timestamp_ns"] for row in valid_rows]
        body_pos, body_rot = _interpolate_body_poses(timestamps, gt_ts, gt_pos, gt_quat)

        T_body_cam = _read_yaml_matrix4(cam_yaml, "T_BS")
        poses_w2c = []
        for pos, rot in zip(body_pos, body_rot):
            T_world_body = np.eye(4, dtype=np.float64)
            T_world_body[:3, :3] = rot.as_matrix()
            T_world_body[:3, 3] = pos

            T_world_cam = T_world_body @ T_body_cam
            T_cam_world = np.linalg.inv(T_world_cam)
            poses_w2c.append(_matrix_to_se3_vec(T_cam_world))

        intrinsics = _read_yaml_vector(cam_yaml, "intrinsics")
        if intrinsics is None or intrinsics.size != 4:
            raise ValueError(f"cam0 intrinsics not found in {cam_yaml}")

        distortion = _read_yaml_vector(cam_yaml, "distortion_coefficients")
        K = np.array(
            [
                [intrinsics[0], 0.0, intrinsics[2]],
                [0.0, intrinsics[1], intrinsics[3]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        imu_priors = None
        if self.use_imu:
            imu_priors = self._load_or_build_imu_priors(
                mav0_root,
                valid_rows,
                cam_csv,
                cam_yaml,
                imu_csv,
                imu_yaml,
            )

        return {
            "id": mav0_root,
            "images": [osp.join(cam_dir, "data", row["filename"]) for row in valid_rows],
            "timestamps_ns": np.asarray(timestamps, dtype=np.int64),
            "frame_indices": np.asarray([row["frame_index"] for row in valid_rows], dtype=np.int64),
            "poses": np.stack(poses_w2c, axis=0),
            "intrinsics": np.tile(intrinsics.astype(np.float32)[None], (len(valid_rows), 1)),
            "K": K,
            "distortion": None if distortion is None else distortion.astype(np.float64),
            "imu_priors": imu_priors,
        }

    def _load_or_build_imu_priors(
        self,
        mav0_root,
        valid_rows,
        cam_csv,
        cam_yaml,
        imu_csv,
        imu_yaml,
    ):
        from tools.imu_preintegrate import (
            build_imu_calibration,
            compute_imu_weight,
            estimate_preintegration_uncertainty,
            integrate_imu_window,
            load_imu,
            select_imu_window,
        )

        if not osp.isfile(imu_csv):
            candidates = [
                osp.join(mav0_root, "droid_cam0", self.imu_prior_name),
                osp.join(mav0_root, "droid_cam0", "imu_prior_cam0.csv"),
                osp.join(mav0_root, self.imu_prior_name),
            ]
            for path in candidates:
                if osp.isfile(path):
                    return RGBDDataset.read_imu_prior_csv(path)

            if self.imu_require:
                raise FileNotFoundError(f"imu0/data.csv not found: {imu_csv}")
            return None

        calibration = build_imu_calibration(
            cam_sensor_yaml=Path(cam_yaml),
            imu_sensor_yaml=Path(imu_yaml),
        )
        imu = load_imu(imu_csv, calibration=calibration)
        imu_timestamps = [int(sample["timestamp_ns"]) for sample in imu]

        priors = {}
        previous_ns = None
        for row in _read_camera_rows(cam_csv):
            current_ns = int(row["timestamp_ns"])
            window, status = select_imu_window(imu, imu_timestamps, previous_ns, current_ns)
            integ = integrate_imu_window(window, previous_ns, current_ns, window_status=status)
            _, valid, _ = compute_imu_weight(
                dt=float(integ["dt"]),
                imu_count=int(integ["imu_count"]),
                used_steps=int(integ["imu_used_steps"]),
                dr_norm_deg=float(integ["dr_norm_deg"]),
                has_nan=bool(integ["has_nan"]),
                window_status=str(integ["window_status"]),
            )
            uncertainty = estimate_preintegration_uncertainty(float(integ["dt"]), calibration)
            priors[int(row["frame_index"])] = {
                "dt": float(integ["dt"]),
                "dr_x": float(integ["dr_x"]),
                "dr_y": float(integ["dr_y"]),
                "dr_z": float(integ["dr_z"]),
                "dq_w": float(integ["dq_w"]),
                "dq_x": float(integ["dq_x"]),
                "dq_y": float(integ["dq_y"]),
                "dq_z": float(integ["dq_z"]),
                "dv_x": float(integ["dv_x"]),
                "dv_y": float(integ["dv_y"]),
                "dv_z": float(integ["dv_z"]),
                "dp_x": float(integ["dp_x"]),
                "dp_y": float(integ["dp_y"]),
                "dp_z": float(integ["dp_z"]),
                "rot_var": float(uncertainty["rot_var"]),
                "vel_var": float(uncertainty["vel_var"]),
                "pos_var": float(uncertainty["pos_var"]),
                "rot_info": float(uncertainty["rot_info"]),
                "vel_info": float(uncertainty["vel_info"]),
                "pos_info": float(uncertainty["pos_info"]),
                "imu_valid": int(valid),
            }
            previous_ns = current_ns

        return priors

    def _read_image(self, scene, index):
        image = cv2.imread(scene["images"][index], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(scene["images"][index])

        distortion = scene["distortion"]
        if distortion is not None and np.any(np.abs(distortion) > 0.0):
            image = cv2.undistort(image, scene["K"], distortion, None, scene["K"])

        return image

    def _build_imu_sequence(self, scene, inds):
        imu_delta, imu_valid, imu_info = RGBDDataset._empty_imu_sequence(len(inds))
        priors = scene.get("imu_priors")
        if priors is None:
            return imu_delta, imu_valid, imu_info

        original = scene["frame_indices"]
        for k in range(1, len(inds)):
            start = int(original[int(inds[k - 1])])
            end = int(original[int(inds[k])])
            delta, valid, info = RGBDDataset._compose_imu_interval(priors, start, end)
            imu_delta[k] = delta
            imu_valid[k] = valid
            imu_info[k] = info

        return imu_delta, imu_valid, imu_info

    def __getitem__(self, index):
        scene_id, start = self.dataset_index[index % len(self.dataset_index)]
        scene = self.scenes[scene_id]
        inds = np.arange(start, start + self.n_frames, dtype=np.int64)

        images = np.stack([self._read_image(scene, int(i)) for i in inds]).astype(np.float32)
        poses = scene["poses"][inds].astype(np.float32)
        intrinsics = scene["intrinsics"][inds].astype(np.float32)

        h, w = images.shape[1:3]
        depths = np.full((len(inds), h, w), self.placeholder_depth, dtype=np.float32)

        images = torch.from_numpy(images).float().permute(0, 3, 1, 2)
        disps = torch.from_numpy(1.0 / depths).float()
        poses = torch.from_numpy(poses).float()
        intrinsics = torch.from_numpy(intrinsics).float()

        if self.aug is not None:
            images, poses, disps, intrinsics = self.aug(images, poses, disps, intrinsics)

        if self.use_imu:
            imu_delta, imu_valid, imu_info = self._build_imu_sequence(scene, inds)
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

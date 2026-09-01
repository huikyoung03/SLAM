import glob
import os.path as osp

import cv2
import numpy as np
import torch
import torch.utils.data as data

from .augmentation import RGBDAugmentor
from .base import RGBDDataset


class TartanAirV2(data.Dataset):
    """TartanAir V2 reader for DROID-SLAM training.

    This reader targets the V2 layout downloaded by the official `tartanair`
    package:

        root/EnvName/Data_easy/P000/image_lcam_front/*.png
        root/EnvName/Data_easy/P000/depth_lcam_front/*.png
        root/EnvName/Data_easy/P000/pose_lcam_front.txt
        root/EnvName/Data_easy/P000/imu/*.txt

    It keeps the original DROID TartanAir pose/depth scaling convention so the
    existing visual losses remain comparable to the V1 loader.
    """

    DEPTH_SCALE = 5.0
    has_dense_depth = True

    def __init__(
        self,
        datapath,
        n_frames=7,
        crop_size=[384, 512],
        do_aug=True,
        use_imu=False,
        imu_prior_name="imu_prior.csv",
        imu_require=False,
        camera_name="lcam_front",
        difficulties=None,
        **kwargs,
    ):
        self.root = datapath
        self.n_frames = int(n_frames)
        self.use_imu = bool(use_imu)
        self.imu_prior_name = imu_prior_name
        self.imu_require = bool(imu_require)
        self.camera_name = camera_name
        self.difficulties = difficulties
        self.aug = RGBDAugmentor(crop_size=crop_size) if do_aug else None

        self.scenes = self._build_scenes()
        self.dataset_index = []
        for scene_id, scene in enumerate(self.scenes):
            count = len(scene["images"])
            for start in range(0, max(count - self.n_frames + 1, 0)):
                self.dataset_index.append((scene_id, start))

    @staticmethod
    def calib_read():
        # TartanAir V2 pinhole cameras are 640x640 with 90 degree FOV.
        return np.array([320.0, 320.0, 320.0, 320.0], dtype=np.float32)

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth_rgba = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        if depth_rgba is None:
            raise FileNotFoundError(depth_file)

        depth = depth_rgba.view("<f4").squeeze(axis=-1).astype(np.float32)
        depth = depth / TartanAirV2.DEPTH_SCALE
        depth[~np.isfinite(depth)] = 1.0
        depth[depth <= 0.0] = 1.0
        return depth

    def _discover_trajectories(self):
        trajs = sorted(glob.glob(osp.join(self.root, "*", "Data_*", "P*")))

        if self.difficulties:
            allowed = {f"Data_{difficulty}" for difficulty in self.difficulties}
            trajs = [traj for traj in trajs if osp.basename(osp.dirname(traj)) in allowed]

        image_dir = f"image_{self.camera_name}"
        depth_dir = f"depth_{self.camera_name}"
        pose_file = f"pose_{self.camera_name}.txt"

        out = []
        for traj in trajs:
            if (
                osp.isdir(osp.join(traj, image_dir))
                and osp.isdir(osp.join(traj, depth_dir))
                and osp.isfile(osp.join(traj, pose_file))
            ):
                out.append(traj)

        return out

    def _build_scenes(self):
        trajs = self._discover_trajectories()
        if not trajs:
            raise FileNotFoundError(
                f"TartanAir V2 trajectories not found under {self.root} for camera={self.camera_name}"
            )

        scenes = []
        for traj in trajs:
            scene = self._load_scene(traj)
            if len(scene["images"]) >= self.n_frames:
                scenes.append(scene)

        if not scenes:
            raise ValueError(f"no TartanAir V2 trajectory has at least {self.n_frames} frames")

        total_frames = sum(len(scene["images"]) for scene in scenes)
        print(
            "Building TartanAir V2 dataset: "
            f"trajectories={len(scenes)}, frames={total_frames}, camera={self.camera_name}"
        )
        return scenes

    def _load_scene(self, traj):
        image_dir = osp.join(traj, f"image_{self.camera_name}")
        depth_dir = osp.join(traj, f"depth_{self.camera_name}")
        pose_file = osp.join(traj, f"pose_{self.camera_name}.txt")

        images = sorted(glob.glob(osp.join(image_dir, "*.png")))
        depths = sorted(glob.glob(osp.join(depth_dir, "*.png")))
        poses = np.loadtxt(pose_file, delimiter=" ").astype(np.float32)

        count = min(len(images), len(depths), len(poses))
        images = images[:count]
        depths = depths[:count]
        poses = poses[:count]

        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
        poses[:, :3] /= self.DEPTH_SCALE
        intrinsics = np.tile(self.calib_read()[None], (count, 1))

        imu_priors = self._build_imu_priors(traj, count) if self.use_imu else None

        return {
            "id": traj,
            "images": images,
            "depths": depths,
            "poses": poses,
            "intrinsics": intrinsics.astype(np.float32),
            "imu_priors": imu_priors,
        }

    def _build_imu_priors(self, traj, num_frames):
        imu_dir = osp.join(traj, "imu")
        required = [
            osp.join(imu_dir, "cam_time.txt"),
            osp.join(imu_dir, "imu_time.txt"),
            osp.join(imu_dir, "gyro.txt"),
        ]
        if not all(osp.isfile(path) for path in required):
            if self.imu_require:
                missing = [path for path in required if not osp.isfile(path)]
                raise FileNotFoundError(f"missing TartanAir V2 IMU files: {missing}")
            return None

        from tools.imu_preintegrate import (
            ImuCalibration,
            compute_imu_weight,
            estimate_preintegration_uncertainty,
            integrate_imu_window,
            select_imu_window,
        )

        cam_time = np.loadtxt(osp.join(imu_dir, "cam_time.txt"), dtype=np.float64)[:num_frames]
        imu_time = np.loadtxt(osp.join(imu_dir, "imu_time.txt"), dtype=np.float64)
        gyro = np.loadtxt(osp.join(imu_dir, "gyro.txt"), dtype=np.float64)

        acc_path = osp.join(imu_dir, "acc_nograv_body.txt")
        if not osp.isfile(acc_path):
            acc_path = osp.join(imu_dir, "acc.txt")
        acc = np.loadtxt(acc_path, dtype=np.float64)

        count = min(len(imu_time), len(gyro), len(acc))
        imu_time = imu_time[:count]
        gyro = gyro[:count]
        acc = acc[:count]

        imu = []
        for t, g, a in zip(imu_time, gyro, acc):
            timestamp_ns = int(round(float(t) * 1_000_000_000))
            imu.append({
                "timestamp_sec": float(t),
                "timestamp_ns": timestamp_ns,
                "gx": float(g[0]),
                "gy": float(g[1]),
                "gz": float(g[2]),
                "ax": float(a[0]),
                "ay": float(a[1]),
                "az": float(a[2]),
            })

        imu_timestamps = [int(sample["timestamp_ns"]) for sample in imu]
        calibration = ImuCalibration(source="tartanair_v2_identity")
        priors = {}

        previous_ns = None
        for frame_index, t in enumerate(cam_time):
            current_ns = int(round(float(t) * 1_000_000_000))
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

            priors[frame_index] = {
                "dt": float(integ["dt"]),
                "dr_x": float(integ["dr_x"]),
                "dr_y": float(integ["dr_y"]),
                "dr_z": float(integ["dr_z"]),
                "dq_w": float(integ["dq_w"]),
                "dq_x": float(integ["dq_x"]),
                "dq_y": float(integ["dq_y"]),
                "dq_z": float(integ["dq_z"]),
                "dv_x": float(integ["dv_x"]) / self.DEPTH_SCALE,
                "dv_y": float(integ["dv_y"]) / self.DEPTH_SCALE,
                "dv_z": float(integ["dv_z"]) / self.DEPTH_SCALE,
                "dp_x": float(integ["dp_x"]) / self.DEPTH_SCALE,
                "dp_y": float(integ["dp_y"]) / self.DEPTH_SCALE,
                "dp_z": float(integ["dp_z"]) / self.DEPTH_SCALE,
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

    def _build_imu_sequence(self, scene, inds):
        imu_delta, imu_valid, imu_info = RGBDDataset._empty_imu_sequence(len(inds))
        priors = scene.get("imu_priors")
        if priors is None:
            return imu_delta, imu_valid, imu_info

        for k in range(1, len(inds)):
            delta, valid, info = RGBDDataset._compose_imu_interval(
                priors,
                int(inds[k - 1]),
                int(inds[k]),
            )
            imu_delta[k] = delta
            imu_valid[k] = valid
            imu_info[k] = info

        return imu_delta, imu_valid, imu_info

    def __getitem__(self, index):
        scene_id, start = self.dataset_index[index % len(self.dataset_index)]
        scene = self.scenes[scene_id]
        inds = np.arange(start, start + self.n_frames, dtype=np.int64)

        images, depths, poses, intrinsics = [], [], [], []
        for i in inds:
            images.append(self.image_read(scene["images"][int(i)]))
            depths.append(self.depth_read(scene["depths"][int(i)]))
            poses.append(scene["poses"][int(i)])
            intrinsics.append(scene["intrinsics"][int(i)])

        if self.use_imu:
            imu_delta, imu_valid, imu_info = self._build_imu_sequence(scene, inds)

        images = torch.from_numpy(np.stack(images).astype(np.float32)).float()
        images = images.permute(0, 3, 1, 2)
        disps = torch.from_numpy(1.0 / np.stack(depths).astype(np.float32)).float()
        poses = torch.from_numpy(np.stack(poses).astype(np.float32)).float()
        intrinsics = torch.from_numpy(np.stack(intrinsics).astype(np.float32)).float()

        if self.aug is not None:
            images, poses, disps, intrinsics = self.aug(images, poses, disps, intrinsics)

        if len(disps[disps > 0.01]) > 0:
            scale = disps[disps > 0.01].mean()
            disps = disps / scale
            poses[..., :3] *= scale
            if self.use_imu:
                imu_delta = imu_delta.copy()
                imu_delta[:, 4:10] *= float(scale.detach().cpu().item())

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

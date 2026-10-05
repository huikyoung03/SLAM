import sys
sys.path.append("droid_slam")

import argparse
import torch
import open3d as o3d
import droid_backends

from lietorch import SE3
from cuda_timer import CudaTimer


def _scalar(blob, key, default=0.0):
    value = blob.get(key, default)
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1)
        if value.numel() == 0:
            return float(default)
        value = value[0].item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _metricize_if_aligned(reconstruction_blob, poses, disps, enabled):
    """Convert a VI-aligned, normalized reconstruction back to metres."""

    aligned = _scalar(reconstruction_blob, "imu_alignment_ready", 0.0) > 0.5
    unit_scale = _scalar(
        reconstruction_blob,
        "pose_translation_unit_scale",
        _scalar(reconstruction_blob, "imu_unit_scale", 0.0),
    )
    if not enabled or not aligned or not torch.isfinite(torch.tensor(unit_scale)) or unit_scale <= 1e-12:
        return poses, disps, False

    # Pose translation has DROID-unit/metre scale U. Inverse depth has the
    # reciprocal unit, so p <- p/U and disparity <- disparity*U.
    poses = poses.clone()
    poses[:, :3] /= unit_scale
    disps = disps * unit_scale
    return poses, disps, True


def _common_frame_indices(reconstruction_blob, other_filename):
    """Return one index per frame ID shared with another reconstruction."""

    if "tstamps" not in reconstruction_blob:
        raise KeyError("reconstruction is missing tstamps required for common-frame export")

    other_blob = torch.load(other_filename, map_location="cpu")
    if "tstamps" not in other_blob:
        raise KeyError(
            f"comparison reconstruction is missing tstamps: {other_filename}"
        )

    other_ids = {
        int(round(float(timestamp)))
        for timestamp in other_blob["tstamps"].detach().cpu().reshape(-1)
    }
    indices = []
    seen = set()
    for index, timestamp in enumerate(
        reconstruction_blob["tstamps"].detach().cpu().reshape(-1)
    ):
        frame_id = int(round(float(timestamp)))
        if frame_id in other_ids and frame_id not in seen:
            indices.append(index)
            seen.add(frame_id)

    if not indices:
        raise ValueError(
            "reconstructions have no shared frame IDs; cannot export a common PLY"
        )
    return indices, len(seen)


def export_ply(
    filename,
    output,
    filter_thresh=0.0,
    filter_count=0,
    metricize_imu=True,
    common_with=None,
):
    reconstruction_blob = torch.load(filename, map_location="cpu")

    indices = None
    if common_with:
        indices, common_count = _common_frame_indices(reconstruction_blob, common_with)
        print(f"[PLY] retaining {common_count} frame IDs shared with {common_with}")

    images = reconstruction_blob["images"]
    disps = reconstruction_blob["disps"]
    poses = reconstruction_blob["poses"]
    intrinsics = reconstruction_blob["intrinsics"]
    if indices is not None:
        images = images[indices]
        disps = disps[indices]
        poses = poses[indices]
        intrinsics = intrinsics[indices]

    images = images.cuda()[..., ::2, ::2]
    disps = disps.cuda()[..., ::2, ::2]
    poses = poses.cuda()
    intrinsics = 4 * intrinsics.cuda()
    poses, disps, metricized = _metricize_if_aligned(
        reconstruction_blob,
        poses,
        disps,
        metricize_imu,
    )
    if metricized:
        print("[PLY] exported in metres using IMU visual-inertial alignment")

    disps = disps.contiguous()
    index = torch.arange(len(images), device="cuda")
    thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))

    with CudaTimer("iproj"):
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
        colors = images[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.0

    with CudaTimer("filter"):
        counts = droid_backends.depth_filter(poses, disps, intrinsics[0], index, thresh)
        mask = (counts >= filter_count) & (disps > 0.25 * disps.mean())

    points_np = points[mask].detach().cpu().numpy()
    colors_np = colors[mask].detach().cpu().numpy()

    print("points:", points_np.shape)
    print("colors:", colors_np.shape)

    if points_np.shape[0] == 0:
        print("[ERROR] no points to save")
        return

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_np)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_np)

    ok = o3d.io.write_point_cloud(output, point_cloud)
    print("saved:", output)
    print("write ok:", ok)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--output", required=True)
    parser.add_argument("--filter_threshold", type=float, default=0.0)
    parser.add_argument("--filter_count", type=int, default=0)
    parser.add_argument(
        "--common-with",
        default=None,
        help="export only frame IDs shared with this reconstruction .pth file",
    )
    parser.add_argument(
        "--raw-units",
        dest="metricize_imu",
        action="store_false",
        help="do not convert a VI-aligned reconstruction from DROID units to metres",
    )
    parser.set_defaults(metricize_imu=True)
    args = parser.parse_args()

    export_ply(
        args.filename,
        args.output,
        args.filter_threshold,
        args.filter_count,
        args.metricize_imu,
        args.common_with,
    )

import sys
sys.path.append("droid_slam")

import argparse
import torch
import open3d as o3d
import droid_backends

from lietorch import SE3
from cuda_timer import CudaTimer


def export_ply(filename, output, filter_thresh=0.0, filter_count=0):
    reconstruction_blob = torch.load(filename)

    images = reconstruction_blob["images"].cuda()[..., ::2, ::2]
    disps = reconstruction_blob["disps"].cuda()[..., ::2, ::2]
    poses = reconstruction_blob["poses"].cuda()
    intrinsics = 4 * reconstruction_blob["intrinsics"].cuda()

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
    args = parser.parse_args()

    export_ply(
        args.filename,
        args.output,
        args.filter_threshold,
        args.filter_count
    )

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt


def extract_poses(obj):
    """
    DROID-SLAM reconstruction pth에서 poses를 최대한 robust하게 추출
    """
    if isinstance(obj, dict):
        for key in ["poses", "traj", "trajectory"]:
            if key in obj:
                return obj[key]

        if "video" in obj:
            video = obj["video"]
            if hasattr(video, "poses"):
                return video.poses
            if isinstance(video, dict) and "poses" in video:
                return video["poses"]

    if hasattr(obj, "poses"):
        return obj.poses

    if hasattr(obj, "video") and hasattr(obj.video, "poses"):
        return obj.video.poses

    raise ValueError("poses를 찾지 못했습니다. pth 구조를 확인해야 합니다.")


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Trajectory top view")
    args = parser.parse_args()

    plt.figure(figsize=(8, 6))

    for idx, file in enumerate(args.files):
        path = Path(file)
        obj = torch.load(path, map_location="cpu")
        poses = to_numpy(extract_poses(obj))

        # poses: [N, 7] = tx, ty, tz, qx, qy, qz, qw 형태라고 가정
        if poses.ndim != 2 or poses.shape[1] < 3:
            raise ValueError(f"pose shape 이상함: {path}, shape={poses.shape}")

        xyz = poses[:, :3]

        # invalid / nan 제거
        mask = np.isfinite(xyz).all(axis=1)
        xyz = xyz[mask]

        # top-view: x-z 평면 사용
        x = xyz[:, 0]
        z = xyz[:, 2]

        label = args.labels[idx] if args.labels and idx < len(args.labels) else path.stem

        plt.plot(x, z, marker="o", markersize=2, linewidth=1.5, label=label)

        # start/end 표시
        if len(x) > 0:
            plt.scatter(x[0], z[0], marker="s", s=50)
            plt.scatter(x[-1], z[-1], marker="x", s=60)

    plt.title(args.title)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

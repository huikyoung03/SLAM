import argparse
from pathlib import Path

import numpy as np
import torch


def q_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0, 0, 0, 1], dtype=np.float64)
    return q / n


def q_inv(q):
    q = q_normalize(q)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def q_mul(q1, q2):
    x1, y1, z1, w1 = q_normalize(q1)
    x2, y2, z2, w2 = q_normalize(q2)
    return q_normalize(np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float64))


def q_angle_deg(q):
    q = q_normalize(q)
    w = abs(float(q[3]))
    w = np.clip(w, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(w))


def load_poses(path):
    blob = torch.load(path, map_location="cpu")

    if isinstance(blob, dict):
        for key in ["poses", "traj", "trajectory"]:
            if key in blob:
                poses = blob[key]
                break
        else:
            raise KeyError(f"Cannot find poses in {path}. keys={list(blob.keys())}")
    else:
        poses = blob

    if torch.is_tensor(poses):
        poses = poses.detach().cpu().numpy()
    else:
        poses = np.asarray(poses)

    poses = poses.astype(np.float64)

    # expected: [tx, ty, tz, qx, qy, qz, qw]
    if poses.ndim != 2 or poses.shape[1] < 7:
        raise ValueError(f"Invalid pose shape {poses.shape} in {path}")

    return poses[:, :7]


def line_deviation(points):
    """
    Fit 3D line by PCA and compute distance of each pose position to the line.
    Useful for mostly straight-walk data.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return np.nan, np.nan

    center = points.mean(axis=0)
    X = points - center

    _, _, vh = np.linalg.svd(X, full_matrices=False)
    direction = vh[0]
    proj = np.outer(X @ direction, direction)
    residual = X - proj
    dist = np.linalg.norm(residual, axis=1)

    return float(np.median(dist)), float(np.percentile(dist, 95))


def metrics(path):
    poses = load_poses(path)
    t = poses[:, :3]
    q = poses[:, 3:7]

    if len(poses) < 2:
        return None

    steps = np.linalg.norm(np.diff(t, axis=0), axis=1)
    path_len = float(np.sum(steps))
    start_end = float(np.linalg.norm(t[-1] - t[0]))

    rots = []
    for i in range(1, len(q)):
        dq = q_mul(q_inv(q[i-1]), q[i])
        rots.append(q_angle_deg(dq))
    rots = np.asarray(rots)

    # step smoothness: second difference of positions
    if len(t) >= 3:
        accel = np.linalg.norm(t[2:] - 2*t[1:-1] + t[:-2], axis=1)
        accel_p95 = float(np.percentile(accel, 95))
    else:
        accel_p95 = np.nan

    line_med, line_p95 = line_deviation(t)

    return {
        "file": Path(path).name,
        "n_poses": len(poses),
        "path_len": path_len,
        "start_end/path": start_end / max(path_len, 1e-12),
        "step_median": float(np.median(steps)),
        "step_p95": float(np.percentile(steps, 95)),
        "step_max": float(np.max(steps)),
        "accel_p95": accel_p95,
        "rot_median_deg": float(np.median(rots)),
        "rot_p95_deg": float(np.percentile(rots, 95)),
        "rot_max_deg": float(np.max(rots)),
        "line_dev_median": line_med,
        "line_dev_p95": line_p95,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    rows = []
    for f in args.files:
        rows.append(metrics(f))

    cols = [
        "file",
        "n_poses",
        "path_len",
        "start_end/path",
        "step_p95",
        "accel_p95",
        "rot_p95_deg",
        "line_dev_median",
        "line_dev_p95",
    ]

    print("\n=== Non-loop trajectory metrics ===")
    print("Lower is generally better for: step_p95, accel_p95, rot_p95_deg, line_dev_median, line_dev_p95")
    print("start_end/path is NOT always lower-better for non-loop data.")
    print()

    print(" | ".join(cols))
    print("-" * 140)

    for r in rows:
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, str):
                vals.append(v)
            elif isinstance(v, int):
                vals.append(str(v))
            else:
                vals.append(f"{v:.6f}")
        print(" | ".join(vals))


if __name__ == "__main__":
    main()

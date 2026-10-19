#!/usr/bin/env python3
"""Plot a metric EuRoC reconstruction against time-matched ground truth.

The plot uses only a rigid SE(3) alignment when the reconstruction has a
valid VI metric scale. It therefore exposes scale/translation error instead
of hiding it with a Sim(3) scale fit.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluate_euroc_recon import (
    apply_alignment,
    body_to_camera_ground_truth,
    estimate_camera_poses,
    load_recon,
    nearest_gt_indices,
    read_camera_to_body_transform,
    read_frames,
    read_gt_poses,
    similarity_alignment,
)


def rmse(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values))) if len(values) else float("nan")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a VI-metric DROID reconstruction against EuRoC GT."
    )
    parser.add_argument("--recon", required=True)
    parser.add_argument("--frames-csv", required=True)
    parser.add_argument("--gt-csv", required=True)
    parser.add_argument("--camera-sensor-yaml", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="EuRoC trajectory: learned VI vs GT")
    parser.add_argument("--max-time-diff-ms", type=float, default=30.0)
    args = parser.parse_args()

    frame_times = read_frames(args.frames_csv)
    gt_ts, gt_positions, gt_rotations = read_gt_poses(args.gt_csv)
    camera_to_body = read_camera_to_body_transform(args.camera_sensor_yaml)
    gt_positions, gt_rotations = body_to_camera_ground_truth(
        gt_positions, gt_rotations, camera_to_body
    )

    tstamps, poses, metadata = load_recon(args.recon, auto_metricize_imu=True)
    if not metadata["metric_scale_available"]:
        raise ValueError(
            "reconstruction has no metric VI scale; refuse Sim(3) plotting for this diagnostic"
        )

    recon_indices, gt_indices, _ = nearest_gt_indices(
        tstamps,
        frame_times,
        gt_ts,
        int(args.max_time_diff_ms * 1e6),
    )
    if len(recon_indices) < 5:
        raise ValueError(f"only {len(recon_indices)} GT matches; need at least 5")

    estimated_positions, estimated_rotations, _ = estimate_camera_poses(
        poses[recon_indices], "inv_center"
    )
    target_positions = gt_positions[gt_indices]
    target_rotations = gt_rotations[gt_indices]

    # Rigid transform only: learned scale is intentionally left unchanged.
    R, translation, _ = similarity_alignment(
        estimated_positions, target_positions, correct_scale=False
    )
    aligned_positions, aligned_rotations = apply_alignment(
        estimated_positions, estimated_rotations, R, translation
    )
    errors = np.linalg.norm(aligned_positions - target_positions, axis=1)
    timestamps = np.asarray(
        [frame_times[int(round(float(frame_id)))] / 1e9 for frame_id in tstamps[recon_indices]],
        dtype=np.float64,
    )
    elapsed = timestamps - timestamps[0]

    segments = np.array_split(np.arange(len(errors)), 4)
    print(f"[GT-PLOT] matches={len(errors)}, metric_scale={metadata['pose_unit_scale']:.6f}")
    print(f"[GT-PLOT] SE3 ATE RMSE={rmse(errors):.6f} m, median={np.median(errors):.6f} m, p90={np.percentile(errors, 90):.6f} m")
    for number, indices in enumerate(segments, start=1):
        print(
            f"[GT-PLOT] quarter={number}, t={elapsed[indices[0]]:.1f}-{elapsed[indices[-1]]:.1f}s, "
            f"ATE_RMSE={rmse(errors[indices]):.6f} m"
        )

    figure, (axis_traj, axis_error) = plt.subplots(1, 2, figsize=(13, 5.5))
    axis_traj.plot(target_positions[:, 0], target_positions[:, 2], "k-", linewidth=2, label="GT")
    axis_traj.plot(
        aligned_positions[:, 0],
        aligned_positions[:, 2],
        color="#d95f02",
        linewidth=1.7,
        label="Estimate (SE(3)-aligned)",
    )
    axis_traj.scatter(target_positions[0, 0], target_positions[0, 2], marker="s", s=45, c="green", label="start")
    axis_traj.scatter(target_positions[-1, 0], target_positions[-1, 2], marker="x", s=55, c="red", label="end")
    axis_traj.set_title("Top view (metric scale retained)")
    axis_traj.set_xlabel("X [m]")
    axis_traj.set_ylabel("Z [m]")
    axis_traj.axis("equal")
    axis_traj.grid(True, alpha=0.3)
    axis_traj.legend(loc="best")

    axis_error.plot(elapsed, errors, color="#7570b3", linewidth=1.4)
    axis_error.axhline(rmse(errors), color="#d95f02", linestyle="--", label=f"RMSE = {rmse(errors):.3f} m")
    axis_error.set_title("Per-frame translational error")
    axis_error.set_xlabel("Time [s]")
    axis_error.set_ylabel("Position error [m]")
    axis_error.grid(True, alpha=0.3)
    axis_error.legend(loc="best")

    figure.suptitle(args.title)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    print(f"[GT-PLOT] saved: {output}")


if __name__ == "__main__":
    main()

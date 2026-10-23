#!/usr/bin/env python3
"""Regression check for final-full-BA confidence pose supervision."""

from collections import OrderedDict
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "droid_slam"))

from lietorch import SE3  # noqa: E402
from geom.losses import final_ba_pose_loss  # noqa: E402


def make_pose_data():
    # lietorch SE3 stores [tx, ty, tz, qx, qy, qz, qw].
    return torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )


def main():
    graph = OrderedDict({0: [1, 2], 1: [0, 2], 2: [0, 1]})
    gt = SE3(make_pose_data())

    exact = SE3(make_pose_data())
    loss, metrics = final_ba_pose_loss(gt, exact, graph, temporal_only=True)
    assert float(loss) < 1e-6
    # Four directed adjacent edges only; 0<->2 are graph edges but not IMU
    # factors and must not supervise scalar temporal confidence.
    assert metrics["post_ba_pose_edges"] == 4.0

    estimate_data = make_pose_data().requires_grad_(True)
    estimate_data.data[0, 1, 0] += 0.25
    estimate = SE3(estimate_data)
    loss, metrics = final_ba_pose_loss(gt, estimate, graph, temporal_only=True)
    assert float(loss) > 0.0
    assert metrics["post_ba_pose_translation"] > 0.0
    loss.backward()
    assert estimate_data.grad is not None
    assert torch.isfinite(estimate_data.grad).all()
    assert float(estimate_data.grad.abs().sum()) > 0.0
    print("final full-BA pose supervision checks passed")


if __name__ == "__main__":
    main()

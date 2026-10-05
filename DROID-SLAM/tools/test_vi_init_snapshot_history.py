#!/usr/bin/env python3
"""Deterministic checks for retained visual-inertial initialization windows."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "droid_slam"))

from vi_initialization import (
    VIAlignmentResult,
    capture_vi_initialization_snapshot,
    rebase_vi_alignment_result,
)


def make_video():
    count = 5
    poses = torch.zeros(count, 7)
    poses[:, 6] = 1.0
    poses[:, 0] = torch.arange(count, dtype=torch.float32)
    delta = torch.zeros(count, 10)
    delta[1:, 0] = 0.1
    valid = torch.ones(count, dtype=torch.bool)
    valid[0] = False
    return SimpleNamespace(
        poses=poses,
        imu_delta=delta,
        imu_valid=valid,
        imu_bias_jacobian=torch.ones(count, 5, 3, 3),
        imu_unit_scale=torch.tensor([2.0]),
        imu_camera_translation=torch.tensor([0.1, 0.0, 0.0]),
        tstamp=torch.tensor([0.0, 2.0, 3.0, 4.0, 5.0]),
        counter=SimpleNamespace(value=count),
    )


def main():
    video = make_video()
    snapshot = capture_vi_initialization_snapshot(video, window=3)
    assert snapshot is not None
    assert snapshot.counter.value == 3
    assert snapshot.tstamp.tolist() == [3.0, 4.0, 5.0]
    assert not bool(snapshot.imu_valid[0].item())
    assert float(snapshot.imu_delta[0].abs().sum().item()) == 0.0

    # The snapshot must survive subsequent live-buffer modification.
    video.poses[-1, 0] = 123.0
    assert float(snapshot.poses[-1, 0].item()) == 4.0

    result = VIAlignmentResult(
        success=True,
        reason="ok",
        scale=2.0,
        gravity_internal=torch.tensor([0.0, -19.62, 0.0]),
        velocities_internal=torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        velocity_indices=torch.tensor([0, 2]),
        accel_bias_metric=torch.tensor([0.01, 0.02, 0.03]),
        gyro_bias_metric=torch.tensor([0.001, 0.002, 0.003]),
        metrics={"gravity_norm_internal": 19.62, "residual_rms_internal": 0.5},
    )

    # Snapshot was captured at unit scale 2; the live state now uses scale 4.
    video.imu_unit_scale.fill_(4.0)
    rebased = rebase_vi_alignment_result(result, snapshot, video)
    assert abs(rebased.scale - 4.0) < 1e-6
    assert torch.allclose(rebased.gravity_internal, torch.tensor([0.0, -39.24, 0.0]))
    assert rebased.velocity_indices.tolist() == [2, 4]
    assert torch.allclose(
        rebased.velocities_internal,
        torch.tensor([[2.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
    )
    assert torch.allclose(rebased.accel_bias_metric, torch.tensor([0.01, 0.02, 0.03]))
    assert abs(rebased.metrics["gravity_norm_internal"] - 39.24) < 1e-6
    assert rebased.metrics["history_snapshot"] == 1.0

    print("VI initialization snapshot history checks passed")


if __name__ == "__main__":
    main()

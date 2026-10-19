#!/usr/bin/env python3
"""Deterministic checks for the training-side visual -> full-IMU schedule."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "droid_slam"))

from droid_net import full_imu_unroll_schedule  # noqa: E402
from train import select_full_imu_stage_outputs  # noqa: E402


def assert_unroll_schedule():
    assert full_imu_unroll_schedule(4, True, 2) == [False, False, True, True]
    assert full_imu_unroll_schedule(4, True, 0) == [True, True, True, True]
    assert full_imu_unroll_schedule(4, False, 2) == [False, False, False, False]


def assert_full_loss_uses_only_full_stage_outputs():
    poses = ["pose0", "pose1", "pose2", "pose3"]
    confidences = ["conf0", "conf1", "conf2", "conf3"]
    motions = ["motion2", "motion3"]

    stage_poses, stage_confidences, stage_motions = select_full_imu_stage_outputs(
        poses,
        confidences,
        motions,
        pose_only_steps=2,
    )
    assert stage_poses == ["pose2", "pose3"]
    assert stage_confidences == ["conf2", "conf3"]
    assert stage_motions == ["motion2", "motion3"]


if __name__ == "__main__":
    assert_unroll_schedule()
    assert_full_loss_uses_only_full_stage_outputs()
    print("training two-stage IMU schedule checks passed")

#!/usr/bin/env python3
"""CPU checks for absolute pre-BA IMU-confidence calibration."""

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "droid_slam"))

from geom.losses import imu_confidence_pose_prior_ranking_loss  # noqa: E402


def assert_absolute_pose_prior_calibration_has_gradients():
    # Four visual-only poses and deliberately non-uniform IMU priors create
    # distinct pre-BA disagreement energies. The two confidence paths are
    # independent leaves so the test proves both receive supervision.
    poses = torch.zeros(1, 4, 7)
    poses[..., 0] = torch.tensor([0.0, 0.01, 0.02, 0.03])
    poses[..., 6] = 1.0

    imu_delta = torch.zeros(1, 4, 10)
    imu_delta[:, 1:, 0] = 0.1
    imu_delta[:, 1, 7] = 0.01
    imu_delta[:, 2, 7] = 0.04
    imu_delta[:, 3, 7] = 0.10
    imu_valid = torch.ones(1, 4, dtype=torch.bool)

    confidence = torch.tensor(
        [[0.5, 0.55, 0.60, 0.65]], requires_grad=True
    )
    quality_only = torch.tensor(
        [[0.5, 0.45, 0.40, 0.35]], requires_grad=True
    )
    reference = torch.tensor(float("nan"))

    loss, metrics = imu_confidence_pose_prior_ranking_loss(
        poses,
        confidence,
        imu_delta,
        imu_valid=imu_valid,
        gravity=torch.zeros(3),
        imu_optimize_bias=False,
        absolute_weight=1.0,
        energy_log_reference=reference,
        energy_reference_momentum=0.98,
        energy_temperature=0.75,
        quality_only_confidence=quality_only,
    )
    loss.backward()

    assert torch.isfinite(reference)
    assert metrics["imu_confidence_pose_prior_absolute"] > 0.0
    assert metrics["imu_confidence_pose_prior_quality_only_absolute"] > 0.0
    assert confidence.grad is not None and torch.count_nonzero(confidence.grad) > 0
    assert quality_only.grad is not None and torch.count_nonzero(quality_only.grad) > 0


def assert_nominal_clean_mode_anchors_only_clean_intervals():
    poses = torch.zeros(1, 4, 7)
    poses[..., 0] = torch.tensor([0.0, 0.01, 0.02, 0.03])
    poses[..., 6] = 1.0
    imu_delta = torch.zeros(1, 4, 10)
    imu_delta[:, 1:, 0] = 0.1
    imu_delta[:, 1:, 7] = torch.tensor([0.01, 0.04, 0.10])
    confidence = torch.full((1, 4), 0.5, requires_grad=True)
    augmented = torch.tensor([[False, False, True, False]])

    loss, metrics = imu_confidence_pose_prior_ranking_loss(
        poses,
        confidence,
        imu_delta,
        imu_valid=torch.ones(1, 4, dtype=torch.bool),
        gravity=torch.zeros(3),
        imu_optimize_bias=False,
        absolute_weight=1.0,
        augmented_mask=augmented,
        absolute_target_mode="nominal_clean",
    )
    loss.backward()

    assert metrics["imu_confidence_pose_prior_target_clean_mean"] == 1.0
    assert 0.0 < metrics["imu_confidence_pose_prior_target_augmented_mean"] < 1.0
    assert confidence.grad is not None and torch.count_nonzero(confidence.grad) > 0


def assert_split_mode_separates_raw_quality_and_fused_targets():
    poses = torch.zeros(1, 4, 7)
    poses[..., 0] = torch.tensor([0.0, 0.01, 0.02, 0.03])
    poses[..., 6] = 1.0
    imu_delta = torch.zeros(1, 4, 10)
    imu_delta[:, 1:, 0] = 0.1
    imu_delta[:, 1:, 7] = torch.tensor([0.01, 0.04, 0.10])
    confidence = torch.full((1, 4), 0.5, requires_grad=True)
    quality_only = torch.full((1, 4), 0.5, requires_grad=True)
    augmented = torch.tensor([[False, False, True, False]])

    loss, metrics = imu_confidence_pose_prior_ranking_loss(
        poses,
        confidence,
        imu_delta,
        imu_valid=torch.ones(1, 4, dtype=torch.bool),
        gravity=torch.zeros(3),
        imu_optimize_bias=False,
        absolute_weight=1.0,
        augmented_mask=augmented,
        absolute_target_mode="split_nominal_quality",
        quality_only_confidence=quality_only,
    )
    loss.backward()

    assert metrics["imu_confidence_pose_prior_quality_target_clean_mean"] == 1.0
    assert metrics["imu_confidence_pose_prior_target_clean_mean"] < 1.0
    assert (
        metrics["imu_confidence_pose_prior_quality_target_augmented_mean"]
        == metrics["imu_confidence_pose_prior_target_augmented_mean"]
    )
    assert confidence.grad is not None and torch.count_nonzero(confidence.grad) > 0
    assert quality_only.grad is not None and torch.count_nonzero(quality_only.grad) > 0


def assert_one_sided_mode_preserves_normal_clean_imu():
    poses = torch.zeros(1, 4, 7)
    poses[..., 0] = torch.tensor([0.0, 0.01, 0.02, 0.03])
    poses[..., 6] = 1.0
    imu_delta = torch.zeros(1, 4, 10)
    imu_delta[:, 1:, 0] = 0.1
    imu_delta[:, 1:, 7] = torch.tensor([0.01, 0.04, 0.10])
    confidence = torch.full((1, 4), 0.5, requires_grad=True)
    quality_only = torch.full((1, 4), 0.5, requires_grad=True)
    augmented = torch.tensor([[False, False, True, False]])

    loss, metrics = imu_confidence_pose_prior_ranking_loss(
        poses,
        confidence,
        imu_delta,
        imu_valid=torch.ones(1, 4, dtype=torch.bool),
        gravity=torch.zeros(3),
        imu_optimize_bias=False,
        absolute_weight=1.0,
        augmented_mask=augmented,
        absolute_target_mode="one_sided_nominal_quality",
        quality_only_confidence=quality_only,
    )
    loss.backward()

    assert metrics["imu_confidence_pose_prior_quality_target_clean_mean"] == 1.0
    assert metrics["imu_confidence_pose_prior_target_clean_mean"] > 0.9
    assert 0.0 < metrics["imu_confidence_pose_prior_target_augmented_mean"] < 1.0
    assert confidence.grad is not None and torch.count_nonzero(confidence.grad) > 0
    assert quality_only.grad is not None and torch.count_nonzero(quality_only.grad) > 0


if __name__ == "__main__":
    assert_absolute_pose_prior_calibration_has_gradients()
    assert_nominal_clean_mode_anchors_only_clean_intervals()
    assert_split_mode_separates_raw_quality_and_fused_targets()
    assert_one_sided_mode_preserves_normal_clean_imu()
    print("absolute pre-BA confidence calibration checks passed")

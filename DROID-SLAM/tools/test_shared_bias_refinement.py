#!/usr/bin/env python3
"""Deterministic checks for the 9DoF plus shared-bias local optimizer."""

from pathlib import Path
from types import SimpleNamespace
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "droid_slam"))

from geom.imu_factor import rotvec_to_quat  # noqa: E402
from vi_initialization import (  # noqa: E402
    _full_bias_residual_and_jacobian,
    refine_window_biases,
)


class MockVideo:
    """Minimal DepthVideo-compatible state for IMU factor tests."""

    def __init__(self, frames=12, dtype=torch.float64):
        self.counter = SimpleNamespace(value=frames)
        self.imu_alignment_ready = SimpleNamespace(value=1)
        self.poses = torch.zeros(frames, 7, dtype=dtype)
        self.poses[:, 6] = 1.0
        self.velocities = torch.zeros(frames, 3, dtype=dtype)
        self.imu_delta = torch.zeros(frames, 10, dtype=dtype)
        self.imu_valid = torch.ones(frames, dtype=torch.bool)
        self.imu_weight = torch.ones(frames, dtype=dtype)
        self.imu_info = torch.ones(frames, 3, dtype=dtype)
        self.imu_bias_random_walk = torch.zeros(frames, 2, dtype=dtype)
        self.imu_bias_jacobian = torch.zeros(frames, 5, 3, 3, dtype=dtype)
        self.imu_global_bias_acc = torch.zeros(3, dtype=dtype)
        self.imu_global_bias_gyro = torch.zeros(3, dtype=dtype)
        self.imu_gravity = torch.zeros(3, dtype=dtype)
        self._lever_arm = torch.tensor([0.02, -0.01, 0.03], dtype=dtype)

    def imu_camera_translation_in_current_units(self):
        return self._lever_arm


def make_random_video():
    torch.manual_seed(7)
    video = MockVideo(frames=7)
    video.poses[:, :3] = torch.randn(7, 3, dtype=torch.float64) * 0.1
    video.poses[:, 3:7] = rotvec_to_quat(
        torch.randn(7, 3, dtype=torch.float64) * 0.15
    )
    video.velocities[:] = torch.randn(7, 3, dtype=torch.float64) * 0.1
    video.imu_delta[1:, 0] = 0.1
    video.imu_delta[1:, 1:10] = torch.randn(6, 9, dtype=torch.float64) * 0.01
    video.imu_bias_jacobian[:] = torch.randn(7, 5, 3, 3, dtype=torch.float64) * 0.01
    video.imu_bias_jacobian[1:, 0] += -0.1 * torch.eye(3, dtype=torch.float64)
    video.imu_gravity[:] = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
    return video


def assert_bias_jacobian_matches_finite_difference():
    video = make_random_video()
    src = torch.arange(0, 6)
    dst = src + 1
    accel_bias = torch.tensor([0.007, -0.009, 0.004], dtype=torch.float64)
    gyro_bias = torch.tensor([0.002, 0.003, -0.004], dtype=torch.float64)
    epsilon = 1e-5

    for compose_order in ("dq_prev", "prev_dq"):
        for inverse_imu in (False, True):
            _, jacobian = _full_bias_residual_and_jacobian(
                video,
                src,
                dst,
                accel_bias,
                gyro_bias,
                video.imu_gravity,
                compose_order,
                inverse_imu,
            )
            max_error = 0.0
            for axis in range(6):
                accel_delta = torch.zeros(3, dtype=torch.float64)
                gyro_delta = torch.zeros(3, dtype=torch.float64)
                if axis < 3:
                    accel_delta[axis] = epsilon
                else:
                    gyro_delta[axis - 3] = epsilon
                plus, _ = _full_bias_residual_and_jacobian(
                    video,
                    src,
                    dst,
                    accel_bias + accel_delta,
                    gyro_bias + gyro_delta,
                    video.imu_gravity,
                    compose_order,
                    inverse_imu,
                )
                minus, _ = _full_bias_residual_and_jacobian(
                    video,
                    src,
                    dst,
                    accel_bias - accel_delta,
                    gyro_bias - gyro_delta,
                    video.imu_gravity,
                    compose_order,
                    inverse_imu,
                )
                numerical_derivative = (plus - minus) / (2.0 * epsilon)
                # The optimizer stores the negative derivative in r - J db.
                max_error = max(
                    max_error,
                    float((numerical_derivative + jacobian[:, :, axis]).abs().max()),
                )
            assert max_error < 1e-6, (
                f"bias Jacobian mismatch for {compose_order}, inverse={inverse_imu}: "
                f"{max_error:.3e}"
            )


def make_consistent_bias_video():
    video = MockVideo(frames=12)
    video._lever_arm.zero_()
    video.true_accel_bias = torch.tensor([0.03, -0.02, 0.01], dtype=torch.float64)
    video.true_gyro_bias = torch.tensor([0.01, -0.015, 0.02], dtype=torch.float64)
    dt = 0.1
    eye = torch.eye(3, dtype=torch.float64)
    for frame in range(1, video.counter.value):
        video.imu_delta[frame, 0] = dt
        # The visual state is static. These raw deltas are exactly cancelled
        # by the known shared bias under dq_prev without inversion.
        video.imu_delta[frame, 1:4] = dt * video.true_gyro_bias
        video.imu_delta[frame, 4:7] = dt * video.true_accel_bias
        video.imu_delta[frame, 7:10] = 0.5 * dt * dt * video.true_accel_bias
        video.imu_bias_jacobian[frame, 0] = -dt * eye
        video.imu_bias_jacobian[frame, 1] = -dt * eye
        video.imu_bias_jacobian[frame, 3] = -0.5 * dt * dt * eye
    return video


def assert_shared_bias_recovery_and_gate():
    video = make_consistent_bias_video()
    result = refine_window_biases(
        video,
        compose_order="dq_prev",
        inverse_imu=False,
        window=12,
        max_dt=0.0,
        pos_weight=1.0,
        vel_weight=1.0,
        damping=1e-8,
        step_scale=1.0,
        max_accel_step=1.0,
        max_gyro_step=1.0,
        max_accel_norm=1.0,
        max_gyro_norm=1.0,
        global_weight=1.0,
        use_imu_info_weighting=True,
        robust_delta=0.0,
        min_edges=3,
        min_duration=0.0,
        min_rotation_deg=0.0,
        max_condition=1e12,
        min_relative_improvement=1e-6,
        accel_prior_std=1e6,
        gyro_prior_std=1e6,
    )
    assert result.get("updated", 0.0) == 1.0, result
    assert torch.allclose(
        video.imu_global_bias_acc, video.true_accel_bias, atol=1e-8, rtol=1e-8
    )
    assert torch.allclose(
        video.imu_global_bias_gyro, video.true_gyro_bias, atol=1e-8, rtol=1e-8
    )

    # A strictly lower but numerically insignificant local objective must not
    # be allowed to accumulate a random-walk bias over many frontend updates.
    insignificant = make_consistent_bias_video()
    rejected_small_step = refine_window_biases(
        insignificant,
        compose_order="dq_prev",
        inverse_imu=False,
        window=12,
        max_dt=0.0,
        pos_weight=1.0,
        vel_weight=1.0,
        damping=1e-8,
        step_scale=1e-8,
        max_accel_step=1.0,
        max_gyro_step=1.0,
        max_accel_norm=1.0,
        max_gyro_norm=1.0,
        global_weight=1.0,
        use_imu_info_weighting=True,
        robust_delta=0.0,
        min_edges=3,
        min_duration=0.0,
        min_rotation_deg=0.0,
        max_condition=1e12,
        min_relative_improvement=1e-6,
        accel_prior_std=1e6,
        gyro_prior_std=1e6,
    )
    assert rejected_small_step.get("reason") == "insufficient_relative_improvement", rejected_small_step
    assert torch.count_nonzero(insignificant.imu_global_bias_acc) == 0
    assert torch.count_nonzero(insignificant.imu_global_bias_gyro) == 0

    no_excitation = make_consistent_bias_video()
    no_excitation.imu_delta[:, 1:4] = 0.0
    rejected = refine_window_biases(
        no_excitation,
        compose_order="dq_prev",
        inverse_imu=False,
        window=12,
        max_dt=0.0,
        global_weight=1.0,
        min_edges=3,
        min_duration=0.0,
        min_rotation_deg=5.0,
    )
    assert rejected.get("reason") == "insufficient_rotation", rejected


if __name__ == "__main__":
    assert_bias_jacobian_matches_finite_difference()
    assert_shared_bias_recovery_and_gate()
    print("shared-bias Jacobian and recovery checks passed")

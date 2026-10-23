#!/usr/bin/env python3
"""Deterministic checks for the post-global shared-bias 15DoF refinement."""

from pathlib import Path
from types import SimpleNamespace
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "droid_slam"))

from vi_initialization import (  # noqa: E402
    _full_bias_residual_and_jacobian,
    recenter_full_imu_biases,
    refine_global_bias_15dof,
)


class MockVideo:
    """Small CPU-only state matching the final global-bias helper contract."""

    def __init__(self, frames=12, dtype=torch.float64):
        self.counter = SimpleNamespace(value=frames)
        self.imu_alignment_ready = SimpleNamespace(value=1)
        self.poses = torch.zeros(frames, 7, dtype=dtype)
        self.poses[:, 6] = 1.0
        self.velocities = torch.zeros(frames, 3, dtype=dtype)
        self.bias_acc = torch.zeros(frames, 3, dtype=dtype)
        self.bias_gyro = torch.zeros(frames, 3, dtype=dtype)
        self.imu_delta = torch.zeros(frames, 10, dtype=dtype)
        self.imu_valid = torch.ones(frames, dtype=torch.bool)
        self.imu_weight = torch.ones(frames, dtype=dtype)
        self.imu_info = torch.ones(frames, 3, dtype=dtype)
        self.imu_bias_random_walk = torch.zeros(frames, 2, dtype=dtype)
        self.imu_bias_jacobian = torch.zeros(frames, 5, 3, 3, dtype=dtype)
        self.imu_global_bias_acc = torch.zeros(3, dtype=dtype)
        self.imu_global_bias_gyro = torch.zeros(3, dtype=dtype)
        self.imu_gravity = torch.zeros(3, dtype=dtype)
        self.dirty = torch.zeros(frames, dtype=torch.bool)
        self.imu_bias_relinearization_required = False

    def imu_camera_translation_in_current_units(self):
        return torch.zeros(3, dtype=self.poses.dtype)


def make_consistent_video():
    video = MockVideo()
    video.true_accel_bias = torch.tensor([0.03, -0.02, 0.01], dtype=torch.float64)
    video.true_gyro_bias = torch.tensor([0.01, -0.015, 0.02], dtype=torch.float64)
    dt = 0.1
    eye = torch.eye(3, dtype=torch.float64)
    for frame in range(1, video.counter.value):
        video.imu_delta[frame, 0] = dt
        # Static visual state. The correct global bias cancels these raw
        # preintegration deltas under dq_prev without inversion.
        video.imu_delta[frame, 1:4] = dt * video.true_gyro_bias
        video.imu_delta[frame, 4:7] = dt * video.true_accel_bias
        video.imu_delta[frame, 7:10] = 0.5 * dt * dt * video.true_accel_bias
        video.imu_bias_jacobian[frame, 0] = -dt * eye
        video.imu_bias_jacobian[frame, 1] = -dt * eye
        video.imu_bias_jacobian[frame, 3] = -0.5 * dt * dt * eye
    return video


def assert_global_bias_recovery():
    video = make_consistent_video()
    result = refine_global_bias_15dof(
        video,
        compose_order="dq_prev",
        inverse_imu=False,
        max_dt=0.0,
        max_residual_deg=0.0,
        pos_weight=1.0,
        vel_weight=1.0,
        damping=1e-10,
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
        min_relative_improvement=1e-8,
        accel_prior_std=1e6,
        gyro_prior_std=1e6,
        relinearize_accel_threshold=0.0,
        relinearize_gyro_threshold=0.0,
    )
    assert result.get("updated", 0.0) == 1.0, result
    assert torch.allclose(
        video.imu_global_bias_acc, video.true_accel_bias, atol=1e-8, rtol=1e-8
    )
    assert torch.allclose(
        video.imu_global_bias_gyro, video.true_gyro_bias, atol=1e-8, rtol=1e-8
    )


def assert_global_bias_jacobian_with_local_state():
    video = make_consistent_video()
    torch.manual_seed(23)
    video.bias_acc[:] = torch.randn_like(video.bias_acc) * 0.01
    video.bias_gyro[:] = torch.randn_like(video.bias_gyro) * 0.01
    src = torch.arange(0, video.counter.value - 1)
    dst = src + 1
    accel_bias = torch.tensor([0.007, -0.009, 0.004], dtype=torch.float64)
    gyro_bias = torch.tensor([0.002, 0.003, -0.004], dtype=torch.float64)
    _, jacobian = _full_bias_residual_and_jacobian(
        video,
        src,
        dst,
        accel_bias,
        gyro_bias,
        video.imu_gravity,
        "dq_prev",
        False,
        video.bias_acc[src],
        video.bias_gyro[src],
    )
    epsilon = 1e-5
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
            "dq_prev",
            False,
            video.bias_acc[src],
            video.bias_gyro[src],
        )
        minus, _ = _full_bias_residual_and_jacobian(
            video,
            src,
            dst,
            accel_bias - accel_delta,
            gyro_bias - gyro_delta,
            video.imu_gravity,
            "dq_prev",
            False,
            video.bias_acc[src],
            video.bias_gyro[src],
        )
        numerical = (plus - minus) / (2.0 * epsilon)
        error = float((numerical + jacobian[:, :, axis]).abs().max())
        assert error < 1e-6, f"global bias Jacobian mismatch on axis {axis}: {error:.3e}"


def assert_recent_window_global_bias_recovery():
    """The online block must use a bounded recent 15DoF window, not all rows."""

    video = make_consistent_video()
    result = refine_global_bias_15dof(
        video,
        compose_order="dq_prev",
        inverse_imu=False,
        max_dt=0.0,
        max_residual_deg=0.0,
        pos_weight=1.0,
        vel_weight=1.0,
        damping=1e-10,
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
        min_relative_improvement=1e-8,
        accel_prior_std=1e6,
        gyro_prior_std=1e6,
        relinearize_accel_threshold=0.0,
        relinearize_gyro_threshold=0.0,
        window=5,
    )
    assert result.get("updated", 0.0) == 1.0, result
    assert result["window_t0"] == 7.0 and result["window_t1"] == 12.0, result
    assert result["edges"] == 4.0, result
    assert torch.allclose(
        video.imu_global_bias_acc, video.true_accel_bias, atol=1e-8, rtol=1e-8
    )
    assert torch.allclose(
        video.imu_global_bias_gyro, video.true_gyro_bias, atol=1e-8, rtol=1e-8
    )


def assert_recenter_preserves_total_bias():
    video = make_consistent_video()
    video.imu_global_bias_acc[:] = torch.tensor(
        [0.3, -0.2, 0.1], dtype=torch.float64
    )
    video.imu_global_bias_gyro[:] = torch.tensor(
        [0.03, -0.02, 0.01], dtype=torch.float64
    )
    accel_offset = torch.tensor([0.04, -0.03, 0.02], dtype=torch.float64)
    gyro_offset = torch.tensor([0.004, -0.003, 0.002], dtype=torch.float64)
    video.bias_acc[:] = accel_offset
    video.bias_gyro[:] = gyro_offset
    # Keep a few asymmetric local deviations. The componentwise median stays
    # at the intended offset, while the total physical bias must remain exact.
    video.bias_acc[0] += torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
    video.bias_gyro[1] += torch.tensor([0.0, 0.001, 0.0], dtype=torch.float64)
    total_accel_before = video.imu_global_bias_acc + video.bias_acc.clone()
    total_gyro_before = video.imu_global_bias_gyro + video.bias_gyro.clone()

    result = recenter_full_imu_biases(
        video,
        max_dt=0.0,
        max_accel_norm=2.0,
        max_gyro_norm=0.2,
    )
    assert result.get("updated", 0.0) == 1.0, result
    assert torch.allclose(
        video.imu_global_bias_acc + video.bias_acc,
        total_accel_before,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        video.imu_global_bias_gyro + video.bias_gyro,
        total_gyro_before,
        atol=1e-12,
        rtol=1e-12,
    )
    assert result["total_bias_preservation_max_abs"] < 1e-12, result


if __name__ == "__main__":
    assert_global_bias_recovery()
    assert_global_bias_jacobian_with_local_state()
    assert_recent_window_global_bias_recovery()
    assert_recenter_preserves_total_bias()
    print("global 15DoF shared-bias refinement checks passed")

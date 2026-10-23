#!/usr/bin/env python3
"""Finite-difference verification for the runtime 15DoF IMU DBA factor.

This is deliberately a CPU/double-precision test.  It perturbs the exact
DROID ``T_CW`` pose representation using LieTorch's left ``SE3.retr`` update,
then checks the numerical residual derivatives against the Jacobians supplied
to CUDA DBA.  A non-zero camera--IMU lever arm and gravity are included so a
test that only validates a camera-centre or zero-gravity special case cannot
pass accidentally.
"""

from pathlib import Path
import sys

import torch
from lietorch import SE3


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "droid_slam"))

from geom.imu_factor import (  # noqa: E402
    full_imu_factor_linearization,
    rotvec_to_quat,
)


DTYPE = torch.float64
EPS = 1e-6


def _pose(translation, rotvec):
    return torch.cat(
        (
            torch.as_tensor(translation, dtype=DTYPE),
            rotvec_to_quat(torch.as_tensor(rotvec, dtype=DTYPE)),
        )
    )


def _left_pose_retract(pose, increment):
    """Apply DROID's actual six-dimensional left pose update."""

    return SE3(pose.view(1, 7)).retr(increment.view(1, 6)).data[0]


def _make_problem():
    torch.manual_seed(20261020)
    return {
        "pose_i": _pose([0.30, -0.20, 0.50], [0.12, -0.07, 0.08]),
        "pose_j": _pose([-0.15, 0.35, 0.42], [-0.04, 0.15, -0.11]),
        "velocity_i": torch.tensor([0.22, -0.11, 0.35], dtype=DTYPE),
        "velocity_j": torch.tensor([-0.06, 0.17, 0.28], dtype=DTYPE),
        "accel_bias_i": torch.tensor([0.012, -0.007, 0.004], dtype=DTYPE),
        "accel_bias_j": torch.tensor([-0.006, 0.009, -0.003], dtype=DTYPE),
        "gyro_bias_i": torch.tensor([0.001, -0.003, 0.002], dtype=DTYPE),
        "gyro_bias_j": torch.tensor([-0.002, 0.004, -0.001], dtype=DTYPE),
        "global_accel_bias": torch.tensor([0.021, -0.013, 0.009], dtype=DTYPE),
        "global_gyro_bias": torch.tensor([-0.004, 0.006, 0.003], dtype=DTYPE),
        "imu_delta": torch.tensor(
            [0.05, 0.011, -0.007, 0.014, 0.024, -0.016, 0.018,
             0.0008, -0.0004, 0.0011],
            dtype=DTYPE,
        ),
        "bias_jacobian": torch.randn(5, 3, 3, dtype=DTYPE) * 0.004,
        "gravity": torch.tensor([0.0, 0.0, -9.81], dtype=DTYPE),
        "lever_arm": torch.tensor([0.0652, -0.0207, -0.0081], dtype=DTYPE),
    }


def _linearize(problem, state_i=None, state_j=None, global_bias_state=None):
    if state_i is None:
        state_i = torch.zeros(15, dtype=DTYPE)
    if state_j is None:
        state_j = torch.zeros(15, dtype=DTYPE)
    if global_bias_state is None:
        global_bias_state = torch.zeros(6, dtype=DTYPE)

    pose_i = _left_pose_retract(problem["pose_i"], state_i[:6])
    pose_j = _left_pose_retract(problem["pose_j"], state_j[:6])
    return full_imu_factor_linearization(
        pose_i,
        pose_j,
        problem["velocity_i"] + state_i[6:9],
        problem["velocity_j"] + state_j[6:9],
        problem["accel_bias_i"] + state_i[9:12],
        problem["accel_bias_j"] + state_j[9:12],
        problem["gyro_bias_i"] + state_i[12:15],
        problem["gyro_bias_j"] + state_j[12:15],
        problem["global_accel_bias"] + global_bias_state[:3],
        problem["global_gyro_bias"] + global_bias_state[3:6],
        problem["imu_delta"],
        problem["bias_jacobian"],
        problem["gravity"],
        problem["lever_arm"],
        compose_order="dq_prev",
        inverse_imu=True,
        optimize_bias=True,
    )


def _numerical_jacobian(problem, endpoint):
    numerical = torch.zeros(15, 15, dtype=DTYPE)
    for axis in range(15):
        plus_i = torch.zeros(15, dtype=DTYPE)
        minus_i = torch.zeros(15, dtype=DTYPE)
        plus_j = torch.zeros(15, dtype=DTYPE)
        minus_j = torch.zeros(15, dtype=DTYPE)
        if endpoint == "i":
            plus_i[axis] = EPS
            minus_i[axis] = -EPS
        else:
            plus_j[axis] = EPS
            minus_j[axis] = -EPS
        residual_plus, *_ = _linearize(problem, plus_i, plus_j)
        residual_minus, *_ = _linearize(problem, minus_i, minus_j)
        numerical[:, axis] = (residual_plus - residual_minus) / (2.0 * EPS)
    return numerical


def _assert_endpoint(problem, endpoint):
    residual, J_i, J_j, _ = _linearize(problem)
    analytic = J_i if endpoint == "i" else J_j
    numerical = _numerical_jacobian(problem, endpoint)
    # DBA receives r - J dx, hence the numerical derivative is -J.
    error = (numerical + analytic).abs()
    row_names = ("position", "velocity", "rotation", "accel_bias", "gyro_bias")
    row_errors = {
        name: float(error[3 * index : 3 * (index + 1)].max())
        for index, name in enumerate(row_names)
    }
    maximum = float(error.max())
    print(
        f"[FULL-IMU-JAC] endpoint={endpoint} max={maximum:.3e} "
        + " ".join(f"{name}={value:.3e}" for name, value in row_errors.items())
    )
    # 1e-5 is well below the scale at which one DBA step would be affected,
    # while allowing central differences through LieTorch's exponential map.
    assert maximum < 1e-5, (
        f"full 15DoF factor Jacobian mismatch at endpoint {endpoint}: "
        f"max={maximum:.3e}, per-row={row_errors}, residual_norm="
        f"{float(torch.linalg.norm(residual)):.3e}"
    )


def _assert_global_bias_jacobian(problem):
    """Global bias affects measurement rows, never local RW transition rows."""

    _, source_jacobian, _, _ = _linearize(problem)
    analytic = torch.zeros(15, 6, dtype=DTYPE)
    analytic[:9] = source_jacobian[:9, 9:15]
    numerical = torch.zeros(15, 6, dtype=DTYPE)
    for axis in range(6):
        plus = torch.zeros(6, dtype=DTYPE)
        minus = torch.zeros(6, dtype=DTYPE)
        plus[axis] = EPS
        minus[axis] = -EPS
        residual_plus, *_ = _linearize(problem, global_bias_state=plus)
        residual_minus, *_ = _linearize(problem, global_bias_state=minus)
        numerical[:, axis] = (residual_plus - residual_minus) / (2.0 * EPS)

    # DBA uses r - J dx, hence numerical residual derivative is -J.
    maximum = float((numerical + analytic).abs().max())
    print(f"[FULL-IMU-JAC] global_bias max={maximum:.3e}")
    assert maximum < 1e-5, (
        "full 15DoF shared-global-bias Jacobian mismatch: "
        f"max={maximum:.3e}"
    )


def main():
    problem = _make_problem()
    _assert_endpoint(problem, "i")
    _assert_endpoint(problem, "j")
    _assert_global_bias_jacobian(problem)
    print("[PASS] full 15DoF IMU factor Jacobians match DROID left-pose finite differences")


if __name__ == "__main__":
    main()

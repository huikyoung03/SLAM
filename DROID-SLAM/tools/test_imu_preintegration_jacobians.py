#!/usr/bin/env python3
"""Numerical regression checks for IMU bias Jacobians and row composition."""

import importlib.util
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from droid_slam.imu_prior import compose_imu_prior_rows  # noqa: E402


def load_preintegrator():
    path = ROOT / "tools" / "imu_preintegrate.py"
    spec = importlib.util.spec_from_file_location("imu_preintegrate_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_window(samples=101, dt=0.005):
    rows = []
    for index in range(samples):
        t = index * dt
        rows.append({
            "timestamp_ns": int(round(t * 1e9)),
            "gx": 0.7 + 0.2 * math.sin(2.3 * t),
            "gy": -0.4 + 0.25 * math.cos(1.7 * t),
            "gz": 0.5 + 0.15 * math.sin(1.1 * t),
            "ax": 1.2 + 0.3 * math.sin(1.5 * t),
            "ay": -0.8 + 0.2 * math.cos(2.1 * t),
            "az": 9.81 + 0.4 * math.sin(0.8 * t),
        })
    return rows


def matrix_from_row(row, prefix):
    return [
        [float(row[f"{prefix}_{i}{j}"]) for j in range(3)]
        for i in range(3)
    ]


def max_matrix_error(first, second):
    return max(
        abs(first[i][j] - second[i][j])
        for i in range(3)
        for j in range(3)
    )


def add_metadata(row, index):
    result = dict(row)
    result.update({
        "frame_index": index,
        "imu_valid": 1,
        "imu_weight": 1.0,
        "imu_reason": "ok",
        "rot_var": 1.0,
        "vel_var": 1.0,
        "pos_var": 1.0,
    })
    return result


def test_composed_jacobians_match_direct_preintegration():
    pre = load_preintegrator()
    raw = make_window()
    direct = pre.integrate_imu_window(
        raw, raw[0]["timestamp_ns"], raw[-1]["timestamp_ns"]
    )
    assert direct["bias_jacobian_method"] == "central_difference"

    rows = []
    for interval in range(10):
        window = raw[interval * 10 : interval * 10 + 11]
        rows.append(add_metadata(
            pre.integrate_imu_window(
                window, window[0]["timestamp_ns"], window[-1]["timestamp_ns"]
            ),
            interval + 1,
        ))

    composed = compose_imu_prior_rows(rows, prev_frame_index=0, frame_index=10)
    assert composed is not None

    for prefix in ("dr", "dv", "dp"):
        error = max(
            abs(float(direct[f"{prefix}_{axis}"]) - float(composed[f"{prefix}_{axis}"]))
            for axis in "xyz"
        )
        assert error < 1e-9, f"{prefix} composition mismatch: {error:.3e}"

    for prefix in ("J_R_bg", "J_v_ba", "J_v_bg", "J_p_ba", "J_p_bg"):
        error = max_matrix_error(
            matrix_from_row(direct, prefix),
            matrix_from_row(composed, prefix),
        )
        assert error < 1e-6, f"{prefix} composition Jacobian mismatch: {error:.3e}"


if __name__ == "__main__":
    test_composed_jacobians_match_direct_preintegration()
    print("[PASS] IMU preintegration and composed bias Jacobians match direct integration")

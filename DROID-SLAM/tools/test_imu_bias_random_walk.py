#!/usr/bin/env python3
"""Regression checks for calibrated 15DoF local-bias transition weighting."""

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "droid_slam"))

from geom.imu_factor import bias_random_walk_relative_sqrt_information  # noqa: E402
from geom.ba import _select_temporal_full_covariance_scale  # noqa: E402
from factor_graph import FactorGraph  # noqa: E402


def main():
    dtype = torch.float64
    # EuRoC ADIS16448 values used by the strict V1_03 prior.
    random_walk = torch.tensor([0.003, 1.9393e-5], dtype=dtype)
    rotation_info = torch.tensor(694652451.7116, dtype=dtype)

    scale_50ms, valid_50ms = bias_random_walk_relative_sqrt_information(
        random_walk, torch.tensor(0.05, dtype=dtype), rotation_info
    )
    scale_200ms, valid_200ms = bias_random_walk_relative_sqrt_information(
        random_walk, torch.tensor(0.20, dtype=dtype), rotation_info
    )

    assert bool(valid_50ms) and bool(valid_200ms)
    # sqrt-information must follow 1 / sqrt(dt), not a fixed manual 0.001.
    ratio = scale_50ms / scale_200ms
    assert torch.allclose(ratio, torch.full_like(ratio, 2.0), rtol=1e-12, atol=1e-12)
    assert torch.allclose(scale_50ms[:3], scale_50ms[0].expand(3))
    assert torch.allclose(scale_50ms[3:], scale_50ms[3].expand(3))
    assert float(scale_50ms[3]) > float(scale_50ms[0])

    # Training uses [B, E, 1] dt and [B, 1] rotation references.
    batched_scale, batched_valid = bias_random_walk_relative_sqrt_information(
        random_walk.view(1, 1, 2).expand(2, 1, 2),
        torch.tensor([[[0.05], [0.20]], [[0.10], [0.40]]], dtype=dtype),
        rotation_info.expand(2).view(2, 1),
    )
    assert tuple(batched_scale.shape) == (2, 2, 6)
    assert bool(batched_valid.all())
    assert torch.allclose(batched_scale[0, 0] / batched_scale[0, 1], torch.full((6,), 2.0, dtype=dtype))

    # Regression for the normal training shape: batch=1 and a clip has more
    # than two temporal intervals.  A misplaced unsqueeze on only ``dt``
    # creates [B, E, E] here and crashes when it meets the two RW densities.
    clip_scale, clip_valid = bias_random_walk_relative_sqrt_information(
        random_walk.view(1, 1, 2),
        torch.full((1, 6, 1), 0.05, dtype=dtype),
        rotation_info.view(1, 1),
    )
    assert tuple(clip_scale.shape) == (1, 6, 6)
    assert bool(clip_valid.all())

    _, invalid = bias_random_walk_relative_sqrt_information(
        torch.tensor([0.0, 1.9393e-5], dtype=dtype),
        torch.tensor(0.05, dtype=dtype),
        rotation_info,
    )
    assert not bool(invalid)

    # The full calibrated path must keep the *cross-block* information ratio.
    # In particular it must not divide p/v by their own median and then hide
    # them behind the legacy manual 0.05 coefficients.
    graph = FactorGraph.__new__(FactorGraph)
    graph.imu_info_weight_eps = 1e-12
    graph.video = type("Video", (), {
        "imu_info": torch.tensor(
            [[0.0, 0.0, 0.0], [100.0, 4.0, 25.0]], dtype=dtype
        )
    })()
    term_scale, scales = graph._imu_full_covariance_term_scale(
        1,
        torch.tensor([100.0, 4.0, 25.0], dtype=dtype),
        torch.device("cpu"),
        dtype,
    )
    assert term_scale is not None
    assert torch.allclose(
        term_scale[:9],
        torch.tensor([0.5] * 3 + [0.2] * 3 + [1.0] * 3, dtype=dtype),
    )
    assert scales == (1.0, 0.2, 0.5)

    # Training must use exactly the same p/v/R convention as FactorGraph:
    # divide all three information categories by the rotation reference, then
    # map [R,v,p] into residual-row order [p,v,R].
    train_scale, train_valid = _select_temporal_full_covariance_scale(
        torch.tensor([[0.0, 0.0, 0.0], [100.0, 4.0, 25.0]], dtype=dtype),
        torch.tensor([100.0, 4.0, 25.0], dtype=dtype),
        torch.tensor([1]),
        batch=1,
        num_frames=2,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    assert bool(train_valid[0, 0])
    assert torch.allclose(
        train_scale[0, 0],
        torch.tensor([1.0, 0.2, 0.5], dtype=dtype),
    )

    print(
        "[PASS] calibrated local-bias random-walk scales: "
        f"accel={float(scale_50ms[0]):.6g}, gyro={float(scale_50ms[3]):.6g}, "
        "dt-law=1/sqrt(dt); p/v/R cross-block covariance ratio preserved"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU checks for component-wise learned IMU confidence wiring."""

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "droid_slam"))

from droid_net import UpdateModule  # noqa: E402
from geom.ba import _select_temporal_confidence  # noqa: E402
from geom.imu_factor import (  # noqa: E402
    edge_imu_quality_features,
    temporal_imu_confidence_from_edges,
)


def assert_scalar_initialization_is_exact():
    torch.manual_seed(7)
    update = UpdateModule().eval()
    update.initialize_component_imu_confidence_from_scalar()

    net = torch.randn(1, 2, 128, 4, 4)
    inp = torch.randn(1, 2, 128, 4, 4)
    corr = torch.randn(1, 2, 196, 4, 4)
    flow = torch.randn(1, 2, 4, 4, 4)
    ii = torch.tensor([0, 1])
    jj = torch.tensor([1, 0])

    with torch.no_grad():
        scalar = update(
            net.clone(), inp, corr, flow, ii, jj,
            return_imu_confidence=True,
        )[-1]
        components = update(
            net.clone(), inp, corr, flow, ii, jj,
            return_imu_confidence=True,
            return_component_imu_confidence=True,
            imu_quality=torch.zeros(1, 2, 7),
        )[-1]

    assert components.shape[-1] == 3
    assert torch.allclose(components, scalar.expand_as(components), atol=1e-6)

    # Quality fusion must start with a zero output without becoming a dead
    # all-zero MLP. Its output projection receives gradients on the first step.
    first_weight = update.imu_quality_fusion[0].weight
    final_weight = update.imu_quality_fusion[2].weight
    assert torch.count_nonzero(first_weight).item() > 0
    assert torch.count_nonzero(final_weight).item() == 0

    quality = torch.randn(2, 7)
    update.zero_grad(set_to_none=True)
    update.imu_quality_fusion(quality).sum().backward()
    assert update.imu_quality_fusion[2].weight.grad is not None
    assert torch.count_nonzero(update.imu_quality_fusion[2].weight.grad).item() > 0


def assert_temporal_component_and_quality_selection():
    ii = torch.tensor([0, 1, 0, 2])
    jj = torch.tensor([1, 0, 2, 3])
    edge_confidence = torch.tensor(
        [[[[[0.2, 0.4, 0.6]]], [[[0.4, 0.6, 0.8]]],
          [[[0.1, 0.1, 0.1]]], [[[0.8, 0.7, 0.5]]]]]
    )
    temporal, coverage = temporal_imu_confidence_from_edges(
        edge_confidence, ii, jj, num_frames=4, return_coverage=True
    )
    assert temporal.shape == (1, 4, 3)
    assert coverage.tolist() == [False, True, False, True]
    assert torch.allclose(temporal[0, 1], torch.tensor([0.3, 0.5, 0.7]))
    assert torch.allclose(temporal[0, 2], torch.ones(3))

    selected = _select_temporal_confidence(
        temporal, torch.tensor([1, 2, 3]), 1, 4, torch.device("cpu"), torch.float32
    )
    assert selected.shape == (1, 3, 3)
    assert torch.allclose(selected[0, 1], torch.ones(3))

    imu_delta = torch.zeros(4, 10)
    imu_delta[1:, 0] = 0.05
    imu_delta[1:, 1:4] = 0.01
    quality = edge_imu_quality_features(
        imu_delta, torch.ones(4, 3), ii, jj, num_frames=4
    )
    assert quality.shape == (1, 4, 7)
    # (0, 2) is a proximity edge, not one physical IMU interval.
    assert torch.allclose(quality[:, 2], torch.zeros_like(quality[:, 2]))
    assert torch.isfinite(quality).all()


if __name__ == "__main__":
    assert_scalar_initialization_is_exact()
    assert_temporal_component_and_quality_selection()
    print("component IMU-confidence checks passed")

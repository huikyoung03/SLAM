#!/usr/bin/env python3
"""CPU wiring checks for the DVI-style scalar IMU confidence branch."""

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "droid_slam"))

from droid_net import UpdateModule  # noqa: E402
from factor_graph import FactorGraph  # noqa: E402
from geom.imu_factor import temporal_imu_confidence_from_edges  # noqa: E402


def assert_dvi_scalar_shape_and_neutral_initialization():
    torch.manual_seed(11)
    update = UpdateModule().eval()

    net = torch.randn(1, 2, 128, 4, 4)
    inp = torch.randn(1, 2, 128, 4, 4)
    corr = torch.randn(1, 2, 196, 4, 4)
    flow = torch.randn(1, 2, 4, 4, 4)
    ii = torch.tensor([0, 1])
    jj = torch.tensor([1, 2])

    with torch.no_grad():
        confidence = update(
            net, inp, corr, flow, ii, jj,
            return_imu_confidence=True,
            use_dvi_imu_confidence=True,
        )[-1]

    assert confidence.shape == (1, 2, 1, 1, 1)
    assert torch.allclose(confidence, torch.full_like(confidence, 0.5))

    temporal, coverage = temporal_imu_confidence_from_edges(
        confidence, ii, jj, num_frames=3, return_coverage=True
    )
    # Scalar confidence stays scalar after temporal aggregation; FactorGraph
    # expands it to [rotation, velocity, position] only when building a full
    # IMU prior.
    assert temporal.shape == (1, 3)
    assert coverage.tolist() == [False, True, True]
    assert torch.allclose(temporal[0, 1:], torch.full((2,), 0.5))


def assert_manual_schedule_is_scalar_and_outer_iteration_based():
    # Avoid a CUDA/video allocation: these helpers depend only on the small
    # scheduling state below.
    graph = FactorGraph.__new__(FactorGraph)
    graph.imu_tuned_confidence_schedule = (0.25, 0.50, 1.00)
    graph._imu_tuned_confidence_iteration = 0
    graph.use_full_imu_ba = True
    graph.use_imu_ba_prior = True
    graph.require_imu_alignment = False

    assert torch.allclose(
        torch.as_tensor(graph._confidence_components_for_pose_index(9), dtype=torch.float64),
        torch.tensor([0.25, 0.25, 0.25], dtype=torch.float64),
    )
    graph._advance_tuned_confidence_schedule()
    assert graph._current_tuned_confidence_value() == 0.50
    graph._advance_tuned_confidence_schedule()
    graph._advance_tuned_confidence_schedule()
    assert graph._current_tuned_confidence_value() == 1.00


def assert_three_frame_scalar_cache_is_not_misread_as_components():
    graph = FactorGraph.__new__(FactorGraph)
    graph.imu_tuned_confidence_schedule = None
    graph.use_learned_imu_confidence = True
    graph.use_component_imu_confidence = False
    graph.last_imu_confidence = torch.tensor([[1.0, 0.4, 0.7]])
    assert torch.allclose(
        torch.as_tensor(graph._confidence_components_for_pose_index(1)),
        torch.tensor([0.4, 0.4, 0.4], dtype=torch.float64),
    )


if __name__ == "__main__":
    assert_dvi_scalar_shape_and_neutral_initialization()
    assert_manual_schedule_is_scalar_and_outer_iteration_based()
    assert_three_frame_scalar_cache_is_not_misread_as_components()
    print("DVI scalar IMU-confidence checks passed")

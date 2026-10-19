#!/usr/bin/env python3
"""CPU-only checks for persistent direct temporal IMU confidence."""

from pathlib import Path
from types import SimpleNamespace
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "droid_slam"))

from factor_graph import FactorGraph  # noqa: E402
from depth_video import DepthVideo  # noqa: E402
from vi_initialization import _shared_bias_confidence  # noqa: E402


class MockVideo:
    update_imu_learned_confidence = DepthVideo.update_imu_learned_confidence

    def __init__(self, frames):
        self.counter = SimpleNamespace(value=frames)
        self.imu_learned_confidence = torch.ones(frames, dtype=torch.float64)
        self.imu_learned_confidence_valid = torch.zeros(frames, dtype=torch.bool)


def assert_only_direct_temporal_edges_are_cached():
    graph = FactorGraph.__new__(FactorGraph)
    graph.device = torch.device("cpu")
    graph.video = MockVideo(frames=6)
    graph.imu_confidence_floor = 0.0
    graph.ii = torch.tensor([0, 1, 1, 2, 3, 0], dtype=torch.long)
    graph.jj = torch.tensor([1, 0, 2, 4, 4, 5], dtype=torch.long)
    edge_confidence = torch.tensor([[0.2, 0.6, 0.8, 0.1, 0.7, 0.05]])

    confidence, coverage = graph._edge_confidence_to_frame_confidence(
        edge_confidence, return_coverage=True
    )
    updated = graph._persist_imu_confidence(confidence, coverage)

    assert updated == 3
    assert torch.equal(
        graph.video.imu_learned_confidence_valid,
        torch.tensor([False, True, True, False, True, False]),
    )
    expected = torch.tensor([1.0, 0.4, 0.8, 1.0, 0.7, 1.0], dtype=torch.float64)
    assert torch.allclose(graph.video.imu_learned_confidence, expected)

    dst = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    selected = _shared_bias_confidence(
        graph.video.imu_learned_confidence,
        dst,
        graph.video.counter.value,
        torch.device("cpu"),
        torch.float64,
    )
    assert torch.allclose(selected, expected[dst])


if __name__ == "__main__":
    assert_only_direct_temporal_edges_are_cached()
    print("persistent direct temporal IMU-confidence checks passed")

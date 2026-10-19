#!/usr/bin/env python3
"""CPU wiring checks for scalar DVI confidence with IMU-quality fusion."""

from pathlib import Path
import sys
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "droid_slam"))

from droid_net import UpdateModule  # noqa: E402
from factor_graph import FactorGraph  # noqa: E402


def run_update(update, net, inp, corr, flow, ii, jj, quality, use_quality):
    return update(
        net,
        inp,
        corr,
        flow,
        ii,
        jj,
        return_imu_confidence=True,
        use_dvi_imu_confidence=True,
        use_dvi_imu_quality_fusion=use_quality,
        imu_quality=quality,
    )[-1]


def assert_dvi_quality_fusion_is_neutral_then_trainable():
    torch.manual_seed(23)
    update = UpdateModule().eval()

    net = torch.randn(1, 2, 128, 4, 4)
    inp = torch.randn(1, 2, 128, 4, 4)
    corr = torch.randn(1, 2, 196, 4, 4)
    flow = torch.randn(1, 2, 4, 4, 4)
    ii = torch.tensor([0, 1])
    jj = torch.tensor([1, 2])
    quality = torch.randn(1, 2, 7)

    with torch.no_grad():
        scalar = run_update(
            update, net.clone(), inp, corr, flow, ii, jj, quality, False
        )
        fused = run_update(
            update, net.clone(), inp, corr, flow, ii, jj, quality, True
        )

    assert fused.shape == (1, 2, 1, 1, 1)
    # A zero output projection makes the new branch backward-compatible at
    # initialization, even though the first MLP layer is nonzero.
    assert torch.allclose(fused, scalar, atol=1e-6)
    assert torch.count_nonzero(update.dvi_imu_quality_fusion[0].weight).item() > 0
    assert torch.count_nonzero(update.dvi_imu_quality_fusion[2].weight).item() == 0

    update.zero_grad(set_to_none=True)
    confidence = run_update(
        update, net.clone(), inp, corr, flow, ii, jj, quality, True
    )
    confidence.sum().backward()
    output_weight_grad = update.dvi_imu_quality_fusion[2].weight.grad
    assert output_weight_grad is not None
    assert torch.count_nonzero(output_weight_grad).item() > 0

    with torch.no_grad():
        update.dvi_imu_quality_fusion[2].bias.fill_(1.0)
        shifted = run_update(
            update, net.clone(), inp, corr, flow, ii, jj, quality, True
        )
    assert torch.all(shifted > scalar)


def assert_runtime_dvi_quality_features_are_not_dropped():
    """DVI scalar runtime must receive the same seven IMU inputs as training."""

    graph = FactorGraph.__new__(FactorGraph)
    graph.use_component_imu_confidence = False
    graph.use_dvi_imu_quality_fusion = True
    graph.video = SimpleNamespace(
        counter=SimpleNamespace(value=3),
        imu_delta=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.01, 0.0, 0.0, 0.1, 0.0, 0.0, 0.01, 0.0, 0.0],
                [0.1, 0.02, 0.0, 0.0, 0.2, 0.0, 0.0, 0.02, 0.0, 0.0],
            ]
        ),
        imu_info=torch.ones(3, 3),
    )
    quality = graph._edge_imu_quality(
        torch.tensor([0, 1]), torch.tensor([1, 2])
    )
    assert quality is not None
    assert quality.shape == (1, 2, 7)
    assert torch.isfinite(quality).all()

    graph.use_dvi_imu_quality_fusion = False
    assert graph._edge_imu_quality(
        torch.tensor([0, 1]), torch.tensor([1, 2])
    ) is None


def assert_missing_direct_edge_uses_quality_confidence_not_one():
    """A valid IMU row without (j-1,j) visual edge must not bypass c_imu."""

    torch.manual_seed(29)
    update = UpdateModule().eval()
    with torch.no_grad():
        # Make the quality-only prediction visibly different from the legacy
        # neutral fallback of one.
        update.dvi_imu_quality_fusion[2].bias.fill_(-1.2)

    graph = FactorGraph.__new__(FactorGraph)
    graph.device = torch.device("cpu")
    graph.use_dvi_imu_quality_fusion = True
    graph.imu_confidence_floor = 0.0
    graph.update_op = update
    graph.video = SimpleNamespace(
        counter=SimpleNamespace(value=3),
        imu_delta=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.01, 0.0, 0.0, 0.1, 0.0, 0.0, 0.01, 0.0, 0.0],
                [0.1, 0.02, 0.0, 0.0, 0.2, 0.0, 0.0, 0.02, 0.0, 0.0],
            ]
        ),
        imu_info=torch.ones(3, 3),
    )
    edge_confidence = torch.tensor([[[0.8]]])
    temporal, coverage = graph._edge_confidence_to_frame_confidence(
        edge_confidence,
        ii=torch.tensor([0]),
        jj=torch.tensor([1]),
        return_coverage=True,
    )
    expected = graph._temporal_dvi_quality_confidence(3)
    assert coverage.tolist() == [False, True, False]
    assert torch.allclose(temporal[0, 1], torch.tensor(0.8))
    assert torch.allclose(temporal[0, 2], expected[0, 2, 0])
    assert temporal[0, 2].item() < 0.5
    assert not torch.allclose(temporal[0, 2], torch.tensor(1.0))


if __name__ == "__main__":
    assert_dvi_quality_fusion_is_neutral_then_trainable()
    assert_runtime_dvi_quality_features_are_not_dropped()
    assert_missing_direct_edge_uses_quality_confidence_not_one()
    print("DVI scalar IMU-quality fusion checks passed")

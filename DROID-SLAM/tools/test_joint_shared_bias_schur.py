#!/usr/bin/env python3
"""Deterministic algebra check for the coupled 9D plus shared-bias solve."""

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "droid_slam"))

from factor_graph import FactorGraph  # noqa: E402


def assert_schur_reduction_matches_direct_joint_solve():
    torch.manual_seed(17)
    dtype = torch.float64
    states = 4
    state_dim = states * 9

    raw_x = torch.randn(state_dim, state_dim, dtype=dtype)
    hxx = raw_x.transpose(0, 1) @ raw_x + 0.5 * torch.eye(state_dim, dtype=dtype)
    hxb = torch.randn(states, 9, 6, dtype=dtype) * 0.1
    raw_b = torch.randn(6, 6, dtype=dtype)
    hbb = raw_b.transpose(0, 1) @ raw_b + 2.0 * torch.eye(6, dtype=dtype)
    gx = torch.randn(state_dim, dtype=dtype)
    gb = torch.randn(6, dtype=dtype)

    hxb_dense = hxb.reshape(state_dim, 6)
    full_hessian = torch.cat(
        [
            torch.cat([hxx, hxb_dense], dim=1),
            torch.cat([hxb_dense.transpose(0, 1), hbb], dim=1),
        ],
        dim=0,
    )
    direct_step = torch.linalg.solve(full_hessian, torch.cat([gx, gb], dim=0))

    h_correction, g_correction = FactorGraph._schur_eliminate_shared_bias(
        hxb, hbb, gb
    )
    reduced_hessian = hxx + h_correction.permute(0, 2, 1, 3).reshape(
        state_dim, state_dim
    )
    reduced_gradient = gx + g_correction.reshape(-1)
    reduced_x = torch.linalg.solve(reduced_hessian, reduced_gradient)
    recovered_b = torch.linalg.solve(
        hbb,
        gb - hxb_dense.transpose(0, 1) @ reduced_x,
    )

    assert torch.allclose(reduced_x, direct_step[:state_dim], atol=1e-10, rtol=1e-10)
    assert torch.allclose(recovered_b, direct_step[state_dim:], atol=1e-10, rtol=1e-10)


def assert_outer_relinearization_schedule():
    """Check that the shared bias is recovered after its matching CUDA step."""

    class FakeVideo:
        def __init__(self):
            self.calls = []

        def ba(self, *args, **kwargs):
            self.calls.append(kwargs)
            return (torch.zeros(1, 9),)

    graph = FactorGraph.__new__(FactorGraph)
    graph.use_full_imu_ba = True
    graph.use_joint_shared_bias_ba = True
    graph.video = FakeVideo()
    build_calls = []
    bias_updates = []

    def build_joint(*args, **kwargs):
        build_calls.append(kwargs)
        return None, None, None, None, {"outer": len(build_calls)}

    def recover_bias(*args, **kwargs):
        bias_updates.append(kwargs["stage"])

    graph._build_full_imu_ba_prior = build_joint
    graph._apply_joint_shared_bias_update = recover_bias

    graph._run_dba_with_imu_prior(
        target=None,
        weight=None,
        damping=None,
        ii=None,
        jj=None,
        ba_t0=1,
        ba_t1=4,
        itrs=2,
        lm=1e-4,
        ep=0.1,
        motion_only=False,
        stage="test",
        joint_bias_requested=True,
        outer_relinearize=True,
    )

    assert [call["itrs"] for call in graph.video.calls] == [1, 1]
    assert len(build_calls) == 2
    assert all(call["enable_joint_bias"] for call in build_calls)
    assert all(not call["mark_joint_attempt"] for call in build_calls)
    assert bias_updates == ["test-outer1", "test-outer2"]

    graph = FactorGraph.__new__(FactorGraph)
    graph.use_full_imu_ba = True
    graph.use_joint_shared_bias_ba = False
    graph.video = FakeVideo()
    fixed_builds = []

    def build_fixed(*args, **kwargs):
        fixed_builds.append((args, kwargs))
        return None, None, None, None

    graph._build_full_imu_ba_prior = build_fixed
    graph._run_dba_with_imu_prior(
        target=None,
        weight=None,
        damping=None,
        ii=None,
        jj=None,
        ba_t0=1,
        ba_t1=4,
        itrs=2,
        lm=1e-4,
        ep=0.1,
        motion_only=False,
        stage="test",
        outer_relinearize=True,
    )
    assert [call["itrs"] for call in graph.video.calls] == [1, 1]
    assert len(fixed_builds) == 2

    graph.video.calls.clear()
    fixed_builds.clear()
    graph._run_dba_with_imu_prior(
        target=None,
        weight=None,
        damping=None,
        ii=None,
        jj=None,
        ba_t0=1,
        ba_t1=4,
        itrs=2,
        lm=1e-4,
        ep=0.1,
        motion_only=False,
        stage="test",
        outer_relinearize=False,
    )
    assert [call["itrs"] for call in graph.video.calls] == [2]
    assert len(fixed_builds) == 1


if __name__ == "__main__":
    assert_schur_reduction_matches_direct_joint_solve()
    assert_outer_relinearization_schedule()
    print("joint shared-bias Schur reduction and outer relinearization schedule passed")

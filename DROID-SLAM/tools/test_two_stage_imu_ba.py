#!/usr/bin/env python3
"""Small deterministic checks for the local pose-only -> full-15DoF schedule."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "droid_slam"))

from droid_frontend import DroidFrontend  # noqa: E402
from factor_graph import FactorGraph  # noqa: E402


def assert_factor_graph_stages_are_disjoint():
    graph = FactorGraph.__new__(FactorGraph)
    graph._configured_full_imu_ba = True
    graph._configured_imu_ba_prior = True
    graph._configured_imu_optimize_bias = True
    graph.use_full_imu_ba = True
    graph.use_imu_ba_prior = True
    graph.imu_optimize_bias = True
    graph.apply_imu_residual = True

    graph.set_imu_ba_stage("pose_only")
    assert not graph.use_full_imu_ba
    assert not graph.use_imu_ba_prior
    assert not graph.imu_optimize_bias
    assert not graph.apply_imu_residual

    graph.set_imu_ba_stage("full_15dof")
    assert graph.use_full_imu_ba
    assert graph.use_imu_ba_prior
    assert graph.imu_optimize_bias
    assert not graph.apply_imu_residual


def assert_frontend_runs_pose_before_full_stage():
    class FakeGraph:
        def __init__(self):
            self.events = []

        def set_imu_ba_stage(self, stage):
            self.events.append(("stage", stage))

        def update(self, *args, **kwargs):
            self.events.append(("update", kwargs["use_inactive"]))

    frontend = DroidFrontend.__new__(DroidFrontend)
    frontend.graph = FakeGraph()
    frontend.use_two_stage_imu_ba = True

    frontend._run_local_dba_stage("pose_only", 3)
    frontend._run_local_dba_stage("full_15dof", 2)

    assert frontend.graph.events == [
        ("stage", "pose_only"),
        ("update", True),
        ("update", True),
        ("update", True),
        ("stage", "full_15dof"),
        ("update", True),
        ("update", True),
    ]


def assert_visual_bootstrap_control_never_requires_imu_alignment():
    class Counter:
        value = 12

    class Video:
        counter = Counter()

    frontend = DroidFrontend.__new__(DroidFrontend)
    frontend.visual_bootstrap_control = True
    frontend.vi_init_bootstrap_frames = 60
    frontend.video = Video()

    assert frontend._retain_startup_keyframes_for_vi()
    frontend.video.counter.value = 60
    assert not frontend._retain_startup_keyframes_for_vi()


if __name__ == "__main__":
    assert_factor_graph_stages_are_disjoint()
    assert_frontend_runs_pose_before_full_stage()
    assert_visual_bootstrap_control_never_requires_imu_alignment()
    print("two-stage IMU BA schedule checks passed")

#!/usr/bin/env python3
"""Export a compact, weight-free HDF5 architecture specification.

This is deliberately *not* a Keras H5 model.  DROID-SLAM uses PyTorch and
custom CUDA correlation / dense BA operators, which cannot be represented as
a portable Keras computation graph.  The output is a small diagram/reproduc-
ibility artifact: JSON records for layers, graph nodes, graph edges, and the
runtime configuration of the current 9DoF visual-inertial experiment.
"""

import argparse
import json
from pathlib import Path

import h5py


def dump_records(group, name, records):
    """Store human-readable JSON records without serialising model weights."""
    dtype = h5py.string_dtype(encoding="utf-8")
    values = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
    group.create_dataset(name, data=values, dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/dvi_mh03_10fps_stage3_10000_005000.pth",
        help="Reference checkpoint only; no weight tensor is copied.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    layers = [
        {
            "id": "fnet",
            "name": "DROID FNet",
            "kind": "visual feature encoder",
            "output": "128-channel feature map",
            "status": "original DROID",
        },
        {
            "id": "cnet",
            "name": "DROID CNet",
            "kind": "context encoder",
            "output": "256 channels split into 128-D ConvGRU hidden state + 128-D context",
            "status": "original DROID",
        },
        {
            "id": "corr",
            "name": "Correlation volume",
            "kind": "multi-scale feature correspondence",
            "status": "original DROID / custom CUDA path",
        },
        {
            "id": "convgru",
            "name": "UpdateModule ConvGRU",
            "kind": "recurrent visual update operator",
            "input": "context, correlation features, current coordinate/residual state",
            "status": "original DROID; raw IMU is not a direct input",
        },
        {
            "id": "visual_heads",
            "name": "Visual update heads",
            "kind": "delta / residual / confidence-weight prediction",
            "status": "original DROID visual update path",
        },
        {
            "id": "dvi_confidence",
            "name": "DVI scalar IMU-confidence head",
            "kind": "1x1 convolution (128→1) followed by spatial average pooling and sigmoid",
            "output": "one scalar w^u per visual/IMU factor",
            "status": "present in checkpoint, inactive in this fixed-confidence 9DoF run",
        },
        {
            "id": "imu_preintegration",
            "name": "IMU preintegration",
            "kind": "frame-interval integration of raw gyro/accelerometer samples",
            "output": "delta rotation, velocity, position; covariance/information; bias Jacobians",
            "status": "added VI module",
        },
        {
            "id": "vi_init",
            "name": "Visual-inertial initialization",
            "kind": "visual poses + preintegrated IMU alignment",
            "output": "metric scale, gravity, global bias seed, initial velocities",
            "status": "enabled for this run",
        },
        {
            "id": "dba",
            "name": "Dense Bundle Adjustment",
            "kind": "joint visual factor graph optimization",
            "state": "pose (6 DoF) + velocity (3 DoF) + dense disparity",
            "status": "DROID DBA extended with 9DoF IMU residual; custom CUDA path",
        },
        {
            "id": "trajectory_filler",
            "name": "PoseTrajectoryFiller",
            "kind": "estimate poses for non-keyframes after final backend refinement",
            "status": "original DROID termination path",
        },
    ]

    nodes = [
        {"id": "images", "label": "Monocular image stream", "group": "input", "active": True},
        {"id": "raw_imu", "label": "Raw gyro + accelerometer stream", "group": "input", "active": True},
        {"id": "fnet", "label": "FNet\nfeature encoder", "group": "droid_visual", "active": True},
        {"id": "cnet", "label": "CNet\ncontext encoder", "group": "droid_visual", "active": True},
        {"id": "corr", "label": "Multi-scale\ncorrelation volume", "group": "droid_visual", "active": True},
        {"id": "convgru", "label": "ConvGRU\nUpdate Operator", "group": "droid_visual", "active": True},
        {"id": "visual_heads", "label": "Visual residual /\nupdate heads", "group": "droid_visual", "active": True},
        {"id": "dvi_confidence", "label": "Optional DVI confidence\n1×1 Conv → AvgPool → Sigmoid", "group": "optional_dvi", "active": False},
        {"id": "preintegrate", "label": "IMU preintegration\nΔR, Δv, Δp, Σ, J_b", "group": "imu", "active": True},
        {"id": "vi_init", "label": "VI initialization\nscale, gravity, bias, velocity", "group": "imu", "active": True},
        {"id": "visual_factor", "label": "Visual reprojection\nfactors", "group": "factor_graph", "active": True},
        {"id": "imu_factor", "label": "Covariance-normalized\nIMU residual", "group": "factor_graph", "active": True},
        {"id": "dba", "label": "9DoF Dense BA\npose + velocity + disparity", "group": "factor_graph", "active": True},
        {"id": "output", "label": "Metric pose trajectory\n+ dense reconstruction", "group": "output", "active": True},
    ]

    edges = [
        {"source": "images", "target": "fnet", "label": "images", "active": True},
        {"source": "images", "target": "cnet", "label": "images", "active": True},
        {"source": "fnet", "target": "corr", "label": "features", "active": True},
        {"source": "cnet", "target": "convgru", "label": "context + hidden state", "active": True},
        {"source": "corr", "target": "convgru", "label": "correspondence", "active": True},
        {"source": "convgru", "target": "visual_heads", "label": "recurrent visual state", "active": True},
        {"source": "visual_heads", "target": "visual_factor", "label": "visual residuals / weights", "active": True},
        {"source": "raw_imu", "target": "preintegrate", "label": "timestamped IMU samples", "active": True},
        {"source": "preintegrate", "target": "imu_factor", "label": "preintegrated delta + covariance", "active": True},
        {"source": "visual_heads", "target": "vi_init", "label": "visual pose window", "active": True},
        {"source": "preintegrate", "target": "vi_init", "label": "temporal IMU constraints", "active": True},
        {"source": "vi_init", "target": "dba", "label": "scale / gravity / velocity / fixed global bias", "active": True},
        {"source": "visual_factor", "target": "dba", "label": "visual objective", "active": True},
        {"source": "imu_factor", "target": "dba", "label": "0.002 · w^pu · w^u; w^u=1 here", "active": True},
        {"source": "dba", "target": "convgru", "label": "updated coordinates / residual state", "active": True},
        {"source": "dba", "target": "output", "label": "final pose / disparity", "active": True},
        {"source": "convgru", "target": "dvi_confidence", "label": "hidden feature", "active": False},
        {"source": "dvi_confidence", "target": "imu_factor", "label": "optional learned w^u", "active": False},
    ]

    runtime = {
        "primary_diagram_mode": "current 9DoF fixed-confidence monocular-inertial run",
        "checkpoint_reference": args.checkpoint,
        "weights_included": False,
        "framework": "PyTorch + custom CUDA extensions",
        "image_rate_hz": 10,
        "state_optimized_by_dba": ["pose: 6 DoF", "velocity: 3 DoF", "dense disparity"],
        "state_not_optimized_per_keyframe": ["accelerometer bias", "gyroscope bias"],
        "vi_initialization": True,
        "two_stage_15dof": False,
        "covariance_noise_model": True,
        "imu_global_weight": 0.002,
        "learned_imu_confidence_head_present": True,
        "learned_imu_confidence_enabled": False,
        "fixed_confidence_wu": 1.0,
        "important_note": "Raw IMU does not enter the ConvGRU directly; it enters DBA through preintegration.",
    }

    two_stage_overlay = {
        "mode": "optional two-stage 15DoF extension, not active in the primary diagram",
        "additional_state": ["accelerometer-bias correction: 3 DoF", "gyroscope-bias correction: 3 DoF"],
        "schedule": ["visual-oriented local DBA", "full 15DoF local DBA", "global full-state backend refinement"],
        "total_bias_expression": "b_i = b_global_from_VI_initialization + delta_b_i_local",
    }

    with h5py.File(args.output, "w") as handle:
        handle.attrs["format"] = "DVI-DROID architecture specification"
        handle.attrs["format_version"] = "1.0"
        handle.attrs["contains_weights"] = False
        handle.attrs["keras_loadable"] = False
        handle.attrs["purpose"] = "Diagram and reproducibility reference only"

        metadata = handle.create_group("metadata")
        metadata.attrs["title"] = "DROID-SLAM + IMU: 9DoF fixed-confidence architecture"
        metadata.attrs["checkpoint_reference"] = args.checkpoint
        metadata.attrs["notes"] = (
            "This file stores no neural weights and is not executable by Keras. "
            "Read graph/nodes and graph/edges to draw the architecture."
        )

        model = handle.create_group("model")
        dump_records(model, "layers_json", layers)

        graph = handle.create_group("graph")
        dump_records(graph, "nodes_json", nodes)
        dump_records(graph, "edges_json", edges)
        graph.attrs["primary_mode"] = runtime["primary_diagram_mode"]
        graph.attrs["active_edges_use_active_true"] = True
        graph.attrs["dashed_or_optional_edges_use_active_false"] = True

        runtime_group = handle.create_group("runtime")
        runtime_group.create_dataset(
            "configuration_json",
            data=json.dumps(runtime, ensure_ascii=False, indent=2),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        modes = handle.create_group("optional_modes")
        modes.create_dataset(
            "two_stage_15dof_json",
            data=json.dumps(two_stage_overlay, ensure_ascii=False, indent=2),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    print(f"wrote {args.output}")
    print(f"size_bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()

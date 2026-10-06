#!/usr/bin/env python3
"""Deterministic checks for runtime image/IMU input association safeguards."""

import csv
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "droid_slam"))

# The test invokes only MotionFilter's pure IMU helper methods. CorrBlock is
# imported while defining the class, but its CUDA extension is never called.
sys.modules.setdefault("droid_backends", types.SimpleNamespace())

from geom.imu_factor import predict_imu_rotation, rotvec_to_quat  # noqa: E402
from imu_contract import (  # noqa: E402
    load_frame_manifest,
    validate_frame_manifest_images,
    validate_frame_prior_mapping,
)
from imu_prior import load_imu_prior_csv  # noqa: E402
from motion_filter import MotionFilter  # noqa: E402


def imu_row(frame_index):
    return {
        "frame_id": frame_index + 10,
        "frame_index": frame_index,
        "timestamp_ns": (frame_index + 1) * 100,
        "filename": f"{frame_index}.webp",
        "dt": 0.1,
        "imu_count": 2,
        "imu_used_steps": 1,
        "imu_valid": 1,
        "imu_weight": 1.0,
        "imu_reason": "ok",
        "dr_x": 0.0,
        "dr_y": 0.0,
        "dr_z": 0.0,
        "dq_w": 1.0,
        "dq_x": 0.0,
        "dq_y": 0.0,
        "dq_z": 0.0,
        "dv_x": 0.0,
        "dv_y": 0.0,
        "dv_z": 0.0,
        "dp_x": 0.0,
        "dp_y": 0.0,
        "dp_z": 0.0,
    }


def assert_manifest_contract():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        image_dir = root / "images"
        image_dir.mkdir()
        for name in ("0.webp", "1.webp", "2.webp"):
            (image_dir / name).write_bytes(b"placeholder")

        frames_csv = root / "frames.csv"
        with frames_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["frame_id", "timestamp_ns", "timestamp_sec", "filename"],
            )
            writer.writeheader()
            # Deliberately shuffled input: timestamp ordering is the contract.
            writer.writerows(
                [
                    {"frame_id": 12, "timestamp_ns": 300, "timestamp_sec": 3e-7, "filename": "2.webp"},
                    {"frame_id": 10, "timestamp_ns": 100, "timestamp_sec": 1e-7, "filename": "0.webp"},
                    {"frame_id": 11, "timestamp_ns": 200, "timestamp_sec": 2e-7, "filename": "1.webp"},
                ]
            )

        frames = load_frame_manifest(frames_csv)
        assert [frame["filename"] for frame in frames] == ["0.webp", "1.webp", "2.webp"]
        assert [frame["frame_index"] for frame in frames] == [0, 1, 2]
        validate_frame_manifest_images(frames, image_dir)

        priors = {
            frame["frame_index"]: {
                "frame_id": frame["frame_id"],
                "timestamp_ns": frame["timestamp_ns"],
                "filename": frame["filename"],
            }
            for frame in frames
        }
        validate_frame_prior_mapping(frames, priors)

        priors[1]["filename"] = "wrong.webp"
        try:
            validate_frame_prior_mapping(frames, priors)
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched frame/IMU filename was accepted")


def assert_shared_prior_parser_preserves_contract_fields():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        prior_csv = root / "imu_prior.csv"
        with prior_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "frame_id",
                    "frame_index",
                    "timestamp_ns",
                    "filename",
                    "dt",
                    "imu_count",
                    "imu_valid",
                    "imu_weight",
                    "dr_x",
                    "dr_y",
                    "dr_z",
                    "dv_x",
                    "dv_y",
                    "dv_z",
                    "dp_x",
                    "dp_y",
                    "dp_z",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "frame_id": 10,
                        "frame_index": 0,
                        "timestamp_ns": 100,
                        "filename": "nested\\0.webp",
                        "dt": 0.0,
                        "imu_count": 0,
                        "imu_valid": 0,
                        "imu_weight": 0.0,
                        "dr_x": 0.0,
                        "dr_y": 0.0,
                        "dr_z": 0.0,
                        "dv_x": 0.0,
                        "dv_y": 0.0,
                        "dv_z": 0.0,
                        "dp_x": 0.0,
                        "dp_y": 0.0,
                        "dp_z": 0.0,
                    }
                ]
            )

        priors = load_imu_prior_csv(prior_csv)
        assert priors[0]["filename"] == "nested/0.webp"
        assert priors[0]["frame_id"] == 10
        assert priors[0]["timestamp_ns"] == 100


def make_filter(priors, previous_tstamp=0):
    motion_filter = MotionFilter.__new__(MotionFilter)
    motion_filter.video = SimpleNamespace(
        counter=SimpleNamespace(value=1),
        tstamp=torch.tensor([float(previous_tstamp)]),
    )
    motion_filter.imu_priors = priors
    motion_filter._reported_imu_interval_failures = set()
    return motion_filter


def assert_missing_interval_is_invalid():
    priors = {1: imu_row(1), 3: imu_row(3)}
    result = make_filter(priors).compose_imu_prior_for_selected_interval(
        3,
        priors[3],
    )
    assert result["imu_valid"] == 0
    assert result["imu_reason"] == "missing_interval_row"
    assert result["dt"] == 0.0


def assert_complete_interval_composes():
    priors = {index: imu_row(index) for index in (1, 2, 3)}
    result = make_filter(priors).compose_imu_prior_for_selected_interval(
        3,
        priors[3],
    )
    assert result["imu_valid"] == 1
    assert abs(result["dt"] - 0.3) < 1e-9
    assert result["frame_index"] == 3
    assert result["prev_frame_index"] == 0


def assert_pose_prior_uses_ba_convention():
    motion_filter = MotionFilter.__new__(MotionFilter)
    previous_quaternion = rotvec_to_quat(torch.tensor([0.2, -0.1, 0.05]))
    motion_filter.video = SimpleNamespace(
        counter=SimpleNamespace(value=1),
        poses=torch.cat([torch.zeros(3), previous_quaternion]).reshape(1, 7),
    )
    motion_filter.use_imu_pose_prior = True
    motion_filter.use_imu_translation_prior = False
    motion_filter.imu_pose_rot_weight = 1.0
    motion_filter.imu_pose_max_rot = 1.0
    motion_filter.imu_pose_compose_order = "dq_prev"
    motion_filter.imu_pose_full_inverse = True

    prior = imu_row(1)
    prior.update({"dr_x": 0.04, "dr_y": 0.03, "dr_z": -0.02})
    pose = motion_filter.build_imu_pose_prior(prior)
    expected = predict_imu_rotation(
        previous_quaternion,
        rotvec_to_quat(torch.tensor([0.04, 0.03, -0.02])),
        compose_order="dq_prev",
        inverse_imu=True,
    )
    assert torch.allclose(pose[3:], expected, atol=1e-6, rtol=1e-6)


def main():
    assert_manifest_contract()
    assert_shared_prior_parser_preserves_contract_fields()
    assert_missing_interval_is_invalid()
    assert_complete_interval_composes()
    assert_pose_prior_uses_ba_convention()
    print("IMU input-contract checks passed")


if __name__ == "__main__":
    main()

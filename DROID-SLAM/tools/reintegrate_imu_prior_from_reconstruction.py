#!/usr/bin/env python3
"""Rebuild an IMU prior at the bias estimated by a completed VI-DROID run.

This is an offline outer-loop operation, not a per-frame runtime shortcut.
The reconstruction stores the bias correction in camera coordinates because
the existing ``imu_prior.csv`` is already expressed in that frame.  Raw IMU
samples, however, live in the IMU sensor frame.  This utility maps the
estimated correction back through ``R_CI.T``, adds the nominal bias used by
the input prior, and regenerates the camera-rate preintegration CSV.

The next DROID invocation uses the rebuilt CSV and estimates only the
remaining bias correction.  That avoids applying a large first-order
preintegration correction indefinitely after visual-inertial initialization.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Tuple

import torch

from imu_preintegrate import (
    build_imu_calibration,
    build_imu_prior,
    mat3_transpose,
    mat3_vec,
)


def _three(values: Iterable[float], name: str) -> Tuple[float, float, float]:
    values = tuple(float(value) for value in values)
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return values


def _reconstruction_bias(blob: dict, key: str) -> Tuple[float, float, float]:
    value = blob.get(key)
    if value is None:
        raise KeyError(f"reconstruction does not contain {key}")
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    return _three(value, key)


def _nominal_sensor_bias(imu_prior: Path, prefix: str) -> Tuple[float, float, float]:
    """Read the sensor-frame nominal bias used to build an existing prior."""

    if not imu_prior.exists():
        return 0.0, 0.0, 0.0

    with imu_prior.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                return _three(
                    (
                        row.get(f"{prefix}_x", 0.0),
                        row.get(f"{prefix}_y", 0.0),
                        row.get(f"{prefix}_z", 0.0),
                    ),
                    f"{prefix} nominal bias",
                )
            except (TypeError, ValueError):
                continue
    return 0.0, 0.0, 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", required=True, help="saved demo.py reconstruction .pth")
    parser.add_argument("--frames", required=True, help="camera frames.csv used for the run")
    parser.add_argument("--imu", required=True, help="raw IMU csv used for the run")
    parser.add_argument(
        "--input_imu_prior",
        required=True,
        help="prior used by the completed run; its nominal sensor bias is preserved",
    )
    parser.add_argument("--output", required=True, help="output reintegrated imu_prior.csv")
    parser.add_argument("--cam_sensor_yaml", default=None)
    parser.add_argument("--imu_sensor_yaml", default=None)
    parser.add_argument(
        "--imu_to_cam_rotation",
        type=float,
        nargs=9,
        default=None,
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
    )
    parser.add_argument(
        "--imu_to_cam_translation",
        type=float,
        nargs=3,
        default=None,
        metavar=("TX", "TY", "TZ"),
    )
    parser.add_argument(
        "--allow_unaligned",
        action="store_true",
        help="allow reconstruction without a successful VI alignment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reconstruction = Path(args.reconstruction).resolve()
    frames = Path(args.frames).resolve()
    imu = Path(args.imu).resolve()
    input_prior = Path(args.input_imu_prior).resolve()
    output = Path(args.output).resolve()

    for path in (reconstruction, frames, imu, input_prior):
        if not path.exists():
            raise FileNotFoundError(path)

    blob = torch.load(reconstruction, map_location="cpu")
    if not isinstance(blob, dict):
        raise TypeError("reconstruction must be a dictionary written by demo.py")
    if not args.allow_unaligned and not bool(blob.get("imu_alignment_ready", 0)):
        raise RuntimeError(
            "reconstruction has no successful VI alignment; pass --allow_unaligned "
            "only for a deliberate diagnostic run"
        )

    accel_correction_camera = _reconstruction_bias(blob, "imu_global_bias_acc")
    gyro_correction_camera = _reconstruction_bias(blob, "imu_global_bias_gyro")
    calibration = build_imu_calibration(
        cam_sensor_yaml=(
            Path(args.cam_sensor_yaml).resolve() if args.cam_sensor_yaml else None
        ),
        imu_sensor_yaml=(
            Path(args.imu_sensor_yaml).resolve() if args.imu_sensor_yaml else None
        ),
        imu_to_cam_rotation=args.imu_to_cam_rotation,
        imu_to_cam_translation=args.imu_to_cam_translation,
    )

    # The global DBA bias is camera-frame.  ``build_imu_prior`` subtracts a
    # sensor-frame value before applying R_CI, hence the transpose here.
    R_ic = mat3_transpose(calibration.R_imu_to_camera)
    accel_correction_sensor = mat3_vec(R_ic, accel_correction_camera)
    gyro_correction_sensor = mat3_vec(R_ic, gyro_correction_camera)
    nominal_accel_sensor = _nominal_sensor_bias(input_prior, "acc_bias")
    nominal_gyro_sensor = _nominal_sensor_bias(input_prior, "gyro_bias")
    accel_sensor = tuple(
        nominal_accel_sensor[index] + accel_correction_sensor[index]
        for index in range(3)
    )
    gyro_sensor = tuple(
        nominal_gyro_sensor[index] + gyro_correction_sensor[index]
        for index in range(3)
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    corrected_calibration = build_imu_calibration(
        cam_sensor_yaml=(
            Path(args.cam_sensor_yaml).resolve() if args.cam_sensor_yaml else None
        ),
        imu_sensor_yaml=(
            Path(args.imu_sensor_yaml).resolve() if args.imu_sensor_yaml else None
        ),
        imu_to_cam_rotation=args.imu_to_cam_rotation,
        imu_to_cam_translation=args.imu_to_cam_translation,
        gyro_bias=gyro_sensor,
        acc_bias=accel_sensor,
    )
    result = build_imu_prior(frames, imu, output, calibration=corrected_calibration)

    print(f"[REINTEGRATE] source reconstruction={reconstruction}")
    print(
        "[REINTEGRATE] camera-frame correction "
        f"ba={accel_correction_camera}, bg={gyro_correction_camera}"
    )
    print(
        "[REINTEGRATE] total sensor-frame bias "
        f"ba={accel_sensor}, bg={gyro_sensor}"
    )
    print(
        f"[REINTEGRATE] wrote {output} "
        f"rows={result.get('priors', 'unknown')} calibration={corrected_calibration.source}"
    )


if __name__ == "__main__":
    main()

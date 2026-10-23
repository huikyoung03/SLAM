#!/usr/bin/env python3
"""Create rectified-camera IMU assets for DROID's original EuRoC protocol.

The original DROID evaluator rectifies raw cam0 with a non-identity rotation
before running the network.  A 15DoF IMU factor must be expressed in that
same rectified camera frame.  This tool writes a transformed camera
``sensor.yaml`` and regenerates a central-difference IMU prior; it never
copies or rewrites input images or the input frame manifest.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from droid_slam.data_readers.euroc_protocol import (  # noqa: E402
    EUROC_CAM0_P,
    rectified_t_body_cam,
)
from tools.imu_preintegrate import (  # noqa: E402
    build_imu_calibration,
    build_imu_prior,
    read_yaml_matrix4,
)


def _replace_vector(text: str, key: str, values) -> str:
    value_text = ", ".join(f"{float(value):.17g}" for value in values)
    pattern = rf"(^\s*{re.escape(key)}\s*:\s*)\[[^\]]*\]"
    result, count = re.subn(pattern, rf"\g<1>[{value_text}]", text, flags=re.M)
    if count != 1:
        raise ValueError(f"expected one {key}: [...] entry, found {count}")
    return result


def _replace_t_bs(text: str, matrix: np.ndarray) -> str:
    values = ", ".join(f"{float(value):.17g}" for value in matrix.reshape(-1))
    pattern = r"(T_BS\s*:\s*.*?\bdata\s*:\s*)\[[^\]]*\]"
    result, count = re.subn(pattern, rf"\g<1>[{values}]", text, flags=re.S)
    if count != 1:
        raise ValueError(f"expected one T_BS.data entry, found {count}")
    return result


def _write_or_refuse(path: Path, text: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared",
        required=True,
        help="prepared EuRoC directory containing frames.csv, imu.csv, cam0_sensor.yaml, imu0_sensor.yaml",
    )
    parser.add_argument("--output", required=True, help="new strict-protocol asset directory")
    parser.add_argument("--force", action="store_true", help="replace generated assets in --output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = Path(args.prepared).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    inputs = {
        "frames": prepared / "frames.csv",
        "imu": prepared / "imu.csv",
        "camera": prepared / "cam0_sensor.yaml",
        "imu_yaml": prepared / "imu0_sensor.yaml",
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")

    source_text = inputs["camera"].read_text(encoding="utf-8")
    T_body_cam = np.asarray(read_yaml_matrix4(inputs["camera"]), dtype=np.float64)
    strict_T_body_cam = rectified_t_body_cam(T_body_cam)

    strict_yaml = _replace_t_bs(source_text, strict_T_body_cam)
    strict_yaml = _replace_vector(
        strict_yaml,
        "intrinsics",
        (EUROC_CAM0_P[0, 0], EUROC_CAM0_P[1, 1], EUROC_CAM0_P[0, 2], EUROC_CAM0_P[1, 2]),
    )
    strict_yaml = _replace_vector(strict_yaml, "distortion_coefficients", (0.0, 0.0, 0.0, 0.0))
    strict_yaml = _replace_vector(strict_yaml, "resolution", (752, 480))

    strict_camera_yaml = output / "cam0_sensor_rectified.yaml"
    strict_prior = output / "imu_prior_rectified_fd_jacobian.csv"
    strict_calib = output / "calib_rectified.txt"
    protocol = output / "protocol.txt"

    _write_or_refuse(strict_camera_yaml, strict_yaml, args.force)
    _write_or_refuse(
        strict_calib,
        " ".join(
            f"{value:.17g}"
            for value in (
                EUROC_CAM0_P[0, 0],
                EUROC_CAM0_P[1, 1],
                EUROC_CAM0_P[0, 2],
                EUROC_CAM0_P[1, 2],
            )
        )
        + "\n",
        args.force,
    )

    if strict_prior.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {strict_prior}; pass --force")
    calibration = build_imu_calibration(
        cam_sensor_yaml=strict_camera_yaml,
        imu_sensor_yaml=inputs["imu_yaml"],
    )
    result = build_imu_prior(
        inputs["frames"],
        inputs["imu"],
        strict_prior,
        calibration=calibration,
    )
    _write_or_refuse(
        protocol,
        "\n".join(
            [
                "protocol=droid_original_euroc_rectified_cam0_v1",
                f"prepared={prepared}",
                f"camera_yaml={strict_camera_yaml}",
                f"imu_prior={strict_prior}",
                "images=raw_cam0; rectification_is_applied_on_the_fly",
                "rectification=official_evaluation_scripts/test_euroc.py_cam0",
                "image_size=320x512_at_runtime",
                "imu_prior_bias_jacobians=central_difference",
                "",
            ]
        ),
        args.force,
    )
    print(f"[STRICT ASSETS] output={output}")
    print(f"[STRICT ASSETS] camera_yaml={strict_camera_yaml}")
    print(f"[STRICT ASSETS] imu_prior={strict_prior}")
    print(f"[STRICT ASSETS] priors={result.get('priors', 'unknown')}")


if __name__ == "__main__":
    main()

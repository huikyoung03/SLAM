#!/usr/bin/env python3
"""Convert the standardized EuRoC MCAP mirror to the cam0/IMU layout we train on.

The ``gpdaniels/slam-datasets`` mirror stores EuRoC as ROS 2 CDR messages:

* ``/sensor/image_01``: mono8 left-camera frames;
* ``/sensor/imu_01``: angular velocity and linear acceleration;
* ``/tf``: root->ego ground truth plus ego->sensor calibration transforms.

This utility writes only the left camera because the DROID EuRoC reader is
monocular.  It deliberately leaves the MCAP input untouched and refuses to
overwrite an already converted sequence.

Requirements (in the Python environment that runs this script):
``mcap``, ``mcap-ros2-support``, ``numpy``, and ``opencv-python``.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


# These are the ADIS16448 values supplied with the official EuRoC ``imu0``
# sensor.yaml.  The standardized MCAP stores measurements and transforms but
# not the noise model, so retaining the official values keeps the downstream
# preintegration covariance consistent with the native EuRoC release.
EUROC_IMU_NOISE = {
    "gyroscope_noise_density": 1.6968e-04,
    "gyroscope_random_walk": 1.9393e-05,
    "accelerometer_noise_density": 2.0000e-03,
    "accelerometer_random_walk": 3.0000e-03,
}


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def transform_matrix(transform) -> np.ndarray:
    """Return the sensor-to-parent transform carried by a ROS TF message."""

    q = transform.rotation
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("encountered a zero-norm TF quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix[:3, 3] = (float(transform.translation.x), float(transform.translation.y), float(transform.translation.z))
    return matrix


def yaml_matrix(matrix: np.ndarray) -> str:
    values = ", ".join(f"{value:.16g}" for value in matrix.reshape(-1))
    return "T_BS:\n  cols: 4\n  rows: 4\n  data: [" + values + "]\n"


def yaml_vector(values: Iterable[float]) -> str:
    return ", ".join(f"{float(value):.16g}" for value in values)


def write_camera_yaml(path: Path, camera_info, t_body_sensor: np.ndarray) -> None:
    k = camera_info.k
    text = (
        "# Generated from the standardized EuRoC MCAP mirror.\n"
        "sensor_type: camera\n"
        f"{yaml_matrix(t_body_sensor)}"
        "rate_hz: 20\n"
        f"resolution: [{int(camera_info.width)}, {int(camera_info.height)}]\n"
        "camera_model: pinhole\n"
        f"intrinsics: [{yaml_vector((k[0], k[4], k[2], k[5]))}]\n"
        "distortion_model: radtan\n"
        f"distortion_coefficients: [{yaml_vector(camera_info.d)}]\n"
    )
    path.write_text(text, encoding="utf-8")


def write_imu_yaml(path: Path, t_body_sensor: np.ndarray) -> None:
    noise = "\n".join(f"{key}: {value:.16g}" for key, value in EUROC_IMU_NOISE.items())
    text = (
        "# Official EuRoC ADIS16448 noise values; transform from MCAP /tf.\n"
        "sensor_type: imu\n"
        f"{yaml_matrix(t_body_sensor)}"
        "rate_hz: 200\n"
        f"{noise}\n"
    )
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="input .mcap file")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="sequence directory to create; its mav0/ child will be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    mav0 = output / "mav0"

    if not source.is_file():
        raise FileNotFoundError(source)
    if (mav0 / "cam0" / "data.csv").exists():
        raise FileExistsError(f"refusing to overwrite existing conversion: {mav0}")
    if mav0.exists() and any(mav0.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty directory: {mav0}")

    cam_data_dir = mav0 / "cam0" / "data"
    cam_data_dir.mkdir(parents=True, exist_ok=False)
    imu_dir = mav0 / "imu0"
    gt_dir = mav0 / "state_groundtruth_estimate0"
    imu_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    images: "OrderedDict[int, str]" = OrderedDict()
    imu_rows = []
    gt_rows: "OrderedDict[int, Tuple[float, float, float, float, float, float, float]]" = OrderedDict()
    camera_info = None
    t_body_camera = None
    t_body_imu = None
    decoder_factory = DecoderFactory()

    with source.open("rb") as stream:
        reader = make_reader(stream)
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            if topic not in {"/sensor/image_01", "/sensor/image_01/camera_info", "/sensor/imu_01", "/tf"}:
                continue

            decoder = decoder_factory.decoder_for(channel.message_encoding, schema)
            if decoder is None:
                raise ValueError(f"unsupported MCAP encoding for {topic}: {channel.message_encoding}")
            decoded = decoder(message.data)

            if topic == "/sensor/image_01":
                timestamp = stamp_ns(decoded.header.stamp)
                image = np.frombuffer(decoded.data, dtype=np.uint8).reshape(decoded.height, decoded.width)
                filename = f"{timestamp}.png"
                if not cv2.imwrite(str(cam_data_dir / filename), image):
                    raise RuntimeError(f"failed to write image {filename}")
                images[timestamp] = filename
            elif topic == "/sensor/image_01/camera_info":
                camera_info = decoded
            elif topic == "/sensor/imu_01":
                timestamp = stamp_ns(decoded.header.stamp)
                gyro = decoded.angular_velocity
                accel = decoded.linear_acceleration
                imu_rows.append(
                    (timestamp, gyro.x, gyro.y, gyro.z, accel.x, accel.y, accel.z)
                )
            else:
                for transform in decoded.transforms:
                    parent = transform.header.frame_id
                    child = transform.child_frame_id
                    if parent == "ego" and child == "sensor/image_01":
                        t_body_camera = transform_matrix(transform.transform)
                    elif parent == "ego" and child == "sensor/imu_01":
                        t_body_imu = transform_matrix(transform.transform)
                    elif parent == "root" and child == "ego":
                        timestamp = stamp_ns(transform.header.stamp)
                        translation = transform.transform.translation
                        rotation = transform.transform.rotation
                        gt_rows[timestamp] = (
                            float(translation.x),
                            float(translation.y),
                            float(translation.z),
                            float(rotation.w),
                            float(rotation.x),
                            float(rotation.y),
                            float(rotation.z),
                        )

    if camera_info is None or t_body_camera is None or t_body_imu is None:
        raise ValueError("MCAP is missing camera info or required calibration transforms")
    if not images or not imu_rows or not gt_rows:
        raise ValueError("MCAP did not yield left-camera images, IMU samples, and ground truth")

    with (mav0 / "cam0" / "data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["#timestamp [ns]", "filename"])
        writer.writerows(images.items())

    with (imu_dir / "data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "#timestamp [ns]",
            "w_RS_S_x [rad s^-1]",
            "w_RS_S_y [rad s^-1]",
            "w_RS_S_z [rad s^-1]",
            "a_RS_S_x [m s^-2]",
            "a_RS_S_y [m s^-2]",
            "a_RS_S_z [m s^-2]",
        ])
        writer.writerows(imu_rows)

    with (gt_dir / "data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "#timestamp [ns]",
            "p_RS_R_x [m]",
            "p_RS_R_y [m]",
            "p_RS_R_z [m]",
            "q_RS_w []",
            "q_RS_x []",
            "q_RS_y []",
            "q_RS_z []",
        ])
        for timestamp, pose in gt_rows.items():
            writer.writerow((timestamp, *pose))

    write_camera_yaml(mav0 / "cam0" / "sensor.yaml", camera_info, t_body_camera)
    write_imu_yaml(imu_dir / "sensor.yaml", t_body_imu)

    missing_gt = [timestamp for timestamp in images if timestamp not in gt_rows]
    if missing_gt:
        raise ValueError(f"{len(missing_gt)} camera frames do not have matching root->ego GT")
    print(
        f"Converted {source.name}: images={len(images)}, imu={len(imu_rows)}, "
        f"gt={len(gt_rows)}, output={mav0}"
    )


if __name__ == "__main__":
    main()

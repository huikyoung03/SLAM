"""The fixed EuRoC cam0 protocol used by the original DROID evaluator.

The official ``evaluation_scripts/test_euroc.py`` does not feed raw EuRoC
cam0 pixels to the network.  It rectifies them with the fixed cam0 calibration
below and then resizes to 320 x 512.  Keeping these constants in one small
module makes the learned VI path use the *same* camera coordinates as the
published visual-DROID baseline.

``R_RECT`` maps a vector expressed in the raw cam0 coordinate system to the
rectified cam0 coordinate system.  Consequently an IMU-to-camera transform
must be left-multiplied by ``R_RECT`` and ``T_BS`` (camera -> body) must be
right-multiplied by ``R_RECT.T``.
"""

from __future__ import annotations

import cv2
import numpy as np


EUROC_CAM0_SIZE = (752, 480)  # (width, height)
EUROC_DROID_IMAGE_SIZE = (320, 512)  # (height, width)

EUROC_CAM0_K = np.array(
    [[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
EUROC_CAM0_DISTORTION = np.array(
    [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0],
    dtype=np.float64,
)
EUROC_CAM0_R_RECT = np.array(
    [
        [0.999966347530033, -0.001422739138722922, 0.008079580483432283],
        [0.001365741834644127, 0.9999741760894847, 0.007055629199258132],
        [-0.008089410156878961, -0.007044357138835809, 0.9999424675829176],
    ],
    dtype=np.float64,
)
EUROC_CAM0_P = np.array(
    [
        [435.2046959714599, 0.0, 367.4517211914062],
        [0.0, 435.2046959714599, 252.2008514404297],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def strict_cam0_intrinsics(image_size=EUROC_DROID_IMAGE_SIZE):
    """Return official rectified intrinsics scaled to ``(height, width)``."""

    height, width = (int(image_size[0]), int(image_size[1]))
    scale_x = width / float(EUROC_CAM0_SIZE[0])
    scale_y = height / float(EUROC_CAM0_SIZE[1])
    return np.array(
        [
            EUROC_CAM0_P[0, 0] * scale_x,
            EUROC_CAM0_P[1, 1] * scale_y,
            EUROC_CAM0_P[0, 2] * scale_x,
            EUROC_CAM0_P[1, 2] * scale_y,
        ],
        dtype=np.float32,
    )


def strict_cam0_rectify_map(image_shape=None):
    """Build the exact OpenCV map used by the original DROID evaluator."""

    if image_shape is not None:
        height, width = image_shape[:2]
        if (width, height) != EUROC_CAM0_SIZE:
            raise ValueError(
                "the original EuRoC protocol requires raw cam0 images at "
                f"{EUROC_CAM0_SIZE[0]}x{EUROC_CAM0_SIZE[1]}, got {width}x{height}"
            )
    return cv2.initUndistortRectifyMap(
        EUROC_CAM0_K,
        EUROC_CAM0_DISTORTION,
        EUROC_CAM0_R_RECT,
        EUROC_CAM0_P,
        EUROC_CAM0_SIZE,
        cv2.CV_32F,
    )


def strict_rectify_cam0(image, rectify_map=None):
    """Rectify one raw EuRoC cam0 BGR image in the published coordinate frame."""

    if image is None:
        raise ValueError("cannot rectify an empty EuRoC image")
    if rectify_map is None:
        rectify_map = strict_cam0_rectify_map(image.shape)
    return cv2.remap(image, rectify_map[0], rectify_map[1], interpolation=cv2.INTER_LINEAR)


def rectified_t_body_cam(T_body_cam):
    """Convert EuRoC's camera-to-body transform to rectified cam0 coordinates."""

    result = np.asarray(T_body_cam, dtype=np.float64).copy()
    if result.shape != (4, 4):
        raise ValueError(f"expected 4x4 T_BS, got {result.shape}")
    result[:3, :3] = result[:3, :3] @ EUROC_CAM0_R_RECT.T
    return result


def rectified_imu_calibration(calibration, source_suffix="euroc_cam0_rectified"):
    """Express an existing static IMU calibration in rectified cam0 coordinates."""

    from dataclasses import replace

    R_ci = np.asarray(calibration.R_imu_to_camera, dtype=np.float64)
    t_ci = np.asarray(calibration.t_imu_to_camera, dtype=np.float64)
    if R_ci.shape != (3, 3) or t_ci.shape != (3,):
        raise ValueError("IMU calibration must contain a 3x3 R_CI and 3-vector t_CI")

    return replace(
        calibration,
        R_imu_to_camera=tuple(
            tuple(float(value) for value in row)
            for row in (EUROC_CAM0_R_RECT @ R_ci)
        ),
        t_imu_to_camera=tuple(float(value) for value in (EUROC_CAM0_R_RECT @ t_ci)),
        source=f"{calibration.source}|{source_suffix}",
    )

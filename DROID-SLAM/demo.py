import sys

# droid_slam 폴더를 Python import 경로에 추가
# 이렇게 해야 droid_slam 내부 모듈인 droid.py, droid_async.py 등을 import할 수 있음
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
import csv
import hashlib
import math
from pathlib import Path

from torch.multiprocessing import Process

# DROID-SLAM 기본 동기식 실행 클래스
from droid import Droid

# DROID-SLAM 비동기 실행 클래스
# frontend와 backend를 분리해서 실행할 때 사용
from droid_async import DroidAsync

try:
    from data_readers.imu_preintegration import (
        build_imu_prior_csv,
        resolve_imu_calibration,
    )
except Exception:
    build_imu_prior_csv = None
    resolve_imu_calibration = None

try:
    from imu_contract import (
        frame_manifest_digest,
        load_frame_manifest,
        selected_frame_records,
        validate_frame_manifest_images,
        validate_frame_prior_mapping,
    )
except ImportError:
    from droid_slam.imu_contract import (
        frame_manifest_digest,
        load_frame_manifest,
        selected_frame_records,
        validate_frame_manifest_images,
        validate_frame_prior_mapping,
    )

try:
    from imu_prior import load_imu_prior_csv
except ImportError:
    from droid_slam.imu_prior import load_imu_prior_csv

try:
    from data_readers.euroc_protocol import (
        EUROC_DROID_IMAGE_SIZE,
        strict_cam0_intrinsics,
        strict_cam0_rectify_map,
        strict_rectify_cam0,
    )
except ImportError:
    from droid_slam.data_readers.euroc_protocol import (
        EUROC_DROID_IMAGE_SIZE,
        strict_cam0_intrinsics,
        strict_cam0_rectify_map,
        strict_rectify_cam0,
    )

import torch.nn.functional as F


'''
DROID-SLAM을 실제로 실행하는 entry point야.
이미지 폴더와 카메라 캘리브레이션 파일을 입력으로 받아서,
프레임을 하나씩 읽고 Droid 또는 DroidAsync에 넘겨 SLAM을 수행해.
마지막에는 trajectory를 얻고, 옵션이 있으면 reconstruction 결과를 .pth로 저장해.


이미지 폴더
   ↓
image_stream()
   ↓
resize + undistort + intrinsics 보정
   ↓
Droid 또는 DroidAsync 생성
   ↓
droid.track(t, image, intrinsics)
   ↓
MotionFilter
   ↓
DroidFrontend
   ↓
Backend / AsyncBackend
   ↓
droid.terminate()
   ↓
trajectory 반환
   ↓
save_reconstruction()



'''


def show_image(image):
    """
    현재 입력 프레임을 OpenCV 창으로 보여주는 함수.

    입력:
        image:
            torch tensor 형태 이미지.
            shape: [3, H, W]

    처리:
        1. [3, H, W] -> [H, W, 3]으로 변환
        2. CPU numpy array로 변환
        3. OpenCV imshow로 표시
    """

    # PyTorch image tensor는 보통 [C, H, W] 형태
    # OpenCV는 [H, W, C] 형태를 사용하므로 permute
    image = image.permute(1, 2, 0).cpu().numpy()

    # 0~255 이미지를 0~1 범위로 나누어 화면 표시
    cv2.imshow('image', image / 255.0)

    # waitKey(1)을 호출해야 OpenCV 창이 갱신됨
    cv2.waitKey(1)


def load_imu_priors(imu_prior_path):
    """Load the same checked IMU-prior schema used by dataset readers."""

    if imu_prior_path is None:
        return {}

    imu_prior_path = Path(imu_prior_path)
    if not imu_prior_path.exists():
        print(f"[IMU] imu_prior.csv not found: {imu_prior_path}")
        return {}

    return load_imu_prior_csv(imu_prior_path, report=True)


def resolve_runtime_frame_records(args, imu_priors):
    """Use the preintegration manifest as the runtime image/IMU contract."""

    if not imu_priors:
        return None, None

    explicit = getattr(args, "imu_frame_manifest", None)
    build_manifest = getattr(args, "imu_frames", None)
    inferred = None
    if getattr(args, "imu_prior", None):
        inferred = Path(args.imu_prior).expanduser().resolve().parent / "frames.csv"

    candidates = [
        ("--imu_frame_manifest", explicit),
        ("--imu_frames", build_manifest),
        ("imu_prior sibling frames.csv", inferred),
    ]
    manifest_path = None
    source = None
    for candidate_source, candidate in candidates:
        if candidate in (None, ""):
            continue
        candidate_path = Path(candidate).expanduser()
        if candidate_path.is_file():
            manifest_path = candidate_path.resolve()
            source = candidate_source
            break
        if candidate_source != "imu_prior sibling frames.csv":
            raise FileNotFoundError(
                f"{candidate_source} does not exist: {candidate_path}"
            )

    if manifest_path is None:
        if getattr(args, "allow_legacy_imu_frame_mapping", False):
            print(
                "[IMU WARNING] no frames.csv mapping is available; using legacy "
                "sorted-directory indices because --allow_legacy_imu_frame_mapping was set"
            )
            return None, None
        raise FileNotFoundError(
            "IMU runtime requires the frames.csv used to create imu_prior.csv. "
            "Pass --imu_frame_manifest, retain frames.csv next to imu_prior.csv, "
            "or explicitly opt in to the unsafe legacy mapping with "
            "--allow_legacy_imu_frame_mapping."
        )

    frame_records = load_frame_manifest(manifest_path)
    validate_frame_manifest_images(frame_records, Path(args.imagedir))
    validate_frame_prior_mapping(frame_records, imu_priors)
    print(
        f"[IMU] frame mapping verified: source={source}, frames={len(frame_records)}, "
        f"manifest={manifest_path}"
    )
    return str(manifest_path), frame_records


def resolve_runtime_imu_calibration(args):
    """Resolve the calibration the runtime believes was used for preintegration."""

    if resolve_imu_calibration is None:
        return None

    has_runtime_calibration = any(
        value is not None
        for value in (
            args.cam_sensor_yaml,
            args.imu_sensor_yaml,
            args.imu_to_cam_rotation,
            args.imu_to_cam_translation,
        )
    )
    if not has_runtime_calibration:
        return None

    try:
        return resolve_imu_calibration(
            cam_sensor_yaml=args.cam_sensor_yaml,
            imu_sensor_yaml=args.imu_sensor_yaml,
            imu_to_cam_rotation=args.imu_to_cam_rotation,
            imu_to_cam_translation=args.imu_to_cam_translation,
        )
    except Exception as exc:
        print(f"[IMU WARNING] failed to resolve runtime calibration: {exc}")
        return None


def _prior_calibration_fingerprint(imu_priors, runtime_calibration=None):
    """Return a stable provenance digest for calibration-sensitive IMU inputs."""

    if runtime_calibration is not None:
        rotation = np.asarray(runtime_calibration.R_imu_to_camera, dtype=np.float64)
        translation = np.asarray(
            runtime_calibration.t_imu_to_camera,
            dtype=np.float64,
        )
        values = [
            *rotation.reshape(-1).tolist(),
            *translation.tolist(),
            float(runtime_calibration.gyro_noise_density),
            float(runtime_calibration.gyro_random_walk),
            float(runtime_calibration.accel_noise_density),
            float(runtime_calibration.accel_random_walk),
            float(runtime_calibration.extrinsic_applied),
            float(runtime_calibration.translation_extrinsic_applied),
        ]
    elif imu_priors:
        row = next(iter(imu_priors.values()))
        values = [
            *[
                float(row.get(f"R_cam_imu_{i}{j}", 0.0))
                for i in range(3)
                for j in range(3)
            ],
            *[
                float(row.get(key, 0.0))
                for key in ("imu_to_cam_tx", "imu_to_cam_ty", "imu_to_cam_tz")
            ],
            *[
                float(row.get(key, 0.0))
                for key in (
                    "gyro_noise_density",
                    "gyro_random_walk",
                    "accel_noise_density",
                    "accel_random_walk",
                )
            ],
            float(row.get("imu_extrinsic_applied", 0)),
            float(row.get("imu_translation_extrinsic_applied", 0)),
        ]
    else:
        return None

    digest = hashlib.sha256()
    digest.update(b"droid-imu-calibration-v1\0")
    digest.update(
        ",".join(f"{value:.12e}" for value in values).encode("ascii")
    )
    return digest.hexdigest()


def validate_imu_calibration_contract(args, imu_priors, runtime_calibration):
    """Reject a preintegrated CSV generated with a different IMU calibration."""

    fingerprint = _prior_calibration_fingerprint(
        imu_priors,
        runtime_calibration,
    )
    if not imu_priors or runtime_calibration is None:
        return fingerprint

    row = next(iter(imu_priors.values()))
    errors = []
    tolerance = 1e-6

    if bool(row.get("imu_extrinsic_applied", 0)) != bool(
        runtime_calibration.extrinsic_applied
    ):
        errors.append("R_CI calibration-applied flag differs from imu_prior.csv")
    if bool(row.get("imu_translation_extrinsic_applied", 0)) != bool(
        runtime_calibration.translation_extrinsic_applied
    ):
        errors.append("t_CI calibration-applied flag differs from imu_prior.csv")

    if int(row.get("imu_calibration_rotation_present", 0)):
        prior_rotation = np.asarray(
            [
                row.get(f"R_cam_imu_{i}{j}", 0.0)
                for i in range(3)
                for j in range(3)
            ],
            dtype=np.float64,
        ).reshape(3, 3)
        runtime_rotation = np.asarray(
            runtime_calibration.R_imu_to_camera,
            dtype=np.float64,
        )
        if not np.allclose(prior_rotation, runtime_rotation, atol=tolerance, rtol=0.0):
            errors.append("R_CI rotation differs from the calibration used by imu_prior.csv")

    if int(row.get("imu_calibration_translation_present", 0)):
        prior_translation = np.asarray(
            [
                row.get("imu_to_cam_tx", 0.0),
                row.get("imu_to_cam_ty", 0.0),
                row.get("imu_to_cam_tz", 0.0),
            ],
            dtype=np.float64,
        )
        runtime_translation = np.asarray(
            runtime_calibration.t_imu_to_camera,
            dtype=np.float64,
        )
        if not np.allclose(
            prior_translation,
            runtime_translation,
            atol=tolerance,
            rtol=0.0,
        ):
            errors.append("t_CI lever arm differs from the calibration used by imu_prior.csv")

    if int(row.get("imu_calibration_noise_present", 0)):
        prior_noise = np.asarray(
            [
                row.get("gyro_noise_density", 0.0),
                row.get("gyro_random_walk", 0.0),
                row.get("accel_noise_density", 0.0),
                row.get("accel_random_walk", 0.0),
            ],
            dtype=np.float64,
        )
        runtime_noise = np.asarray(
            [
                runtime_calibration.gyro_noise_density,
                runtime_calibration.gyro_random_walk,
                runtime_calibration.accel_noise_density,
                runtime_calibration.accel_random_walk,
            ],
            dtype=np.float64,
        )
        if not np.allclose(prior_noise, runtime_noise, atol=1e-12, rtol=1e-6):
            errors.append("IMU noise/random-walk values differ from imu_prior.csv")

    if errors:
        message = (
            "IMU calibration mismatch. Rebuild imu_prior.csv with the active "
            "sensor calibration instead of mixing preintegration and runtime settings:\n  - "
            + "\n  - ".join(errors)
        )
        if getattr(args, "allow_imu_calibration_mismatch", False):
            print(f"[IMU WARNING] {message}")
        else:
            raise ValueError(message)

    return fingerprint


def validate_imu_bias_jacobian_contract(args, imu_priors):
    """Require real bias derivatives before opening a full VI state optimizer.

    A rotation-only prior can fall back to ``-dt I`` for legacy CSV files, but
    a 9/15DoF factor corrects rotation, velocity, and position using all five
    preintegration Jacobians. Silently replacing those terms with zeros lets a
    bias state optimize against a different measurement model.
    """

    if not getattr(args, "use_full_imu_ba", False) or not imu_priors:
        return

    valid_rows = [
        row for row in imu_priors.values()
        if int(row.get("imu_valid", 0)) != 0 and float(row.get("dt", 0.0)) > 0.0
    ]
    if not valid_rows:
        raise ValueError("full IMU BA requires at least one valid IMU prior row")

    missing = sum(
        not bool(row.get("bias_jacobian_present", 0)) for row in valid_rows
    )
    if missing:
        raise ValueError(
            "full IMU BA requires imu_prior.csv with J_R_bg/J_v_ba/J_v_bg/"
            "J_p_ba/J_p_bg for every valid interval. The supplied prior uses "
            f"legacy fallback Jacobians on {missing}/{len(valid_rows)} rows. "
            "Regenerate it with tools/imu_preintegrate.py before running 9DoF "
            "or 15DoF visual-inertial DBA."
        )

    methods = {
        str(row.get("bias_jacobian_method", "")).strip()
        for row in valid_rows
    }
    if methods != {"central_difference"}:
        print(
            "[IMU WARNING] full IMU BA received non-central-difference bias "
            f"Jacobians: {sorted(methods)}"
        )


def resolve_imu_camera_translation(args, imu_priors, runtime_calibration=None):
    """Resolve the IMU-origin lever arm used by full position/velocity factors.

    The prior CSV stores this calibration when it was generated by the current
    preintegration tool.  Sensor YAML or an explicit CLI value takes precedence
    so that legacy CSV files can be reused without silently falling back to a
    camera-centre model.
    """

    if runtime_calibration is not None:
        translation = np.asarray(
            runtime_calibration.t_imu_to_camera,
            dtype=np.float32,
        )
        if np.isfinite(translation).all():
            return (
                translation.tolist(),
                str(runtime_calibration.source),
                bool(runtime_calibration.translation_extrinsic_applied),
            )

    for row in imu_priors.values():
        translation = np.asarray(
            [
                row.get("imu_to_cam_tx", 0.0),
                row.get("imu_to_cam_ty", 0.0),
                row.get("imu_to_cam_tz", 0.0),
            ],
            dtype=np.float32,
        )
        if bool(row.get("imu_translation_extrinsic_applied", 0)) and np.isfinite(translation).all():
            return translation.tolist(), row.get("imu_calibration_source", "imu_prior.csv"), True

    return [0.0, 0.0, 0.0], "zero_lever_arm_fallback", False


def estimate_imu_info_reference(imu_priors):
    """Return the nominal raw-camera IMU information for one session.

    MotionFilter can compose many camera-rate preintegrations into a single
    keyframe interval.  Normalizing information by that local keyframe window
    would erase the uncertainty growth of the composed interval.  This
    reference is therefore computed once from the original valid camera-rate
    rows before DROID starts filtering frames.
    """

    if not imu_priors:
        return None

    references = []
    for key in ("rot_info", "vel_info", "pos_info"):
        values = []
        for row in imu_priors.values():
            if int(row.get("imu_valid", 1)) == 0:
                continue
            value = float(row.get(key, 0.0))
            if np.isfinite(value) and value > 1e-12:
                values.append(value)
        if not values:
            return None
        references.append(float(np.median(np.asarray(values, dtype=np.float64))))

    return references


def get_imu_prior_for_frame(imu_priors, frame_index):
    if not imu_priors:
        return None

    return imu_priors.get(int(frame_index))


def estimate_gravity_from_imu_priors(
    imu_priors,
    max_frames=80,
    min_norm=6.0,
    max_norm=12.5,
):
    """
    Estimate an initial gravity vector from preintegrated delta-velocity rows.

    This assumes the early trajectory is not undergoing extreme linear
    acceleration. It is dataset-neutral and only uses the generic imu_prior.csv
    columns:

        g_world ~= -Delta v_imu / Delta t

    The result is an initial guess for the full IMU residual. It is not a
    replacement for a full visual-inertial initialization.
    """

    if not imu_priors:
        return None

    vectors = []
    for frame_index in sorted(imu_priors.keys()):
        row = imu_priors[frame_index]
        if int(row.get("imu_valid", 1)) == 0:
            continue

        dt = float(row.get("dt", 0.0))
        if dt <= 1e-6:
            continue

        dv = np.asarray([
            float(row.get("dv_x", 0.0)),
            float(row.get("dv_y", 0.0)),
            float(row.get("dv_z", 0.0)),
        ], dtype=np.float64)
        acc_norm = float(np.linalg.norm(dv) / dt)
        if min_norm > 0.0 and acc_norm < float(min_norm):
            continue
        if max_norm > 0.0 and acc_norm > float(max_norm):
            continue

        vectors.append(-dv / dt)
        if len(vectors) >= int(max_frames):
            break

    if not vectors:
        return None

    gravity = np.mean(np.stack(vectors, axis=0), axis=0)
    if not np.isfinite(gravity).all():
        return None

    return gravity.astype(np.float32).tolist()


def maybe_build_imu_prior_from_raw(args):
    """
    Optional terminal workflow:

        demo.py ... --imu_frames frames.csv --imu_csv imu.csv

    This keeps EuRoC and phone-capture handling outside the SLAM core. Any data
    source only needs to provide the generic frames/imu CSV schema.
    """

    if args.imu_csv is None and not args.build_imu_prior:
        return args.imu_prior

    if args.imu_frames is None or args.imu_csv is None:
        print("[IMU WARNING] --imu_csv/--build_imu_prior requires both --imu_frames and --imu_csv.")
        return args.imu_prior

    if build_imu_prior_csv is None:
        print("[IMU WARNING] failed to import IMU preintegration builder; using existing --imu_prior.")
        return args.imu_prior

    output = args.imu_prior_output or args.imu_prior
    if output is None:
        output = str(Path(args.imu_frames).resolve().parent / "imu_prior.csv")

    print(f"[IMU] building imu_prior: frames={args.imu_frames}, imu={args.imu_csv}, output={output}")
    build_imu_prior_csv(
        args.imu_frames,
        args.imu_csv,
        output,
        cam_sensor_yaml=args.cam_sensor_yaml,
        imu_sensor_yaml=args.imu_sensor_yaml,
        imu_to_cam_rotation=args.imu_to_cam_rotation,
        imu_to_cam_translation=args.imu_to_cam_translation,
        gyro_bias=args.gyro_bias,
        acc_bias=args.acc_bias,
    )

    return output


def image_stream(
    imagedir,
    calib,
    stride,
    max_frames=None,
    frame_records=None,
    euroc_original_protocol=False,
):
    """
    image generator

    이미지 폴더에서 프레임을 하나씩 읽어 DROID-SLAM에 넣을 형태로 변환하는 generator.

    입력:
        imagedir:
            이미지들이 저장된 폴더 경로

        calib:
            카메라 캘리브레이션 파일 경로
            보통 한 줄에 fx fy cx cy 또는 fx fy cx cy distortion... 형태

        stride:
            몇 프레임마다 하나씩 사용할지 결정
            예: stride=3이면 이미지 목록에서 3개마다 하나씩 사용

    출력 yield:
        t:
            현재 frame index

        image[None]:
            DROID 입력용 이미지 tensor
            shape: [1, 3, H, W]

        intrinsics:
            resize된 이미지 크기에 맞게 보정된 [fx, fy, cx, cy]
    """

    ############################################################
    # 1. calibration file 읽기
    ############################################################

    # calib.txt에서 카메라 파라미터 읽기
    # 예:
    #   fx fy cx cy
    # 또는
    #   fx fy cx cy k1 k2 p1 p2 ...
    calib = np.loadtxt(calib, delimiter=" ")

    # 앞 4개 값은 pinhole intrinsics
    fx, fy, cx, cy = calib[:4]

    ############################################################
    # 2. OpenCV용 camera matrix K 구성
    ############################################################

    # OpenCV undistort에 사용할 3x3 intrinsic matrix
    K = np.eye(3)

    # focal length x
    K[0, 0] = fx

    # principal point x
    K[0, 2] = cx

    # focal length y
    K[1, 1] = fy

    # principal point y
    K[1, 2] = cy

    # The original EuRoC benchmark evaluates rectified images at 320x512,
    # rather than the generic demo's adaptive-area resize.  Keep this as an
    # explicit opt-in profile because its camera coordinate frame also fixes
    # the IMU prior/calibration contract.
    strict_rectify_map = None

    ############################################################
    # 3. 이미지 목록 구성
    ############################################################

    # With IMU, frames.csv is the source of truth. It carries the timestamp
    # order used by preintegration and keeps a missing/corrupt image from
    # silently renumbering all later IMU rows. Legacy visual-only runs retain
    # the original sorted-directory behavior.
    manifest_mode = frame_records is not None
    if manifest_mode:
        selected_records = selected_frame_records(
            frame_records,
            stride,
            max_frames,
        )
        image_records = [
            (int(record["frame_index"]), str(record["filename"]))
            for record in selected_records
        ]
    else:
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        imagedir_path = Path(imagedir)
        skipped_missing = 0
        image_list = []
        for name in sorted(os.listdir(imagedir)):
            if ":" in name or Path(name).suffix.lower() not in valid_exts:
                continue

            image_path = imagedir_path / name
            if not image_path.is_file():
                skipped_missing += 1
                continue

            image_list.append(name)

        if skipped_missing > 0:
            print(
                f"[IMAGE_STREAM WARNING] skipped {skipped_missing} missing or broken image links in {imagedir}"
            )

        # Legacy mode has no timestamp manifest. Preserve the historical index
        # behavior only when the caller explicitly has no usable IMU contract.
        image_records = list(enumerate(image_list))[::max(1, int(stride))]
        if max_frames is not None and max_frames > 0:
            image_records = image_records[:int(max_frames)]

    if len(image_records) == 0:
        raise RuntimeError(
            f"No readable images found in {imagedir}. "
            "Check that the directory contains real image files or valid symlinks."
        )

    ############################################################
    # 4. 이미지 하나씩 읽어서 yield
    ############################################################

    for t, imfile in image_records:

        # 이미지 읽기
        # cv2.imread는 BGR 순서로 이미지를 읽는다.
        image_path = os.path.join(imagedir, imfile)
        image = cv2.imread(image_path)

        if image is None:
            if manifest_mode:
                raise RuntimeError(
                    "frames.csv selected an unreadable image; refusing to skip it "
                    f"and change the visual/IMU timeline: {image_path}"
                )
            print(f"[IMAGE_STREAM WARNING] failed to read image: {image_path}")
            continue

        ########################################################
        # 4-1. distortion 보정
        ########################################################

        # calib에 distortion parameter가 포함되어 있으면 undistort 수행
        #
        # calib[4:]:
        #   k1, k2, p1, p2 또는 SIMPLE_RADIAL의 k 값 등
        #
        # 주의:
        #   distortion 모델과 calib 형식이 OpenCV가 기대하는 형식과 맞아야 한다.
        if euroc_original_protocol:
            if strict_rectify_map is None:
                strict_rectify_map = strict_cam0_rectify_map(image.shape)
            image = strict_rectify_cam0(image, strict_rectify_map)
            fx, fy, cx, cy = strict_cam0_intrinsics((480, 752)).tolist()
        elif len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        ########################################################
        # 4-2. 원본 이미지 크기 확인
        ########################################################

        h0, w0, _ = image.shape

        ########################################################
        # 4-3. DROID 입력 해상도 계산
        ########################################################

        # DROID-SLAM은 너무 큰 이미지를 그대로 쓰지 않고,
        # 대략 384*512 픽셀 수에 맞도록 resize한다.
        #
        # 원본 비율은 유지하면서 전체 픽셀 수를 384*512 근처로 맞추는 방식.
        #
        # scale = sqrt((384*512) / (h0*w0))
        # h1 = h0 * scale
        # w1 = w0 * scale
        if euroc_original_protocol:
            h1, w1 = EUROC_DROID_IMAGE_SIZE
        else:
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        ########################################################
        # 4-4. 이미지 resize
        ########################################################

        image = cv2.resize(image, (w1, h1))

        ########################################################
        # 4-5. 이미지 크기를 8의 배수로 자르기
        ########################################################

        # DROID 네트워크는 feature map을 1/8 해상도에서 다룬다.
        # 따라서 H, W가 8로 나누어떨어지는 것이 안전하다.
        #
        # 예:
        #   h1 = 383이면 h1 - h1%8 = 376
        image = image[:h1 - h1 % 8, :w1 - w1 % 8]

        ########################################################
        # 4-6. torch tensor 변환
        ########################################################

        # cv2 image: [H, W, 3]
        # torch image: [3, H, W]
        image = torch.as_tensor(image).permute(2, 0, 1)

        ########################################################
        # 4-7. resize 비율에 맞춰 intrinsics 보정
        ########################################################

        # 원본 intrinsic
        intrinsics = torch.as_tensor([fx, fy, cx, cy])

        # x축 관련 값 fx, cx는 width 비율만큼 scale
        intrinsics[0::2] *= (w1 / w0)

        # y축 관련 값 fy, cy는 height 비율만큼 scale
        intrinsics[1::2] *= (h1 / h0)

        # generator output
        # image[None]을 하는 이유:
        #   batch/frame dimension을 추가해서 [1, 3, H, W] 형태로 만들기 위함
        yield t, image[None], intrinsics


def apply_euroc_original_protocol(args):
    """Apply the fixed visual settings from DROID's published EuRoC evaluator.

    This is a named reproducibility profile, not an optimizer or an ATE-tuning
    switch.  It deliberately overwrites the generic-demo defaults so a run
    cannot accidentally compare adaptive 384-area images or changed frontend
    thresholds against the paper's 320x512 visual baseline.
    """

    if not getattr(args, "euroc_original_protocol", False):
        return

    args.stride = 2
    args.trajectory_fill_stride = 1
    # Manifest-only mode bypasses the frontend/backend on purpose, so it also
    # bypasses the normal keyframe retirement that keeps an online DROID run
    # inside its 512-slot video buffer.  Its larger buffer is storage only:
    # MotionFilter's visual decision is unchanged.  Real baseline/learned
    # executions always receive the published 512-slot setting below.
    if not getattr(args, "motion_filter_manifest_only", False):
        args.buffer = 512
    args.image_size = list(EUROC_DROID_IMAGE_SIZE)
    args.filter_thresh = 2.4
    args.warmup = 15
    args.keyframe_thresh = 3.0
    args.frontend_thresh = 17.5
    args.frontend_window = 20
    args.frontend_radius = 2
    args.frontend_nms = 1
    args.backend_thresh = 24.0
    args.backend_radius = 2
    args.backend_nms = 2
    print(
        "[EuRoC ORIGINAL PROTOCOL] rectified_cam0=752x480->320x512, "
        f"track_stride=2, fill_stride=1, buffer={args.buffer}, filter=2.4, warmup=15, "
        "keyframe=3.0, frontend=(17.5,20,2,1), backend=(24.0,2,2)"
    )


def write_motion_filter_manifest(path, selected_rows):
    """Persist the input frames accepted by MotionFilter in one run.

    ``frame_index`` is the original 20 Hz EuRoC camera-row index, not the
    decimated-stream ordinal. The ordered training reader can consequently
    compose all raw IMU samples between adjacent selected rows.
    """

    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            "refusing to overwrite an existing MotionFilter manifest: "
            f"{output}"
        )
    if not selected_rows:
        raise ValueError("MotionFilter selected no frames; manifest was not written")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("frame_index", "timestamp_ns"),
        )
        writer.writeheader()
        writer.writerows(selected_rows)
    print(
        "[MOTION-FILTER MANIFEST] "
        f"selected_frames={len(selected_rows)}, output={output}"
    )


def _sha256_file(path):
    """Return a content digest for a small run-input file, if it exists."""

    if not path:
        return None

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return None

    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reconstruction_input_provenance(args):
    """Describe the exact image and calibration input used by one run.

    A reconstruction is otherwise only a tensor bundle.  Persisting this
    compact fingerprint makes it possible for validation scripts to reject a
    trajectory from another EuRoC sequence before it is accidentally reused.
    """

    if args is None or not getattr(args, "imagedir", None):
        return None

    image_dir = Path(args.imagedir).expanduser().resolve()
    calibration = Path(getattr(args, "calib", "")).expanduser().resolve()
    stride = max(1, int(getattr(args, "stride", 1) or 1))
    max_frames = max(0, int(getattr(args, "max_frames", 0) or 0))
    euroc_original_protocol = bool(
        getattr(args, "euroc_original_protocol", False)
    )
    local_tracking_only = bool(
        getattr(args, "local_tracking_only", False)
    )
    local_vio_only = bool(getattr(args, "local_vio_only", False))
    frame_records = getattr(args, "frame_records", None)
    frame_manifest_path = getattr(args, "imu_frame_manifest_path", None)

    if frame_records is not None:
        selected_records = selected_frame_records(frame_records, stride, max_frames)
        selected = [
            (int(record["frame_index"]), str(record["filename"]))
            for record in selected_records
        ]
        mapping_source = "frames_csv"
        mapping_digest = frame_manifest_digest(frame_records)
    else:
        valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        image_names = []
        if image_dir.is_dir():
            for name in sorted(os.listdir(image_dir)):
                candidate = image_dir / name
                if ":" in name or candidate.suffix.lower() not in valid_extensions:
                    continue
                if candidate.is_file():
                    image_names.append(name)

        selected = list(enumerate(image_names))[::stride]
        if max_frames > 0:
            selected = selected[:max_frames]
        mapping_source = "legacy_sorted_directory"
        mapping_digest = None

    manifest = hashlib.sha256()
    manifest.update(b"droid-reconstruction-input-v2\0")
    manifest.update(str(image_dir).encode("utf-8", "surrogateescape"))
    manifest.update(b"\0")
    manifest.update(str(stride).encode("ascii"))
    manifest.update(b"\0")
    manifest.update(str(max_frames).encode("ascii"))
    manifest.update(b"\0")
    manifest.update(str(int(euroc_original_protocol)).encode("ascii"))
    manifest.update(b"\0")
    for original_index, name in selected:
        manifest.update(f"{original_index}:{name}\n".encode("utf-8", "surrogateescape"))

    imu_prior = getattr(args, "imu_prior", None)
    imu_prior_path = (
        str(Path(imu_prior).expanduser().resolve()) if imu_prior else None
    )
    cam_sensor_yaml = getattr(args, "cam_sensor_yaml", None)
    imu_sensor_yaml = getattr(args, "imu_sensor_yaml", None)
    cam_sensor_path = (
        str(Path(cam_sensor_yaml).expanduser().resolve()) if cam_sensor_yaml else None
    )
    imu_sensor_path = (
        str(Path(imu_sensor_yaml).expanduser().resolve()) if imu_sensor_yaml else None
    )
    return {
        "version": 2,
        "image_directory": str(image_dir),
        "calibration_path": str(calibration),
        "calibration_sha256": _sha256_file(calibration),
        "stride": stride,
        "max_frames": max_frames,
        "euroc_original_protocol": euroc_original_protocol,
        # A local-only reconstruction deliberately stops before DROID's
        # termination-time backend and PoseTrajectoryFiller. Persist this so
        # it cannot later be mistaken for a paper-facing dense SLAM result.
        "local_tracking_only": local_tracking_only,
        # This mode produces a dense trajectory but intentionally skips every
        # termination-time global backend pass.
        "local_vio_only": local_vio_only,
        "selected_image_count": len(selected),
        "first_selected_image": (
            f"{selected[0][0]}:{selected[0][1]}" if selected else None
        ),
        "last_selected_image": (
            f"{selected[-1][0]}:{selected[-1][1]}" if selected else None
        ),
        "sequence_sha256": manifest.hexdigest(),
        "imu_prior_path": imu_prior_path,
        "imu_prior_sha256": _sha256_file(imu_prior_path),
        "frame_mapping_source": mapping_source,
        "frames_csv_path": frame_manifest_path,
        "frames_csv_sha256": _sha256_file(frame_manifest_path),
        "frame_manifest_sha256": mapping_digest,
        "imu_calibration_fingerprint": getattr(
            args,
            "imu_calibration_fingerprint",
            None,
        ),
        "cam_sensor_yaml_path": cam_sensor_path,
        "cam_sensor_yaml_sha256": _sha256_file(cam_sensor_path),
        "imu_sensor_yaml_path": imu_sensor_path,
        "imu_sensor_yaml_sha256": _sha256_file(imu_sensor_path),
    }


def save_reconstruction(
    droid,
    save_path,
    dense_tstamps=None,
    dense_poses=None,
    dense_protocol=None,
):
    """
    DROID-SLAM 결과를 .pth 파일로 저장하는 함수.

    저장 내용:
        tstamps:
            선택된 frame들의 timestamp/index

        images:
            선택된 frame 이미지

        disps:
            upsample된 disparity map

        poses:
            추정된 camera pose

        intrinsics:
            frame별 camera intrinsics

        dense_tstamps, dense_poses:
            ``Droid.terminate()`` 또는 ``Droid.terminate_local()``가
            PoseTrajectoryFiller로 완성한 trajectory.
            ``video``에는 MotionFilter가 남긴 keyframe만 있으므로, 전체
            camera 시간축 평가는 이 dense trajectory를 사용해야 한다.

        dense_protocol:
            Dense trajectory를 만들 때의 tracking/filling stride와 frame
            수를 기록한다. 이전 10 Hz reconstruction을 실수로 20 Hz
            paper-protocol 평가에 재사용하지 않기 위한 재현성 메타데이터다.

    입력:
        droid:
            Droid 또는 DroidAsync 객체

        save_path:
            저장할 .pth 경로
    """

    ############################################################
    # 1. 사용할 video buffer 선택
    ############################################################

    # DroidAsync는 frontend용 video1, backend용 video2를 가진다.
    # 최종 refinement 결과는 video2에 있으므로 video2를 저장한다.
    if hasattr(droid, "video2"):
        video = droid.video2

    # 동기식 Droid는 video 하나만 가진다.
    else:
        video = droid.video

    ############################################################
    # 2. 현재 저장된 frame 수 확인
    ############################################################

    t = video.counter.value

    if (dense_tstamps is None) != (dense_poses is None):
        raise ValueError(
            "dense_tstamps and dense_poses must either both be supplied or both be None"
        )
    if dense_tstamps is not None:
        dense_tstamps = np.asarray(dense_tstamps, dtype=np.int64).reshape(-1)
        dense_poses = np.asarray(dense_poses, dtype=np.float32)
        if dense_poses.ndim != 2 or dense_poses.shape[1] != 7:
            raise ValueError(
                "dense_poses must have shape [N, 7] as returned by Droid.terminate()"
            )
        if len(dense_tstamps) != len(dense_poses):
            raise ValueError(
                "dense trajectory timestamp/pose length mismatch: "
                f"{len(dense_tstamps)} != {len(dense_poses)}"
            )
        if not np.isfinite(dense_poses).all():
            raise ValueError("dense trajectory contains non-finite pose values")

    ############################################################
    # 3. 저장할 데이터 구성
    ############################################################

    alignment_metrics = dict(getattr(video, "imu_alignment_metrics", {}) or {})
    state_diagnostics = dict(
        getattr(video, "imu_state_diagnostics", {}) or {}
    )
    bias_refinement_history = list(
        getattr(video, "imu_bias_refinement_history", []) or []
    )
    alignment_ready = int(video.imu_alignment_ready.value)
    vi_initialization_requested = int(
        bool(getattr(getattr(droid, "args", None), "use_vi_initialization", False))
    )
    if not alignment_ready:
        alignment_source = (
            "not_aligned" if vi_initialization_requested else "visual_only"
        )
    elif float(alignment_metrics.get("backend_fallback", 0.0)) > 0.5:
        alignment_source = "backend_fallback"
    elif float(alignment_metrics.get("history_snapshot", 0.0)) > 0.5:
        alignment_source = "history"
    else:
        alignment_source = "live"

    bias_refinement_updates = sum(
        int(float(entry.get("updated", 0.0)) > 0.5)
        for entry in bias_refinement_history
        if isinstance(entry, dict)
    )
    input_provenance = _reconstruction_input_provenance(
        getattr(droid, "args", None)
    )

    save_data = {
        # timestamp/frame index
        "tstamps": video.tstamp[:t].cpu(),

        # 원본 이미지
        "images": video.images[:t].cpu(),

        # upsample된 disparity
        # args.reconstruction_path가 있으면 args.upsample=True로 설정되므로
        # disps_up이 생성될 수 있음
        "disps": video.disps_up[:t].cpu(),

        # pose
        "poses": video.poses[:t].cpu(),

        # IMU motion state and preintegrated measurements.
        # Full IMU BA updates velocities and local accel/gyro bias when enabled.
        "velocities": video.velocities[:t].cpu(),
        "bias_acc": video.bias_acc[:t].cpu(),
        "bias_gyro": video.bias_gyro[:t].cpu(),
        "imu_delta": video.imu_delta[:t].cpu(),
        "imu_valid": video.imu_valid[:t].cpu(),
        "imu_weight": video.imu_weight[:t].cpu(),
        "imu_used_steps": video.imu_used_steps[:t].cpu(),
        "imu_info": video.imu_info[:t].cpu(),
        # Auditable GRU-to-IMU runtime handoff. ``valid`` means the value came
        # from a direct visual temporal edge; invalid rows are neutral 1.0.
        "imu_learned_confidence": video.imu_learned_confidence[:t].cpu(),
        "imu_learned_confidence_valid": (
            video.imu_learned_confidence_valid[:t].cpu()
        ),
        "imu_bias_random_walk": video.imu_bias_random_walk[:t].cpu(),
        "imu_bias_jacobian": video.imu_bias_jacobian[:t].cpu(),
        # DROID can renormalize monocular pose/depth scale after the VI
        # initializer has set metric scale.  ``imu_unit_scale`` is the number
        # of current DROID length units per metre and lets exporters/evaluators
        # return poses and geometry to metres without guessing a Sim(3) scale.
        "imu_unit_scale": video.imu_unit_scale.cpu(),
        "pose_translation_unit_scale": video.imu_unit_scale.cpu(),
        "pose_translation_unit": (
            "droid_units_per_metre"
            if int(video.imu_alignment_ready.value)
            else "arbitrary_monocular_units"
        ),
        "imu_gravity": video.imu_gravity.cpu(),
        "imu_gravity_metric": (
            video.imu_gravity / video.imu_unit_scale[0].clamp_min(1e-12)
        ).cpu(),
        "imu_global_bias_acc": video.imu_global_bias_acc.cpu(),
        "imu_global_bias_gyro": video.imu_global_bias_gyro.cpu(),
        "imu_bias_refinement_history": bias_refinement_history,
        # Store execution provenance alongside the trajectory. A run with
        # low-rate bias enabled but zero accepted updates is a fixed-bias
        # result, not a bias-refinement ablation result.
        "imu_bias_refinement_attempts": len(bias_refinement_history),
        "imu_bias_refinement_updates": bias_refinement_updates,
        "imu_bias_relinearization_required": int(
            bool(getattr(video, "imu_bias_relinearization_required", False))
        ),
        "imu_camera_translation": video.imu_camera_translation.cpu(),
        "imu_camera_translation_source": getattr(
            video, "imu_camera_translation_source", "unknown"
        ),
        "imu_alignment_scale": video.imu_alignment_scale.cpu(),
        "imu_alignment_ready": alignment_ready,
        "imu_vi_initialization_requested": vi_initialization_requested,
        "imu_alignment_source": alignment_source,
        "imu_alignment_metrics": alignment_metrics,
        # Named snapshots isolate scale changes caused by VI initialization,
        # local tracking, and the two offline backend passes.
        "imu_state_diagnostics": state_diagnostics,
        # Input provenance lets a later held-out evaluation prove that every
        # reused reconstruction came from the requested sequence and sampling.
        "input_provenance": input_provenance,

        # intrinsics
        "intrinsics": video.intrinsics[:t].cpu(),
    }

    if dense_tstamps is not None:
        # ``terminate()`` returns camera-to-world poses with quaternion order
        # [qw, qx, qy, qz]. Keep that representation explicit rather than
        # pretending it has DepthVideo's world-to-camera qxyzw convention.
        save_data.update(
            {
                "dense_tstamps": torch.from_numpy(dense_tstamps.copy()),
                "dense_poses": torch.from_numpy(dense_poses.copy()),
                "dense_pose_format": "camera_c2w_wxyz",
            }
        )
        if dense_protocol is not None:
            if not isinstance(dense_protocol, dict):
                raise TypeError("dense_protocol must be a dictionary when supplied")
            save_data["dense_protocol"] = dict(dense_protocol)

    ############################################################
    # 4. torch save
    ############################################################

    torch.save(save_data, save_path)


if __name__ == '__main__':

    ############################################################
    # 1. command line argument 정의
    ############################################################

    parser = argparse.ArgumentParser()

    # 입력 이미지 폴더
    parser.add_argument("--imagedir", type=str, help="path to image directory")

    # 카메라 캘리브레이션 파일
    parser.add_argument("--calib", type=str, help="path to calibration file")

    # 시작 frame index
    parser.add_argument("--t0", default=0, type=int, help="starting frame")

    # frame stride
    # 예: 3이면 전체 이미지 중 3개마다 하나씩 사용
    parser.add_argument("--stride", default=3, type=int, help="tracking frame stride")
    parser.add_argument(
        "--trajectory_fill_stride",
        default=1,
        type=int,
        help=(
            "frame stride used by PoseTrajectoryFiller after local or global BA; "
            "use 1 to recover the full raw camera timeline while tracking with --stride 2"
        ),
    )
    parser.add_argument(
        "--local_tracking_only",
        action="store_true",
        help=(
            "diagnostic local-VO/VIO export: save only online "
            "MotionFilter-selected keyframe states; skip every termination-time "
            "global backend pass and PoseTrajectoryFiller"
        ),
    )
    parser.add_argument(
        "--local_vio_only",
        action="store_true",
        help=(
            "local VO/VIO evaluation export: skip termination-time global BA, "
            "but fill non-keyframe poses from the online local-BA state"
        ),
    )
    parser.add_argument("--max_frames", default=0, type=int, help="maximum number of selected frames to process")

    ############################################################
    # 2. DROID 기본 설정
    ############################################################

    # pretrained weight 파일
    parser.add_argument("--weights", default="droid.pth")

    # DepthVideo buffer 크기
    parser.add_argument("--buffer", type=int, default=512)

    # 초기 image size
    # 실제 실행 중 첫 이미지 크기로 다시 설정됨
    parser.add_argument("--image_size", default=[240, 320])

    parser.add_argument(
        "--euroc_original_protocol",
        action="store_true",
        help=(
            "use the original DROID EuRoC visual protocol exactly: fixed cam0 "
            "rectification, 320x512 input, track stride 2, dense fill stride 1, "
            "and the published frontend/backend settings"
        ),
    )

    # visualization 비활성화 옵션
    parser.add_argument("--disable_vis", action="store_true")

    ############################################################
    # 3. frontend / keyframe 관련 파라미터
    ############################################################

    # frame distance 계산에서 translation / rotation component에 대한 가중치
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="weight for translation / rotation components of flow"
    )

    # MotionFilter에서 새 keyframe으로 볼 motion threshold
    # 값이 작으면 더 많은 frame이 keyframe으로 들어가고,
    # 값이 크면 움직임이 큰 frame만 들어감
    parser.add_argument(
        "--filter_thresh",
        type=float,
        default=2.4,
        help="how much motion before considering new keyframe"
    )
    parser.add_argument(
        "--motion_filter_manifest_out",
        type=str,
        default=None,
        help=(
            "optional CSV destination for exact input frame IDs accepted by "
            "MotionFilter; intended for frozen ordered-training manifests"
        ),
    )
    parser.add_argument(
        "--motion_filter_manifest_only",
        action="store_true",
        help=(
            "export MotionFilter selections and exit before frontend/backend "
            "refinement; requires --motion_filter_manifest_out and --disable_vis"
        ),
    )

    # SLAM 초기화에 필요한 warmup frame 수
    # DepthVideo.counter.value가 이 값에 도달하면 frontend initialization 수행
    parser.add_argument(
        "--warmup",
        type=int,
        default=8,
        help="number of warmup frames"
    )

    # keyframe 제거 threshold
    # frame distance가 너무 작으면 중복 keyframe으로 보고 제거
    parser.add_argument(
        "--keyframe_thresh",
        type=float,
        default=4.0,
        help="threshold to create a new keyframe"
    )

    # frontend에서 proximity edge를 추가할 distance threshold
    parser.add_argument(
        "--frontend_thresh",
        type=float,
        default=16.0,
        help="add edges between frames whithin this distance"
    )

    # frontend local optimization window 크기
    parser.add_argument(
        "--frontend_window",
        type=int,
        default=25,
        help="frontend optimization window"
    )

    # frontend에서 시간적으로 가까운 frame을 강제 연결하는 radius
    parser.add_argument(
        "--frontend_radius",
        type=int,
        default=2,
        help="force edges between frames within radius"
    )

    # frontend edge 후보의 non-maximum suppression 값
    parser.add_argument(
        "--frontend_nms",
        type=int,
        default=1,
        help="non-maximal supression of edges"
    )

    ############################################################
    # 4. backend 관련 파라미터
    ############################################################

    # backend proximity edge threshold
    parser.add_argument("--backend_thresh", type=float, default=22.0)

    # backend에서 시간적으로 가까운 frame을 강제 연결하는 radius
    parser.add_argument("--backend_radius", type=int, default=2)

    # backend edge 후보 non-maximum suppression 값
    parser.add_argument("--backend_nms", type=int, default=3)

    # disparity upsampling 사용 여부
    # reconstruction 저장 시 고해상도 disparity가 필요하므로 자동으로 True가 될 수 있음
    parser.add_argument("--upsample", action="store_true")

    ############################################################
    # 5. async 실행 관련 파라미터
    ############################################################

    # 비동기 실행 여부
    # True이면 DroidAsync 사용
    # False이면 Droid 사용
    parser.add_argument("--asynchronous", action="store_true")

    # frontend가 사용할 device
    parser.add_argument("--frontend_device", type=str, default="cuda")

    # backend가 사용할 device
    parser.add_argument("--backend_device", type=str, default="cuda")
    
    ############################################################
    # 6. reconstruction 저장 경로
    ############################################################

    # 지정하면 save_reconstruction()으로 결과 저장
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")

    ############################################################
    # 7. IMU prior 관련 옵션
    ############################################################

    parser.add_argument("--imu_prior", type=str, default=None, help="path to imu_prior.csv")
    parser.add_argument(
        "--imu_frames",
        type=str,
        default=None,
        help="path to frames.csv for IMU preintegration and runtime frame mapping",
    )
    parser.add_argument(
        "--imu_frame_manifest",
        type=str,
        default=None,
        help="explicit frames.csv used to build --imu_prior; defaults to its sibling frames.csv",
    )
    parser.add_argument(
        "--allow_legacy_imu_frame_mapping",
        action="store_true",
        help="unsafe compatibility mode: use sorted image-directory indices when frames.csv is unavailable",
    )
    parser.add_argument(
        "--allow_imu_calibration_mismatch",
        action="store_true",
        help="unsafe compatibility mode: run even when runtime calibration differs from imu_prior.csv",
    )
    parser.add_argument("--imu_csv", type=str, default=None, help="path to raw imu.csv for building imu_prior.csv")
    parser.add_argument("--imu_prior_output", type=str, default=None, help="output path when building imu_prior.csv")
    parser.add_argument("--build_imu_prior", action="store_true", help="build imu_prior.csv before running demo")
    parser.add_argument("--cam_sensor_yaml", type=str, default=None, help="optional camera sensor.yaml for IMU-camera extrinsic")
    parser.add_argument("--imu_sensor_yaml", type=str, default=None, help="optional IMU sensor.yaml for IMU-camera extrinsic/noise")
    parser.add_argument(
        "--imu_to_cam_rotation",
        type=float,
        nargs=9,
        default=None,
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
        help="optional row-major 3x3 rotation from IMU frame to camera frame",
    )
    parser.add_argument(
        "--imu_to_cam_translation",
        type=float,
        nargs=3,
        default=None,
        metavar=("TX", "TY", "TZ"),
        help="IMU origin in camera coordinates t_CI [m], where p_C = R_CI p_I + t_CI",
    )
    parser.add_argument("--gyro_bias", type=float, nargs=3, default=None, metavar=("BX", "BY", "BZ"))
    parser.add_argument("--acc_bias", type=float, nargs=3, default=None, metavar=("BX", "BY", "BZ"))
    parser.add_argument("--use_imu_filter", action="store_true", help="use IMU rotation to keep rotating frames")
    parser.add_argument("--force_all_frames", action="store_true", help="append every frame without motion filtering")
    parser.add_argument("--imu_rot_thresh", type=float, default=0.035, help="IMU rotation threshold in radians")
    parser.add_argument("--imu_filter_gain", type=float, default=20.0, help="gain for lowering visual motion threshold")
    parser.add_argument("--imu_min_filter_thresh", type=float, default=0.5, help="minimum visual motion threshold")
    parser.add_argument("--use_imu_pose_prior", action="store_true", help="initialize pose rotation from IMU gyro preintegration")
    parser.add_argument("--imu_pose_rot_weight", type=float, default=1.0, help="rotation weight for IMU pose prior")
    parser.add_argument("--imu_pose_max_rot", type=float, default=0.35, help="max IMU rotation per frame in radians")
    parser.add_argument("--imu_axis_sign_x", type=float, default=1.0)
    parser.add_argument("--imu_axis_sign_y", type=float, default=1.0)
    parser.add_argument("--imu_axis_sign_z", type=float, default=1.0)
    parser.add_argument("--imu_pose_inverse", action="store_true", help="invert IMU delta rotation before applying")
    parser.add_argument(
        "--use_imu_translation_prior",
        action="store_true",
        help="deprecated unsafe direct dp pose prior; rejected in favor of the 9DoF position residual",
    )
    parser.add_argument("--imu_translation_weight", type=float, default=0.0, help="scale for experimental IMU translation prior")
    parser.add_argument("--imu_translation_max", type=float, default=0.05, help="max translation delta from IMU prior")
    parser.add_argument("--use_imu_ba_prior", action="store_true", help="add rotation-only IMU normal-equation prior inside CUDA DBA")
    parser.add_argument("--imu_ba_prior_weight", type=float, default=0.002, help="base Hessian weight for rotation-only IMU DBA prior")
    parser.add_argument(
        "--use_learned_imu_ba_weight",
        action="store_true",
        help="replace --imu_ba_prior_weight with the checkpoint-learned global IMU BA weight",
    )
    parser.add_argument(
        "--imu_confidence_only_weighting",
        action="store_true",
        help=(
            "fix the runtime global IMU factor weight to 1.0 and use learned "
            "temporal IMU confidence as the only adaptive attenuation"
        ),
    )
    parser.add_argument(
        "--learned_imu_ba_weight_scale",
        type=float,
        default=1.0,
        help="scale applied to the checkpoint-learned global IMU BA weight at runtime",
    )
    parser.add_argument(
        "--use_learned_imu_residual_weights",
        action="store_true",
        help="replace full IMU position/velocity weights with checkpoint-learned relative weights",
    )
    parser.add_argument("--imu_ba_prior_max_deg", type=float, default=10.0, help="legacy angular residual hard limit in degrees; zero disables it")
    parser.add_argument("--use_full_imu_ba", action="store_true", help="extend runtime CUDA DBA state to pose+velocity+accel-bias+gyro-bias")
    parser.add_argument(
        "--use_two_stage_imu_ba",
        action="store_true",
        help=(
            "local visual pose-only DBA followed by normal joint 15DoF IMU DBA; "
            "run one final global 15DoF backend refinement"
        ),
    )
    parser.add_argument(
        "--imu_two_stage_visual_steps",
        type=int,
        default=3,
        help=(
            "visual pose/depth-only local DBA updates before each two-stage "
            "15DoF update; default preserves the published experiment path"
        ),
    )
    parser.add_argument(
        "--imu_two_stage_full_steps",
        type=int,
        default=2,
        help=(
            "joint 15DoF local DBA updates after VI alignment in each "
            "two-stage frontend update; default preserves the experiment path"
        ),
    )
    parser.add_argument(
        "--imu_two_stage_global_steps",
        type=int,
        default=12,
        help="iterations in the primary final global 15DoF backend pass",
    )
    parser.add_argument(
        "--imu_two_stage_global_visual_pre_steps",
        type=int,
        default=0,
        help=(
            "DROID visual-only global-BA iterations before the final 15DoF "
            "backend pass; zero keeps the historical VI schedule"
        ),
    )
    parser.add_argument(
        "--imu_two_stage_matched_post_steps",
        type=int,
        default=0,
        help=(
            "extra final 15DoF backend iterations for a no-global-bias "
            "matched-iteration control"
        ),
    )
    parser.add_argument(
        "--imu_two_stage_skip_global_backend",
        action="store_true",
        help=(
            "diagnostic-only: stop after local two-stage DBA and skip the "
            "final global 15DoF backend refinement"
        ),
    )
    parser.add_argument(
        "--use_pose_velocity_imu_ba",
        action="store_true",
        help="use runtime IMU DBA with pose+velocity active state and fixed global gravity/bias",
    )
    parser.add_argument(
        "--imu_full_frontend",
        action="store_true",
        help=(
            "use the full preintegration factor in frontend updates; "
            "enabled automatically by --use_pose_velocity_imu_ba"
        ),
    )
    parser.add_argument("--imu_full_pos_weight", type=float, default=0.05, help="position residual weight inside full IMU DBA prior")
    parser.add_argument("--imu_full_vel_weight", type=float, default=0.05, help="velocity residual weight inside full IMU DBA prior")
    parser.add_argument("--imu_full_bias_weight", type=float, default=0.001, help="bias smoothness residual weight inside full IMU DBA prior")
    parser.add_argument(
        "--use_imu_bias_random_walk",
        action="store_true",
        help=(
            "replace the fixed 15DoF local-bias transition coefficient with "
            "per-edge sensor random-walk covariance, normalized to calibrated "
            "rotation information; requires --use_imu_noise_model"
        ),
    )
    parser.add_argument(
        "--use_imu_full_covariance_weighting",
        action="store_true",
        help=(
            "use calibrated diagonal preintegration information to preserve "
            "the p/v/R cross-block covariance ratio; replaces the historical "
            "0.05 position/velocity DBA weights and requires --use_imu_noise_model"
        ),
    )
    parser.add_argument("--imu_full_compose_order", type=str, default="dq_prev", choices=["prev_dq", "dq_prev"], help="quaternion compose order for full IMU BA")
    parser.add_argument("--imu_full_inverse", dest="imu_full_inverse", action="store_true", default=True, help="invert IMU delta rotation before full IMU BA")
    parser.add_argument("--no_imu_full_inverse", dest="imu_full_inverse", action="store_false", help="do not invert IMU delta rotation before full IMU BA")
    parser.add_argument("--use_imu_info_weighting", action="store_true", help="weight full IMU residuals by median-normalized preintegration information")
    parser.add_argument("--imu_info_weight_clip", type=float, default=4.0, help="clip for relative IMU information residual scales")
    parser.add_argument("--imu_info_weight_eps", type=float, default=1e-12, help="epsilon for IMU information weighting")
    parser.add_argument(
        "--use_imu_noise_model",
        action="store_true",
        help="use unclipped covariance-relative IMU information; pair with learned IMU confidence instead of the legacy threshold gate",
    )
    parser.add_argument("--use_adaptive_imu_gate", action="store_true", help="adapt IMU BA strength using visual quality, motion, and residual consistency")
    parser.add_argument("--imu_gate_min", type=float, default=0.2, help="minimum adaptive IMU gate multiplier")
    parser.add_argument("--imu_gate_max", type=float, default=1.25, help="maximum adaptive IMU gate multiplier")
    parser.add_argument("--imu_gate_visual_thresh", type=float, default=0.2, help="boost IMU below this mean visual confidence")
    parser.add_argument("--imu_gate_visual_boost", type=float, default=0.25, help="maximum visual-quality IMU boost above one")
    parser.add_argument("--imu_gate_motion_thresh_deg", type=float, default=5.0, help="rotation magnitude where the motion gate starts boosting IMU")
    parser.add_argument("--imu_gate_motion_boost", type=float, default=0.15, help="maximum motion-gate IMU boost above one")
    parser.add_argument("--imu_gate_residual_soft_deg", type=float, default=2.5, help="rotation residual where IMU consistency attenuation starts")
    parser.add_argument("--imu_gate_residual_hard_deg", type=float, default=8.0, help="rotation residual where IMU consistency reaches its minimum")
    parser.add_argument("--imu_gate_residual_pos_soft", type=float, default=0.0, help="optional position residual soft threshold; zero disables it")
    parser.add_argument("--imu_gate_residual_vel_soft", type=float, default=0.0, help="optional velocity residual soft threshold; zero disables it")
    parser.add_argument("--imu_motion_prior_weight", type=float, default=0.0, help="optional weak velocity prior weight for full runtime IMU BA")
    parser.add_argument("--imu_local_bias_prior_weight", type=float, default=0.0, help="optional weak local bias prior weight for full runtime IMU BA")
    parser.add_argument("--imu_gravity", type=float, nargs=3, default=None, metavar=("GX", "GY", "GZ"), help="gravity vector in world/DROID units for full IMU residuals")
    parser.add_argument("--estimate_imu_gravity", action="store_true", help="estimate initial gravity from imu_prior delta-velocity rows")
    parser.add_argument("--allow_zero_imu_gravity", action="store_true", help="debug only: allow full IMU BA to run with zero gravity")
    parser.add_argument("--imu_gravity_estimate_frames", type=int, default=80, help="max valid imu_prior rows used for gravity estimation")
    parser.add_argument("--imu_gravity_estimate_min_norm", type=float, default=6.0, help="minimum |dv/dt| used for gravity estimation")
    parser.add_argument("--imu_gravity_estimate_max_norm", type=float, default=12.5, help="maximum |dv/dt| used for gravity estimation")
    parser.add_argument("--imu_full_max_dt", type=float, default=0.5, help="legacy maximum preintegrated dt; zero disables the hard limit")
    parser.add_argument("--imu_full_max_dv", type=float, default=5.0, help="legacy maximum delta-velocity norm; zero disables the hard limit")
    parser.add_argument("--imu_full_max_dp", type=float, default=1.0, help="legacy maximum delta-position norm; zero disables the hard limit")
    parser.add_argument("--use_vi_initialization", action="store_true", help="estimate metric scale, gravity, global biases, and initial velocities after visual warm-up")
    parser.add_argument("--vi_init_min_frames", type=int, default=30, help="minimum keyframes before visual-inertial alignment")
    parser.add_argument("--vi_init_window", type=int, default=45, help="maximum recent keyframes used by visual-inertial alignment")
    parser.add_argument("--vi_init_retry_interval", type=int, default=5, help="frontend updates between failed alignment retries")
    parser.add_argument("--vi_init_max_dt", type=float, default=0.5, help="maximum IMU edge duration admitted by visual-inertial alignment")
    parser.add_argument("--vi_init_min_edges", type=int, default=20, help="minimum valid temporal IMU edges before alignment")
    parser.add_argument("--vi_init_min_duration", type=float, default=1.5, help="minimum valid IMU duration in seconds before alignment")
    parser.add_argument("--vi_init_min_rotation_deg", type=float, default=5.0, help="minimum accumulated IMU rotation excitation before alignment")
    parser.add_argument("--vi_init_min_visual_span_ratio", type=float, default=3.0, help="minimum visual span divided by median temporal displacement")
    parser.add_argument("--vi_init_max_rms_m", type=float, default=0.5, help="reject visual-inertial alignment above this metric residual RMS; zero disables")
    parser.add_argument("--vi_init_bootstrap", dest="vi_init_bootstrap", action="store_true", default=True, help="retain a bounded startup keyframe set for VI alignment")
    parser.add_argument("--no_vi_init_bootstrap", dest="vi_init_bootstrap", action="store_false", help="use normal MotionFilter during VI initialization")
    parser.add_argument("--vi_init_bootstrap_frames", type=int, default=60, help="maximum startup keyframes retained solely for VI alignment")
    parser.add_argument(
        "--visual_bootstrap_control",
        action="store_true",
        help=(
            "visual-only ablation: retain the same startup frontend keyframes "
            "as VI initialization, but do not initialize or optimize IMU state"
        ),
    )
    parser.add_argument(
        "--vi_init_snapshot_history_size",
        type=int,
        default=24,
        help="number of compact pre-removal VI initialization windows retained for fallback alignment; zero disables history",
    )
    parser.add_argument(
        "--vi_init_snapshot_stride",
        type=int,
        default=5,
        help="frontend-update interval used to capture one VI initialization snapshot",
    )
    parser.add_argument(
        "--vi_init_snapshot_max_candidates",
        type=int,
        default=6,
        help="maximum observable retained windows solved per failed live VI initialization attempt",
    )
    parser.add_argument(
        "--vi_init_backend_fallback",
        action="store_true",
        help="offline only: retry VI alignment after the first visual backend pass when online initialization did not align",
    )
    parser.add_argument(
        "--vi_init_backend_window",
        type=int,
        default=120,
        help="maximum active keyframes used by the optional post-backend VI alignment fallback",
    )
    parser.add_argument("--vi_init_min_scale", type=float, default=0.05, help="minimum valid visual-to-metric alignment scale")
    parser.add_argument("--vi_init_max_scale", type=float, default=20.0, help="maximum valid visual-to-metric alignment scale")
    parser.add_argument("--vi_init_max_condition", type=float, default=1e9, help="maximum linear-system condition number for visual-inertial alignment")
    parser.add_argument("--vi_init_max_accel_bias_norm", type=float, default=1.0, help="reject initial alignment above this accelerometer-bias norm in m/s^2")
    parser.add_argument("--vi_init_max_gyro_bias_norm", type=float, default=0.2, help="reject initial alignment above this gyroscope-bias norm in rad/s")
    parser.add_argument(
        "--use_low_rate_bias_refinement",
        action="store_true",
        help=(
            "run frontend-only low-rate shared 6D accel/gyro bias refinement "
            "between ordinary 9DoF DBA updates"
        ),
    )
    parser.add_argument(
        "--use_shared_bias_local_ba",
        action="store_true",
        help="legacy alias for --use_low_rate_bias_refinement",
    )
    parser.add_argument(
        "--use_window_bias_refinement",
        action="store_true",
        help="legacy alias for --use_shared_bias_local_ba",
    )
    parser.add_argument("--imu_bias_refine_interval", type=int, default=20, help="frontend updates between shared-bias refinements")
    parser.add_argument("--imu_bias_refine_window", type=int, default=30, help="keyframes used by one shared-bias refinement")
    parser.add_argument("--imu_bias_refine_damping", type=float, default=1e-4, help="damping for the six-variable shared-bias solve")
    parser.add_argument("--imu_bias_refine_step_scale", type=float, default=0.5, help="damped update multiplier for shared-bias refinement")
    parser.add_argument("--imu_bias_refine_max_accel_step", type=float, default=0.15, help="maximum accelerometer-bias update per refinement in m/s^2")
    parser.add_argument("--imu_bias_refine_max_gyro_step", type=float, default=0.015, help="maximum gyroscope-bias update per refinement in rad/s")
    parser.add_argument("--imu_bias_refine_max_accel_norm", type=float, default=0.5, help="maximum total shared accelerometer-bias norm in m/s^2")
    parser.add_argument("--imu_bias_refine_max_gyro_norm", type=float, default=0.1, help="maximum total shared gyroscope-bias norm in rad/s")
    parser.add_argument("--imu_bias_refine_robust_delta", type=float, default=3.0, help="Huber threshold after IMU residual covariance normalization; zero disables robustification")
    parser.add_argument("--imu_bias_refine_min_edges", type=int, default=8, help="minimum valid temporal IMU factors for a shared-bias update")
    parser.add_argument("--imu_bias_refine_min_duration", type=float, default=0.5, help="minimum IMU duration in seconds for a shared-bias update")
    parser.add_argument("--imu_bias_refine_min_rotation_deg", type=float, default=2.0, help="minimum accumulated IMU rotation excitation for a shared-bias update")
    parser.add_argument("--imu_bias_refine_max_condition", type=float, default=1e8, help="maximum shared-bias data-Hessian condition number")
    parser.add_argument(
        "--imu_bias_refine_min_relative_improvement",
        type=float,
        default=1e-7,
        help=(
            "minimum relative local IMU-objective reduction needed to commit "
            "a low-rate bias update; 0 accepts every strict decrease"
        ),
    )
    parser.add_argument(
        "--imu_bias_refine_backend",
        action="store_true",
        help=(
            "also run one low-rate bias refinement in each termination-time "
            "backend pass; disabled by default to preserve online local DBA behavior"
        ),
    )
    parser.add_argument("--imu_bias_refine_accel_random_walk", type=float, default=-1.0, help="override accelerometer bias random walk; negative reads imu_prior.csv")
    parser.add_argument("--imu_bias_refine_gyro_random_walk", type=float, default=-1.0, help="override gyroscope bias random walk; negative reads imu_prior.csv")
    parser.add_argument("--imu_bias_refine_accel_prior_std", type=float, default=0.05, help="minimum accelerometer-bias trust-region std per shared update in m/s^2")
    parser.add_argument("--imu_bias_refine_gyro_prior_std", type=float, default=0.005, help="minimum gyroscope-bias trust-region std per shared update in rad/s")
    parser.add_argument(
        "--use_joint_shared_bias_ba",
        action="store_true",
        help=(
            "jointly couple one shared accel/gyro bias block to each local "
            "9DoF DBA window through Schur elimination"
        ),
    )
    parser.add_argument(
        "--use_joint_shared_bias_15dof",
        action="store_true",
        help=(
            "two-stage 15DoF only: jointly eliminate/recover one shared "
            "6D accel/gyro bias block inside the final full-sequence global "
            "DBA, coupled to pose, velocity, and local bias states"
        ),
    )
    parser.add_argument("--imu_joint_bias_interval", type=int, default=10, help="keyframe interval between coupled shared-bias DBA updates")
    parser.add_argument("--imu_joint_bias_window", type=int, default=30, help="recent keyframes coupled to one shared 6D bias block")
    parser.add_argument("--imu_joint_bias_damping", type=float, default=1e-4, help="damping added to the shared-bias normal block")
    parser.add_argument("--imu_joint_bias_step_scale", type=float, default=1.0, help="shared-bias back-substitution step multiplier")
    parser.add_argument("--imu_joint_bias_max_accel_step", type=float, default=0.15, help="maximum coupled accelerometer-bias update per DBA step in m/s^2")
    parser.add_argument("--imu_joint_bias_max_gyro_step", type=float, default=0.015, help="maximum coupled gyroscope-bias update per DBA step in rad/s")
    parser.add_argument("--imu_joint_bias_max_accel_norm", type=float, default=0.5, help="maximum total shared accelerometer-bias norm in m/s^2")
    parser.add_argument("--imu_joint_bias_max_gyro_norm", type=float, default=0.1, help="maximum total shared gyroscope-bias norm in rad/s")
    parser.add_argument("--imu_joint_bias_min_edges", type=int, default=8, help="minimum temporal IMU factors required for the coupled bias block")
    parser.add_argument("--imu_joint_bias_min_duration", type=float, default=0.5, help="minimum IMU duration in seconds required for the coupled bias block")
    parser.add_argument("--imu_joint_bias_min_rotation_deg", type=float, default=2.0, help="minimum accumulated IMU rotation required for the coupled bias block")
    parser.add_argument("--imu_joint_bias_max_condition", type=float, default=1e8, help="maximum data-Hessian condition number for the coupled bias block")
    parser.add_argument("--imu_joint_bias_accel_random_walk", type=float, default=-1.0, help="override accelerometer bias random walk; negative reads imu_prior.csv")
    parser.add_argument("--imu_joint_bias_gyro_random_walk", type=float, default=-1.0, help="override gyroscope bias random walk; negative reads imu_prior.csv")
    parser.add_argument("--imu_joint_bias_accel_prior_std", type=float, default=0.05, help="minimum coupled accelerometer-bias random-walk prior std in m/s^2")
    parser.add_argument("--imu_joint_bias_gyro_prior_std", type=float, default=0.005, help="minimum coupled gyroscope-bias random-walk prior std in rad/s")
    parser.add_argument("--imu_joint_bias_min_improvement", type=float, default=1e-10, help="minimum local quadratic improvement before a coupled bias update is accepted")
    parser.add_argument("--imu_joint_bias_relinearize_accel", type=float, default=0.05, help="warn when a coupled accel-bias step exceeds this first-order relinearization threshold")
    parser.add_argument("--imu_joint_bias_relinearize_gyro", type=float, default=0.005, help="warn when a coupled gyro-bias step exceeds this first-order relinearization threshold")
    parser.add_argument(
        "--use_global_bias_15dof_refinement",
        action="store_true",
        help=(
            "offline two-stage mode: after the global 15DoF DBA, refine one "
            "shared accel/gyro bias block across the full sequence and rerun "
            "a short 15DoF backend pass"
        ),
    )
    parser.add_argument(
        "--use_online_global_bias_15dof_refinement",
        action="store_true",
        help=(
            "during local two-stage 15DoF tracking, periodically solve one "
            "observability-checked shared accel/gyro bias block over the recent "
            "active keyframes, then rerun local 15DoF DBA"
        ),
    )
    parser.add_argument(
        "--imu_online_global_bias_refine_interval",
        type=int,
        default=20,
        help="selected-keyframe interval between online shared 15DoF bias attempts",
    )
    parser.add_argument(
        "--imu_online_global_bias_refine_window",
        type=int,
        default=30,
        help="recent selected-keyframe window used by each online shared 15DoF bias solve",
    )
    parser.add_argument(
        "--imu_online_global_bias_refine_post_iterations",
        type=int,
        default=1,
        help="local full-15DoF DBA iterations after an accepted online shared-bias update",
    )
    parser.add_argument(
        "--imu_global_bias_refine_post_steps",
        type=int,
        default=2,
        help="global 15DoF DBA iterations after an accepted shared global-bias update",
    )
    parser.add_argument(
        "--imu_global_bias_local_prior_weight",
        type=float,
        default=0.0,
        help=(
            "deprecated compatibility option; set --imu_local_bias_prior_weight "
            "explicitly when a local-bias regularizer is desired"
        ),
    )
    parser.add_argument(
        "--imu_global_bias_refine_weight",
        type=float,
        default=1.0,
        help=(
            "IMU likelihood scale for the post-global 6D bias solve; this is "
            "independent of visual-versus-IMU DBA balance because T/v are fixed"
        ),
    )
    parser.add_argument("--imu_global_bias_refine_damping", type=float, default=1e-4, help="damping for the post-global shared 6D bias block")
    parser.add_argument("--imu_global_bias_refine_step_scale", type=float, default=0.5, help="damped update multiplier for the post-global shared 6D bias block")
    parser.add_argument("--imu_global_bias_refine_max_accel_step", type=float, default=0.15, help="maximum post-global accelerometer-bias update in m/s^2")
    parser.add_argument("--imu_global_bias_refine_max_gyro_step", type=float, default=0.015, help="maximum post-global gyroscope-bias update in rad/s")
    parser.add_argument("--imu_global_bias_refine_max_accel_norm", type=float, default=2.0, help="maximum global accelerometer-bias norm in m/s^2")
    parser.add_argument("--imu_global_bias_refine_max_gyro_norm", type=float, default=0.2, help="maximum global gyroscope-bias norm in rad/s")
    parser.add_argument("--imu_global_bias_refine_robust_delta", type=float, default=3.0, help="Huber threshold for the post-global shared-bias objective")
    parser.add_argument("--imu_global_bias_refine_min_edges", type=int, default=20, help="minimum full-sequence IMU factors for a post-global shared-bias update")
    parser.add_argument("--imu_global_bias_refine_min_duration", type=float, default=2.0, help="minimum IMU duration in seconds for a post-global shared-bias update")
    parser.add_argument("--imu_global_bias_refine_min_rotation_deg", type=float, default=5.0, help="minimum accumulated IMU rotation for a post-global shared-bias update")
    parser.add_argument("--imu_global_bias_refine_max_condition", type=float, default=1e8, help="maximum data-Hessian condition number for the post-global shared-bias block")
    parser.add_argument("--imu_global_bias_refine_min_relative_improvement", type=float, default=1e-4, help="minimum relative IMU-objective reduction required for a post-global shared-bias update")
    parser.add_argument("--imu_global_bias_refine_accel_random_walk", type=float, default=-1.0, help="override accelerometer bias random walk for the post-global shared-bias trust region")
    parser.add_argument("--imu_global_bias_refine_gyro_random_walk", type=float, default=-1.0, help="override gyroscope bias random walk for the post-global shared-bias trust region")
    parser.add_argument("--imu_global_bias_refine_accel_prior_std", type=float, default=0.5, help="accelerometer-bias trust-region std for the post-global shared block in m/s^2")
    parser.add_argument("--imu_global_bias_refine_gyro_prior_std", type=float, default=0.05, help="gyroscope-bias trust-region std for the post-global shared block in rad/s")
    parser.add_argument("--imu_global_bias_relinearize_accel", type=float, default=0.05, help="mark re-preintegration as recommended above this post-global accel-bias step")
    parser.add_argument("--imu_global_bias_relinearize_gyro", type=float, default=0.005, help="mark re-preintegration as recommended above this post-global gyro-bias step")
    parser.add_argument(
        "--imu_ba_outer_relinearize",
        action="store_true",
        help=(
            "run selected fixed-bias 9DoF DBA updates as one-step outer "
            "relinearizations; use as the matched control for shared-bias BA"
        ),
    )
    parser.add_argument(
        "--imu_ba_outer_relinearize_interval",
        type=int,
        default=10,
        help="keyframe interval between fixed-bias outer-relinearization control updates",
    )
    parser.add_argument(
        "--imu_ba_outer_window",
        type=int,
        default=0,
        help="expanded local DBA window for fixed-bias outer-relinearization control; zero keeps the normal window",
    )
    parser.add_argument("--imu_ba_debug", action="store_true", help="write full IMU BA prior debug csv")
    parser.add_argument("--imu_ba_debug_path", type=str, default=None, help="optional path for full IMU BA prior debug csv")
    parser.add_argument("--imu_ba_debug_max_rows", type=int, default=20000, help="max full IMU BA debug rows to keep in memory")
    parser.add_argument("--use_imu_residual", action="store_true", help="apply rotation-only inertial residual after visual DBA")
    parser.add_argument("--imu_residual_weight", type=float, default=0.02, help="base alpha for rotation-only inertial residual")
    parser.add_argument("--imu_residual_window", type=int, default=12, help="recent keyframe window for inertial residual")
    parser.add_argument("--imu_residual_max_deg", type=float, default=45.0, help="skip inertial residuals above this angular error")
    parser.add_argument("--imu_residual_max_alpha", type=float, default=0.05, help="maximum single-step slerp alpha for inertial residual")
    parser.add_argument("--imu_residual_max_frame_gap", type=int, default=30, help="max original-frame gap to compose IMU deltas")
    parser.add_argument("--imu_residual_compose_order", type=str, default="prev_dq", choices=["prev_dq", "dq_prev"], help="pose/IMU quaternion compose order")
    parser.add_argument("--imu_residual_inverse", action="store_true", help="invert composed IMU delta before residual")
    parser.add_argument(
        "--use_learned_imu_confidence",
        action="store_true",
        help="scale IMU residual/prior strength with the GRU IMU confidence head",
    )
    parser.add_argument(
        "--use_learned_imu_precision",
        action="store_true",
        help=(
            "learn covariance-relative GRU+IMU-quality R/v/p IMU attenuation; "
            "uses no global IMU lambda or fixed R/v/p compatibility weights"
        ),
    )
    parser.add_argument(
        "--use_imu_precision_discrepancy",
        action="store_true",
        help=(
            "append causal covariance-normalized visual--IMU R/v/p discrepancy "
            "to the original direct precision quality input; requires "
            "--use_learned_imu_precision and adds no output-logit adapter"
        ),
    )
    parser.add_argument(
        "--use_imu_precision_innovation",
        action="store_true",
        help=(
            "add causal visual--IMU R/v/p innovation features to the direct "
            "precision head; requires --use_learned_imu_precision"
        ),
    )
    parser.add_argument(
        "--use_imu_precision_visual_innovation",
        action="store_true",
        help=(
            "conditionally fuse each causal R/v/p innovation with DROID's "
            "current visual BA-weight mean/spread; requires "
            "--use_imu_precision_innovation"
        ),
    )
    parser.add_argument(
        "--use_imu_precision_edge_fusion",
        action="store_true",
        help=(
            "fuse each edge's causal IMU innovation, raw IMU quality, frozen "
            "ConvGRU visual evidence, and DROID correspondence-correction "
            "statistics; requires --use_imu_precision_innovation"
        ),
    )
    parser.add_argument(
        "--use_component_imu_confidence",
        action="store_true",
        help=(
            "use separate GRU confidence for IMU rotation, velocity, and position "
            "residuals with preintegration-quality fusion"
        ),
    )
    parser.add_argument(
        "--use_dvi_imu_confidence",
        action="store_true",
        help=(
            "use the DVI-style scalar IMU confidence head: 1x1 GRU projection "
            "followed by spatial average pooling"
        ),
    )
    parser.add_argument(
        "--use_dvi_imu_quality_fusion",
        action="store_true",
        help=(
            "add direct preintegrated-IMU quality features to the DVI scalar "
            "confidence logit"
        ),
    )
    parser.add_argument(
        "--imu_tuned_confidence_schedule",
        type=float,
        nargs="+",
        default=None,
        metavar="W",
        help=(
            "manual scalar IMU confidence per outer 15DoF DBA update; "
            "used only for the fixed/tuned DVI ablation control"
        ),
    )
    parser.add_argument(
        "--imu_confidence_floor",
        "--imu_conf_floor",
        dest="imu_confidence_floor",
        type=float,
        default=0.0,
        help="minimum confidence mixed into learned IMU residual confidence",
    )
    parser.add_argument("--imu_residual_debug", action="store_true", help="write imu residual debug csv")
    parser.add_argument("--imu_residual_debug_path", type=str, default=None, help="optional path for imu residual debug csv")

    # argument parsing
    args = parser.parse_args()

    ############################################################
    # 7. 추가 설정
    ############################################################

    apply_euroc_original_protocol(args)

    # demo.py는 monocular 입력을 기준으로 사용
    args.stereo = False

    if args.use_imu_translation_prior:
        raise ValueError(
            "--use_imu_translation_prior is disabled for this visual-inertial "
            "pipeline. A raw dp cannot be added directly to DROID's world-to-camera "
            "translation; use the 9DoF position residual instead."
        )

    if args.use_imu_pose_prior and (
        any(
            abs(float(value) - 1.0) > 1e-12
            for value in (
                args.imu_axis_sign_x,
                args.imu_axis_sign_y,
                args.imu_axis_sign_z,
            )
        )
        or args.imu_pose_inverse
    ):
        raise ValueError(
            "legacy --imu_axis_sign_* and --imu_pose_inverse are disabled for "
            "--use_imu_pose_prior. Apply R_CI during preintegration; pose prior "
            "now follows --imu_full_compose_order/--imu_full_inverse exactly."
        )

    if args.use_two_stage_imu_ba:
        incompatible = []
        for option, active in (
            ("--use_pose_velocity_imu_ba", args.use_pose_velocity_imu_ba),
            ("--use_low_rate_bias_refinement", args.use_low_rate_bias_refinement),
            ("--use_shared_bias_local_ba", args.use_shared_bias_local_ba),
            ("--use_window_bias_refinement", args.use_window_bias_refinement),
            ("--use_joint_shared_bias_ba", args.use_joint_shared_bias_ba),
            ("--imu_ba_outer_relinearize", args.imu_ba_outer_relinearize),
            ("--imu_bias_refine_backend", args.imu_bias_refine_backend),
            ("--vi_init_backend_fallback", args.vi_init_backend_fallback),
        ):
            if active:
                incompatible.append(option)
        if incompatible:
            raise ValueError(
                "--use_two_stage_imu_ba cannot be combined with "
                + ", ".join(incompatible)
            )

        args.use_full_imu_ba = True
        args.imu_full_frontend = True
        args.use_imu_ba_prior = True
        args.use_vi_initialization = True
        if args.imu_two_stage_visual_steps < 0:
            raise ValueError("--imu_two_stage_visual_steps must be non-negative")
        if args.imu_two_stage_full_steps < 0:
            raise ValueError("--imu_two_stage_full_steps must be non-negative")
        if args.imu_two_stage_global_visual_pre_steps < 0:
            raise ValueError(
                "--imu_two_stage_global_visual_pre_steps must be non-negative"
            )
        print(
            "[IMU] two-stage mode: local pose-only -> full 15DoF; "
            "one global full-state backend pass; "
            f"local_steps=visual:{args.imu_two_stage_visual_steps},"
            f"full:{args.imu_two_stage_full_steps}."
        )
        if args.imu_two_stage_full_steps == 0:
            print(
                "[IMU] local full 15DoF updates are disabled: keep online "
                "tracking visual-only and estimate velocity/local bias only "
                "in the final global 15DoF backend pass."
            )

    if args.use_joint_shared_bias_15dof:
        if not args.use_two_stage_imu_ba:
            raise ValueError(
                "--use_joint_shared_bias_15dof requires --use_two_stage_imu_ba "
                "so the final backend owns 15DoF local bias states."
            )
        if args.use_joint_shared_bias_ba:
            raise ValueError(
                "Choose --use_joint_shared_bias_15dof or the legacy 9DoF "
                "--use_joint_shared_bias_ba, not both."
            )
        if args.use_global_bias_15dof_refinement:
            raise ValueError(
                "--use_joint_shared_bias_15dof replaces the post-hoc "
                "--use_global_bias_15dof_refinement path; do not enable both."
            )
        print(
            "[IMU] final global joint shared-bias 15DoF DBA enabled: "
            "shared 6D bias is Schur-coupled to the full pose/velocity/local-bias state."
        )

    if args.use_global_bias_15dof_refinement:
        if not args.use_two_stage_imu_ba:
            raise ValueError(
                "--use_global_bias_15dof_refinement requires "
                "--use_two_stage_imu_ba so the completed trajectory has "
                "per-keyframe 15DoF bias state."
            )
        if args.imu_global_bias_refine_post_steps < 1:
            raise ValueError(
                "--imu_global_bias_refine_post_steps must be at least one; "
                "an accepted global-bias step needs a following 15DoF DBA pass."
            )
        if args.imu_global_bias_local_prior_weight < 0.0:
            raise ValueError(
                "--imu_global_bias_local_prior_weight must be non-negative."
            )
        if args.imu_two_stage_global_steps < 1:
            raise ValueError("--imu_two_stage_global_steps must be at least one.")
        if args.imu_two_stage_matched_post_steps < 0:
            raise ValueError("--imu_two_stage_matched_post_steps must be non-negative.")
        if args.imu_two_stage_skip_global_backend:
            raise ValueError(
                "--use_global_bias_15dof_refinement requires the final global "
                "15DoF backend pass; remove --imu_two_stage_skip_global_backend"
            )
        if args.imu_global_bias_refine_weight <= 0.0:
            raise ValueError(
                "--imu_global_bias_refine_weight must be positive."
            )
        print(
            "[IMU] final global shared-bias refinement enabled: "
            f"post_steps={args.imu_global_bias_refine_post_steps}, "
            "precenter=total_bias_preserving, "
            f"bias_likelihood_weight={args.imu_global_bias_refine_weight}"
        )
        if args.imu_global_bias_local_prior_weight > 0.0:
            print(
                "[IMU WARNING] --imu_global_bias_local_prior_weight is no longer "
                "injected only at termination; use --imu_local_bias_prior_weight "
                "for an explicit, consistently applied regularizer."
            )

    if args.use_online_global_bias_15dof_refinement:
        if not args.use_two_stage_imu_ba:
            raise ValueError(
                "--use_online_global_bias_15dof_refinement requires "
                "--use_two_stage_imu_ba so the local state contains per-keyframe "
                "velocity and bias corrections."
            )
        if args.imu_online_global_bias_refine_interval < 1:
            raise ValueError(
                "--imu_online_global_bias_refine_interval must be at least one."
            )
        if args.imu_online_global_bias_refine_window < 2:
            raise ValueError(
                "--imu_online_global_bias_refine_window must be at least two."
            )
        if args.imu_online_global_bias_refine_post_iterations < 1:
            raise ValueError(
                "--imu_online_global_bias_refine_post_iterations must be at least one."
            )
        print(
            "[IMU] online shared global-bias 15DoF block enabled: "
            f"interval={args.imu_online_global_bias_refine_interval}, "
            f"window={args.imu_online_global_bias_refine_window}, "
            f"post_iterations={args.imu_online_global_bias_refine_post_iterations}"
        )

    if args.use_pose_velocity_imu_ba:
        args.use_full_imu_ba = True
        # A pose-velocity mode is intended to run the 9DoF factor inside the
        # local frontend DBA after VI alignment. Previously the caller also
        # had to remember the legacy --imu_full_frontend switch; omitting it
        # silently delayed all full IMU optimization until termination-time
        # backend passes. Keep the explicit flag for compatibility, but make
        # the coherent 9DoF path the default requested by this mode.
        if not args.imu_full_frontend:
            args.imu_full_frontend = True
            print(
                "[IMU] --use_pose_velocity_imu_ba enables "
                "frontend local pose-velocity DBA."
            )

    if args.use_low_rate_bias_refinement or args.use_shared_bias_local_ba:
        args.use_window_bias_refinement = True
    if args.use_joint_shared_bias_ba:
        if not args.use_pose_velocity_imu_ba:
            raise ValueError(
                "--use_joint_shared_bias_ba requires --use_pose_velocity_imu_ba "
                "so the per-keyframe CUDA state remains pose+velocity (9D)."
            )
        if not args.use_vi_initialization:
            raise ValueError(
                "--use_joint_shared_bias_ba requires --use_vi_initialization "
                "to seed metric scale, gravity, and the carried bias estimate."
            )
        if not args.imu_ba_outer_relinearize:
            raise ValueError(
                "--use_joint_shared_bias_ba is an experimental outer-"
                "relinearization path and requires the explicit "
                "--imu_ba_outer_relinearize flag. Use "
                "--use_low_rate_bias_refinement for the stable online path."
            )
        if args.use_window_bias_refinement:
            raise ValueError(
                "Choose either --use_joint_shared_bias_ba or the legacy "
                "--use_shared_bias_local_ba, not both."
            )
    if args.use_window_bias_refinement and not args.use_pose_velocity_imu_ba:
        raise ValueError(
            "--use_shared_bias_local_ba requires --use_pose_velocity_imu_ba "
            "so bias is coupled to the fixed-bias 9DoF local DBA."
        )
    if args.use_window_bias_refinement and args.imu_ba_outer_relinearize:
        raise ValueError(
            "--use_low_rate_bias_refinement cannot be combined with "
            "--imu_ba_outer_relinearize. The latter changes the normal local "
            "DBA schedule and is reserved for the joint-bias diagnostic."
        )
    if args.imu_bias_refine_backend and not args.use_window_bias_refinement:
        raise ValueError(
            "--imu_bias_refine_backend requires --use_low_rate_bias_refinement."
        )
    if args.use_window_bias_refinement:
        backend_stage = "enabled" if args.imu_bias_refine_backend else "disabled"
        print(
            "[IMU] low-rate shared-bias refinement enabled: "
            f"frontend interval={args.imu_bias_refine_interval}, "
            f"window={args.imu_bias_refine_window}, backend={backend_stage}"
        )

    if args.use_vi_initialization and not (
        args.use_pose_velocity_imu_ba or args.use_two_stage_imu_ba
    ):
        raise ValueError(
            "--use_vi_initialization requires --use_pose_velocity_imu_ba "
            "or --use_two_stage_imu_ba"
        )
    if args.visual_bootstrap_control:
        incompatible = []
        for option, active in (
            ("--use_vi_initialization", args.use_vi_initialization),
            ("--use_full_imu_ba", args.use_full_imu_ba),
            ("--use_imu_ba_prior", args.use_imu_ba_prior),
            ("--use_pose_velocity_imu_ba", args.use_pose_velocity_imu_ba),
            ("--use_two_stage_imu_ba", args.use_two_stage_imu_ba),
        ):
            if active:
                incompatible.append(option)
        if incompatible:
            raise ValueError(
                "--visual_bootstrap_control is visual-only and cannot be combined with "
                + ", ".join(incompatible)
            )
        print(
            "[VISUAL-CONTROL] retaining startup keyframes without VI "
            "initialization or IMU factors"
        )
    if args.use_vi_initialization and args.asynchronous:
        raise ValueError(
            "--use_vi_initialization currently requires synchronous Droid; "
            "the async frontend/backend coordinate handoff is not yet VI-aligned."
        )
    if args.motion_filter_manifest_out is not None and args.asynchronous:
        raise ValueError(
            "--motion_filter_manifest_out requires synchronous Droid because "
            "the asynchronous tracker does not expose a synchronous selection result"
        )
    if args.motion_filter_manifest_only and args.motion_filter_manifest_out is None:
        raise ValueError(
            "--motion_filter_manifest_only requires --motion_filter_manifest_out"
        )
    if args.motion_filter_manifest_only and not args.disable_vis:
        raise ValueError(
            "--motion_filter_manifest_only requires --disable_vis so no visualizer "
            "process is left running after the early exit"
        )

    if args.use_full_imu_ba:
        args.use_imu_ba_prior = True

    if args.use_learned_imu_residual_weights and not args.use_full_imu_ba:
        raise ValueError("--use_learned_imu_residual_weights requires --use_full_imu_ba")
    if args.use_imu_noise_model and not args.use_full_imu_ba:
        raise ValueError("--use_imu_noise_model requires --use_full_imu_ba")
    if args.use_imu_bias_random_walk:
        if not args.use_full_imu_ba:
            raise ValueError(
                "--use_imu_bias_random_walk requires --use_full_imu_ba "
                "with the 15DoF local bias state."
            )
        if not args.use_imu_noise_model:
            raise ValueError(
                "--use_imu_bias_random_walk requires --use_imu_noise_model "
                "so its rotation-information reference is calibrated."
            )
        if args.use_pose_velocity_imu_ba:
            raise ValueError(
                "--use_imu_bias_random_walk is a 15DoF local-bias factor and "
                "cannot be used with --use_pose_velocity_imu_ba."
            )
        print(
            "[IMU] local bias transitions use sensor random-walk covariance; "
            "the legacy --imu_full_bias_weight is disabled for valid IMU rows."
        )
    if args.use_imu_full_covariance_weighting:
        if not args.use_full_imu_ba:
            raise ValueError(
                "--use_imu_full_covariance_weighting requires --use_full_imu_ba."
            )
        if not args.use_imu_noise_model:
            raise ValueError(
                "--use_imu_full_covariance_weighting requires --use_imu_noise_model "
                "and a calibrated --imu_info_reference."
            )
        print(
            "[IMU] full p/v/R covariance-relative weighting enabled; historical "
            "--imu_full_pos_weight/--imu_full_vel_weight are disabled on valid rows."
        )

    # multiprocessing 시작 방식을 spawn으로 설정
    # CUDA + multiprocessing 환경에서 안전하게 process를 만들기 위해 사용
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    # DROID 객체는 첫 이미지가 들어온 뒤 image size를 알 수 있으므로 처음에는 None
    droid = None

    ############################################################
    # 8. reconstruction 저장 시 upsample 강제
    ############################################################

    # save_reconstruction에서는 video.disps_up을 저장한다.
    # 따라서 reconstruction_path가 있으면 upsample을 켜서 고해상도 disparity를 만들도록 함
    if args.reconstruction_path is not None:
        args.upsample = True

    args.imu_prior = maybe_build_imu_prior_from_raw(args)

    imu_priors = load_imu_priors(args.imu_prior)
    args.imu_priors = imu_priors
    validate_imu_bias_jacobian_contract(args, imu_priors)
    (
        args.imu_frame_manifest_path,
        args.frame_records,
    ) = resolve_runtime_frame_records(args, imu_priors)
    runtime_imu_calibration = resolve_runtime_imu_calibration(args)
    args.imu_calibration_fingerprint = validate_imu_calibration_contract(
        args,
        imu_priors,
        runtime_imu_calibration,
    )
    (
        args.imu_camera_translation,
        args.imu_camera_translation_source,
        args.imu_translation_extrinsic_applied,
    ) = resolve_imu_camera_translation(
        args,
        imu_priors,
        runtime_calibration=runtime_imu_calibration,
    )
    translation = np.asarray(args.imu_camera_translation, dtype=np.float64)
    print(
        "[IMU] camera/IMU lever arm t_CI="
        f"({translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}) m, "
        f"source={args.imu_camera_translation_source}, "
        f"applied={int(args.imu_translation_extrinsic_applied)}"
    )
    if (
        (args.use_pose_velocity_imu_ba or args.use_two_stage_imu_ba)
        and not args.imu_translation_extrinsic_applied
    ):
        print(
            "[IMU WARNING] pose-velocity IMU BA is using a zero camera/IMU lever arm. "
            "Provide sensor YAML or --imu_to_cam_translation for acceleration-based motion."
        )
    args.imu_info_reference = estimate_imu_info_reference(imu_priors)
    if args.imu_info_reference is not None:
        print(
            "[IMU] raw-camera information reference="
            f"({args.imu_info_reference[0]:.6g}, "
            f"{args.imu_info_reference[1]:.6g}, "
            f"{args.imu_info_reference[2]:.6g})"
        )

    if args.estimate_imu_gravity:
        if args.imu_gravity is None:
            gravity = estimate_gravity_from_imu_priors(
                imu_priors,
                max_frames=args.imu_gravity_estimate_frames,
                min_norm=args.imu_gravity_estimate_min_norm,
                max_norm=args.imu_gravity_estimate_max_norm,
            )
            if gravity is None:
                print("[IMU WARNING] failed to estimate gravity from imu_prior; using zero gravity.")
            else:
                args.imu_gravity = gravity
                norm = float(np.linalg.norm(np.asarray(gravity, dtype=np.float64)))
                print(
                    "[IMU] estimated gravity="
                    f"({gravity[0]:.6f}, {gravity[1]:.6f}, {gravity[2]:.6f}), "
                    f"norm={norm:.6f}"
                )
        else:
            print("[IMU] --imu_gravity was provided; skipping gravity estimation.")

    if (
        args.use_full_imu_ba
        and args.imu_prior is not None
        and args.imu_gravity is None
        and not args.use_vi_initialization
    ):
        if args.allow_zero_imu_gravity:
            print("[IMU WARNING] full IMU BA is running with zero gravity; use only for debugging.")
        else:
            raise ValueError(
                "--use_full_imu_ba requires gravity for position/velocity residuals. "
                "Use --estimate_imu_gravity, provide --imu_gravity GX GY GZ, or pass "
                "--allow_zero_imu_gravity for a debug-only run."
            )

    if args.use_full_imu_ba and args.use_imu_residual:
        print("[IMU WARNING] --use_full_imu_ba already includes rotation residual; disabling --use_imu_residual.")
        args.use_imu_residual = False

    if args.imu_prior is not None and args.stride != 1:
        print(
            "[IMU] --stride > 1: using original sorted-image indices as timestamps "
            "so skipped IMU rows can be composed between selected frames."
        )

    if args.trajectory_fill_stride < 1:
        raise ValueError("--trajectory_fill_stride must be a positive integer")
    if args.local_tracking_only and args.local_vio_only:
        raise ValueError(
            "choose only one of --local_tracking_only and --local_vio_only"
        )
    if args.local_tracking_only:
        print(
            "[TRAJECTORY] local_tracking_only=1: exporting online selected "
            "keyframe poses only; global BA and trajectory filling are disabled"
        )
    elif args.local_vio_only:
        print(
            "[TRAJECTORY] local_vio_only=1: global BA is disabled; "
            "PoseTrajectoryFiller will export the full camera timeline from "
            "the online local-BA state"
        )
    else:
        print(
            "[TRAJECTORY] tracking_stride="
            f"{args.stride}, fill_stride={args.trajectory_fill_stride} "
            "(fill is applied only after global BA)"
        )

    if args.use_imu_residual and args.imu_prior is None:
        print("[IMU WARNING] --use_imu_residual requires --imu_prior; residual will be disabled.")

    if args.use_imu_ba_prior and args.imu_prior is None:
        print("[IMU WARNING] --use_imu_ba_prior requires --imu_prior; prior will be disabled.")

    if args.use_learned_imu_confidence and not (args.use_imu_residual or args.use_imu_ba_prior):
        print("[IMU WARNING] --use_learned_imu_confidence has no effect without an IMU residual/prior term.")

    if args.use_learned_imu_precision:
        if not args.use_full_imu_ba or not args.use_imu_ba_prior:
            raise ValueError(
                '--use_learned_imu_precision requires --use_full_imu_ba and --use_imu_ba_prior'
            )
        if not args.use_imu_noise_model or not args.use_imu_full_covariance_weighting:
            raise ValueError(
                '--use_learned_imu_precision requires --use_imu_noise_model and '
                '--use_imu_full_covariance_weighting'
            )
        if not args.use_imu_bias_random_walk:
            raise ValueError(
                '--use_learned_imu_precision requires --use_imu_bias_random_walk '
                'so bias rows retain a physical prior'
            )
        if (
            args.use_learned_imu_confidence
            or args.use_component_imu_confidence
            or args.use_dvi_imu_confidence
            or args.use_dvi_imu_quality_fusion
            or args.imu_confidence_only_weighting
            or args.use_learned_imu_ba_weight
            or args.use_learned_imu_residual_weights
            or args.imu_tuned_confidence_schedule is not None
        ):
            raise ValueError(
                '--use_learned_imu_precision is a standalone direct-precision mode; '
                'do not combine it with legacy confidence/lambda/residual-weight controls'
            )

    if (
        args.use_imu_precision_discrepancy
        and not args.use_learned_imu_precision
    ):
        raise ValueError(
            '--use_imu_precision_discrepancy requires --use_learned_imu_precision'
        )
    if args.use_imu_precision_innovation and not args.use_learned_imu_precision:
        raise ValueError(
            '--use_imu_precision_innovation requires --use_learned_imu_precision'
        )
    if (
        args.use_imu_precision_visual_innovation
        and not args.use_imu_precision_innovation
    ):
        raise ValueError(
            '--use_imu_precision_visual_innovation requires '
            '--use_imu_precision_innovation'
        )
    if (
        args.use_imu_precision_edge_fusion
        and not args.use_imu_precision_innovation
    ):
        raise ValueError(
            '--use_imu_precision_edge_fusion requires '
            '--use_imu_precision_innovation'
        )

    if args.use_component_imu_confidence and not args.use_learned_imu_confidence:
        raise ValueError('--use_component_imu_confidence requires --use_learned_imu_confidence')
    if args.use_component_imu_confidence and not args.use_full_imu_ba:
        raise ValueError('--use_component_imu_confidence requires --use_full_imu_ba')
    if args.use_dvi_imu_confidence and not args.use_learned_imu_confidence:
        raise ValueError('--use_dvi_imu_confidence requires --use_learned_imu_confidence')
    if args.use_dvi_imu_confidence and args.use_component_imu_confidence:
        raise ValueError('choose only one of --use_dvi_imu_confidence and --use_component_imu_confidence')
    if args.use_dvi_imu_confidence and not args.use_full_imu_ba:
        raise ValueError('--use_dvi_imu_confidence requires --use_full_imu_ba')
    if args.use_dvi_imu_quality_fusion and not args.use_dvi_imu_confidence:
        raise ValueError(
            '--use_dvi_imu_quality_fusion requires --use_dvi_imu_confidence'
        )
    if args.imu_confidence_only_weighting:
        if not args.use_full_imu_ba:
            raise ValueError(
                '--imu_confidence_only_weighting requires --use_full_imu_ba'
            )
        if not args.use_imu_ba_prior:
            raise ValueError(
                '--imu_confidence_only_weighting requires --use_imu_ba_prior'
            )
        if not args.use_learned_imu_confidence:
            raise ValueError(
                '--imu_confidence_only_weighting requires '
                '--use_learned_imu_confidence'
            )
        if not (args.use_dvi_imu_confidence or args.use_component_imu_confidence):
            raise ValueError(
                '--imu_confidence_only_weighting requires DVI or component '
                'IMU confidence'
            )
        if args.use_learned_imu_ba_weight:
            raise ValueError(
                '--imu_confidence_only_weighting cannot be combined with '
                '--use_learned_imu_ba_weight'
            )
    if args.imu_tuned_confidence_schedule is not None:
        if args.use_learned_imu_confidence:
            raise ValueError('--imu_tuned_confidence_schedule cannot be combined with learned confidence')
        if args.use_component_imu_confidence or args.use_dvi_imu_confidence:
            raise ValueError('--imu_tuned_confidence_schedule is a scalar fixed/tuned control')
        if not args.use_full_imu_ba:
            raise ValueError('--imu_tuned_confidence_schedule requires --use_full_imu_ba')
        if any(value < 0.0 or value > 1.0 for value in args.imu_tuned_confidence_schedule):
            raise ValueError('--imu_tuned_confidence_schedule values must be in [0, 1]')

    ############################################################
    # 9. 이미지 스트림 처리
    ############################################################

    # The exact input-frame IDs supplied to the tracker. They are later paired
    # with the dense poses returned by ``terminate()``.
    tstamps = []
    selected_motion_filter_rows = []
    frame_timestamp_by_index = {}
    if args.frame_records is not None:
        frame_timestamp_by_index = {
            int(record["frame_index"]): int(record["timestamp_ns"])
            for record in args.frame_records
        }

    # image_stream에서 frame을 하나씩 받아 DROID에 넣음
    for (t, image, intrinsics) in tqdm(
        image_stream(
            args.imagedir,
            args.calib,
            args.stride,
            args.max_frames,
            frame_records=args.frame_records,
            euroc_original_protocol=args.euroc_original_protocol,
        )
    ):

        ########################################################
        # 9-1. 시작 frame 이전은 skip
        ########################################################

        if t < args.t0:
            continue

        tstamps.append(int(t))

        ########################################################
        # 9-2. 입력 이미지 표시
        ########################################################

        # visualization을 끄지 않았다면 현재 입력 frame을 OpenCV 창으로 표시
        if not args.disable_vis:
            show_image(image[0])

        ########################################################
        # 9-3. DROID 객체 최초 생성
        ########################################################

        # 첫 frame에서 실제 image size를 알게 되므로
        # 이때 Droid 또는 DroidAsync 객체를 생성한다.
        if droid is None:

            # image.shape:
            #   [1, 3, H, W]
            #
            # args.image_size:
            #   [H, W]
            args.image_size = [image.shape[2], image.shape[3]]

            # 비동기 옵션이면 DroidAsync,
            # 아니면 일반 Droid 사용
            droid = DroidAsync(args) if args.asynchronous else Droid(args)
        
        ########################################################
        # 9-4. DROID tracking 수행
        ########################################################

        # 현재 frame을 DROID-SLAM에 입력
        #
        # 내부 흐름:
        #   Droid.track()
        #       -> MotionFilter.track()
        #       -> DroidFrontend()
        #
        # 또는 DroidAsync.track()
        #       -> MotionFilter.track()
        #       -> DroidFrontend()
        #       -> backend process는 별도로 video2 refinement 수행
        imu_prior = get_imu_prior_for_frame(imu_priors, t)

        if imu_prior is not None and t < 5:
            print(
                f"[IMU->demo] frame={t}, "
                f"valid={imu_prior.get('imu_valid')}, "
                f"dt={imu_prior.get('dt', 0.0):.6f}, "
                f"dr=({imu_prior.get('dr_x', 0.0):.6f}, "
                f"{imu_prior.get('dr_y', 0.0):.6f}, "
                f"{imu_prior.get('dr_z', 0.0):.6f}), "
                f"dp=({imu_prior.get('dp_x', 0.0):.6f}, "
                f"{imu_prior.get('dp_y', 0.0):.6f}, "
                f"{imu_prior.get('dp_z', 0.0):.6f}), "
                f"imu_count={imu_prior.get('imu_count', 0)}"
            )

        if args.motion_filter_manifest_only:
            # MotionFilter's decision depends only on its own hidden/context
            # state, the current visual image, and the previous selected
            # frame. Frontend/backend updates do not feed back into it. Skip
            # them here so a training manifest does not pay for an unrelated
            # full-sequence reconstruction.
            was_selected = droid.filterx.track(
                t,
                image,
                intrinsics=intrinsics,
                imu_prior=imu_prior,
            )
        else:
            was_selected = droid.track(
                t,
                image,
                intrinsics=intrinsics,
                imu_prior=imu_prior,
            )
        if args.motion_filter_manifest_out is not None and was_selected:
            if int(t) not in frame_timestamp_by_index:
                raise RuntimeError(
                    "MotionFilter manifest requires --imu_frame_manifest so "
                    "each selected original frame index has a verified timestamp"
                )
            selected_motion_filter_rows.append(
                {
                    "frame_index": int(t),
                    "timestamp_ns": frame_timestamp_by_index[int(t)],
                }
            )

    ############################################################
    # 10. DROID 종료 및 trajectory 생성
    ############################################################

    if droid is None:
        raise RuntimeError("No images were processed. Check --imagedir and image files.")

    if args.motion_filter_manifest_out is not None:
        write_motion_filter_manifest(
            args.motion_filter_manifest_out,
            selected_motion_filter_rows,
        )

    if args.motion_filter_manifest_only:
        print("[MOTION-FILTER MANIFEST] frontend/backend refinement skipped")
        raise SystemExit(0)

    if args.local_tracking_only:
        # Do not call ``terminate()`` here. It runs one or more global backend
        # passes and then PoseTrajectoryFiller, so either would turn this into
        # an offline SLAM result. ``video.poses`` and ``video.tstamp`` retain
        # the online frontend state for exactly the selected tracking frames.
        if hasattr(droid, "video2"):
            raise ValueError(
                "--local_tracking_only currently supports synchronous Droid only; "
                "DroidAsync has an independent backend process"
            )
        if hasattr(droid.frontend, "flush_imu_debug"):
            droid.frontend.flush_imu_debug()
        local_keyframes = int(droid.video.counter.value)
        if local_keyframes < 5:
            raise RuntimeError(
                "local tracking produced fewer than five selected keyframes; "
                "cannot evaluate a trajectory"
            )
        dense_tstamps = None
        traj_est = None
        print(
            "[TRAJECTORY] local_keyframes="
            f"{local_keyframes}, global_backend=skipped, pose_filler=skipped"
        )
    else:
        # Both paths below fill poses on the raw camera time axis. The normal
        # path first runs global BA; local_vio_only deliberately skips it and
        # fills from the online local-BA state. The strides must be separate:
        # the official DROID EuRoC protocol tracks every other 20 Hz image,
        # but fills the trajectory on every raw 20 Hz image.
        # Keeping --stride=2 preserves the trained IMU graph; setting
        # --trajectory_fill_stride=1 changes only the final output timeline.
        #
        # With --max_frames, restrict filling to the raw-time span covered by the
        # selected tracking window. Without this guard, a short debug run would
        # extrapolate poses for the remainder of a full sequence.
        fill_stop_t = None
        if args.max_frames is not None and args.max_frames > 0:
            fill_stop_t = int(tstamps[-1]) + max(0, int(args.stride) - 1)

        dense_tstamps = []

        def dense_image_stream():
            for t, image, intrinsics in image_stream(
                args.imagedir,
                args.calib,
                args.trajectory_fill_stride,
                0,
                frame_records=args.frame_records,
                euroc_original_protocol=args.euroc_original_protocol,
            ):
                if t < args.t0:
                    continue
                if fill_stop_t is not None and t > fill_stop_t:
                    break
                dense_tstamps.append(int(t))
                yield t, image, intrinsics

        if args.local_vio_only:
            if hasattr(droid, "video2"):
                raise ValueError(
                    "--local_vio_only currently supports synchronous Droid only; "
                    "DroidAsync has an independent backend process"
                )
            traj_est = droid.terminate_local(dense_image_stream())
        else:
            traj_est = droid.terminate(dense_image_stream())
        if len(traj_est) != len(dense_tstamps):
            raise RuntimeError(
                "dense trajectory length does not match the trajectory-fill timeline: "
                f"poses={len(traj_est)}, frame_ids={len(dense_tstamps)}"
            )
        if args.trajectory_fill_stride == 1 and len(dense_tstamps) < len(tstamps):
            raise RuntimeError(
                "full-rate trajectory filling produced fewer poses than tracking; "
                "check frame-manifest/image-stream alignment"
            )
        print(
            "[TRAJECTORY] tracking_frames="
            f"{len(tstamps)}, dense_output_frames={len(dense_tstamps)}, "
            f"global_backend={'skipped' if args.local_vio_only else 'enabled'}"
        )

    if args.use_vi_initialization:
        aligned = bool(droid.video.imu_alignment_ready.value)
        if aligned:
            metrics = getattr(droid.video, "imu_alignment_metrics", {})
            print(
                "[VI-INIT] final status=aligned, "
                f"scale={float(droid.video.imu_alignment_scale[0].item()):.6f}, "
                f"edges={int(metrics.get('edges', 0.0))}"
            )
        else:
            print(
                "[VI-INIT WARNING] final status=not_aligned; full 9DoF IMU "
                "factors were intentionally withheld, so this reconstruction "
                "must be treated as visual-only rather than a metric VI result."
            )
    
    ############################################################
    # 11. reconstruction 저장
    ############################################################

    # reconstruction_path가 지정되어 있으면
    # pose, disparity, image, intrinsics 등을 .pth 파일로 저장
    if args.reconstruction_path is not None:
        if args.local_tracking_only:
            # No dense trajectory is written: a dense export would imply the
            # offline PoseTrajectoryFiller had been run. The evaluator must
            # use --trajectory-source keyframes for this diagnostic result.
            save_reconstruction(droid, args.reconstruction_path)
        else:
            save_reconstruction(
                droid,
                args.reconstruction_path,
                dense_tstamps=dense_tstamps,
                dense_poses=traj_est,
                dense_protocol={
                    "tracking_stride": int(args.stride),
                    "trajectory_fill_stride": int(args.trajectory_fill_stride),
                    "tracking_frame_count": int(len(tstamps)),
                    "dense_frame_count": int(len(dense_tstamps)),
                    "max_tracking_frames": int(args.max_frames),
                    "euroc_original_protocol": bool(args.euroc_original_protocol),
                    "local_tracking_only": False,
                    "local_vio_only": bool(args.local_vio_only),
                    "global_backend": not bool(args.local_vio_only),
                    "trajectory_filler": True,
                },
            )

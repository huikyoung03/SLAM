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
from pathlib import Path

from torch.multiprocessing import Process

# DROID-SLAM 기본 동기식 실행 클래스
from droid import Droid

# DROID-SLAM 비동기 실행 클래스
# frontend와 backend를 분리해서 실행할 때 사용
from droid_async import DroidAsync

try:
    from data_readers.imu_preintegration import build_imu_prior_csv
except Exception:
    build_imu_prior_csv = None

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


def _to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def load_imu_priors(imu_prior_path):
    """
    imu_prior.csv를 frame_index 기준 dictionary로 읽는다.
    """

    if imu_prior_path is None:
        return {}

    imu_prior_path = Path(imu_prior_path)
    if not imu_prior_path.exists():
        print(f"[IMU] imu_prior.csv not found: {imu_prior_path}")
        return {}

    priors = {}

    with open(imu_prior_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame_index = _to_int(
                row.get("frame_index"),
                _to_int(row.get("frame_id"), len(priors)),
            )

            priors[frame_index] = {
                "frame_id": _to_int(row.get("frame_id"), frame_index),
                "frame_index": frame_index,
                "timestamp_sec": _to_float(row.get("timestamp_sec")),
                "timestamp_ns": _to_int(row.get("timestamp_ns")),
                "prev_timestamp_sec": _to_float(row.get("prev_timestamp_sec")),
                "dt": _to_float(row.get("dt")),
                "imu_count": _to_int(row.get("imu_count")),
                "imu_used_steps": _to_int(
                    row.get("imu_used_steps"),
                    _to_int(row.get("imu_count")),
                ),
                "imu_valid": _to_int(row.get("imu_valid"), 1),
                "imu_weight": _to_float(row.get("imu_weight"), 1.0),
                "imu_reason": row.get("imu_reason", ""),
                "dr_x": _to_float(row.get("dr_x")),
                "dr_y": _to_float(row.get("dr_y")),
                "dr_z": _to_float(row.get("dr_z")),
                "dq_w": _to_float(row.get("dq_w"), 1.0),
                "dq_x": _to_float(row.get("dq_x")),
                "dq_y": _to_float(row.get("dq_y")),
                "dq_z": _to_float(row.get("dq_z")),
                "dv_x": _to_float(row.get("dv_x")),
                "dv_y": _to_float(row.get("dv_y")),
                "dv_z": _to_float(row.get("dv_z")),
                "dp_x": _to_float(row.get("dp_x")),
                "dp_y": _to_float(row.get("dp_y")),
                "dp_z": _to_float(row.get("dp_z")),
                "rot_var": _to_float(row.get("rot_var")),
                "vel_var": _to_float(row.get("vel_var")),
                "pos_var": _to_float(row.get("pos_var")),
                "rot_info": _to_float(row.get("rot_info")),
                "vel_info": _to_float(row.get("vel_info")),
                "pos_info": _to_float(row.get("pos_info")),
                "imu_extrinsic_applied": _to_int(row.get("imu_extrinsic_applied")),
                "imu_calibration_source": row.get("imu_calibration_source", ""),
            }

    print(f"[IMU] loaded {len(priors)} imu priors from {imu_prior_path}")
    return priors


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
        gyro_bias=args.gyro_bias,
        acc_bias=args.acc_bias,
    )

    return output


def image_stream(imagedir, calib, stride, max_frames=None):
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

    ############################################################
    # 3. 이미지 목록 구성
    ############################################################

    # imagedir 안의 실제로 읽을 수 있는 이미지 파일명만 정렬한 뒤 stride 간격으로 선택
    #
    # sorted를 쓰는 이유:
    #   프레임 순서가 중요하기 때문.
    #
    # [::stride]:
    #   일정 간격으로 프레임을 샘플링.
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

    image_list = image_list[::stride]
    if max_frames is not None and max_frames > 0:
        image_list = image_list[:int(max_frames)]

    if len(image_list) == 0:
        raise RuntimeError(
            f"No readable images found in {imagedir}. "
            "Check that the directory contains real image files or valid symlinks."
        )

    ############################################################
    # 4. 이미지 하나씩 읽어서 yield
    ############################################################

    for t, imfile in enumerate(image_list):

        # 이미지 읽기
        # cv2.imread는 BGR 순서로 이미지를 읽는다.
        image_path = os.path.join(imagedir, imfile)
        image = cv2.imread(image_path)

        if image is None:
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
        if len(calib) > 4:
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


def save_reconstruction(droid, save_path):
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

    ############################################################
    # 3. 저장할 데이터 구성
    ############################################################

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
        "imu_unit_scale": video.imu_unit_scale.cpu(),

        # intrinsics
        "intrinsics": video.intrinsics[:t].cpu()
    }

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
    parser.add_argument("--stride", default=3, type=int, help="frame stride")
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
    parser.add_argument("--imu_frames", type=str, default=None, help="path to frames.csv for building imu_prior.csv")
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
    parser.add_argument("--use_imu_translation_prior", action="store_true", help="experimental: initialize pose translation from IMU dp")
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
        "--learned_imu_ba_weight_scale",
        type=float,
        default=1.0,
        help="scale applied to the checkpoint-learned global IMU BA weight at runtime",
    )
    parser.add_argument("--imu_ba_prior_max_deg", type=float, default=10.0, help="skip IMU DBA prior edges above this angular error")
    parser.add_argument("--use_full_imu_ba", action="store_true", help="extend runtime CUDA DBA state to pose+velocity+accel-bias+gyro-bias")
    parser.add_argument("--imu_full_frontend", action="store_true", help="also use full 15D IMU BA in frontend updates")
    parser.add_argument("--imu_full_pos_weight", type=float, default=0.05, help="position residual weight inside full IMU DBA prior")
    parser.add_argument("--imu_full_vel_weight", type=float, default=0.05, help="velocity residual weight inside full IMU DBA prior")
    parser.add_argument("--imu_full_bias_weight", type=float, default=0.001, help="bias smoothness residual weight inside full IMU DBA prior")
    parser.add_argument("--use_imu_info_weighting", action="store_true", help="weight full IMU residuals by median-normalized preintegration information")
    parser.add_argument("--imu_info_weight_clip", type=float, default=4.0, help="clip for relative IMU information residual scales")
    parser.add_argument("--imu_info_weight_eps", type=float, default=1e-12, help="epsilon for IMU information weighting")
    parser.add_argument("--imu_motion_prior_weight", type=float, default=0.0, help="optional weak velocity prior weight for full runtime IMU BA")
    parser.add_argument("--imu_local_bias_prior_weight", type=float, default=0.0, help="optional weak local bias prior weight for full runtime IMU BA")
    parser.add_argument("--imu_gravity", type=float, nargs=3, default=None, metavar=("GX", "GY", "GZ"), help="gravity vector in world/DROID units for full IMU residuals")
    parser.add_argument("--estimate_imu_gravity", action="store_true", help="estimate initial gravity from imu_prior delta-velocity rows")
    parser.add_argument("--allow_zero_imu_gravity", action="store_true", help="debug only: allow full IMU BA to run with zero gravity")
    parser.add_argument("--imu_gravity_estimate_frames", type=int, default=80, help="max valid imu_prior rows used for gravity estimation")
    parser.add_argument("--imu_gravity_estimate_min_norm", type=float, default=6.0, help="minimum |dv/dt| used for gravity estimation")
    parser.add_argument("--imu_gravity_estimate_max_norm", type=float, default=12.5, help="maximum |dv/dt| used for gravity estimation")
    parser.add_argument("--imu_full_max_dt", type=float, default=0.5, help="skip full IMU BA edges above this preintegrated dt")
    parser.add_argument("--imu_full_max_dv", type=float, default=5.0, help="skip full IMU BA edges above this delta-velocity norm")
    parser.add_argument("--imu_full_max_dp", type=float, default=1.0, help="skip full IMU BA edges above this delta-position norm")
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

    # demo.py는 monocular 입력을 기준으로 사용
    args.stereo = False

    if args.use_full_imu_ba:
        args.use_imu_ba_prior = True

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

    if args.use_full_imu_ba and args.imu_prior is not None and args.imu_gravity is None:
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
        print("[IMU WARNING] imu_prior frame matching is safest with --stride=1.")

    if args.use_imu_residual and args.imu_prior is None:
        print("[IMU WARNING] --use_imu_residual requires --imu_prior; residual will be disabled.")

    if args.use_imu_ba_prior and args.imu_prior is None:
        print("[IMU WARNING] --use_imu_ba_prior requires --imu_prior; prior will be disabled.")

    if args.use_learned_imu_confidence and not (args.use_imu_residual or args.use_imu_ba_prior):
        print("[IMU WARNING] --use_learned_imu_confidence has no effect without an IMU residual/prior term.")

    ############################################################
    # 9. 이미지 스트림 처리
    ############################################################

    # timestamp 목록.
    # 현재 코드에서는 append만 하지 않고 선언만 되어 있음.
    tstamps = []

    # image_stream에서 frame을 하나씩 받아 DROID에 넣음
    for (t, image, intrinsics) in tqdm(
        image_stream(args.imagedir, args.calib, args.stride, args.max_frames)
    ):

        ########################################################
        # 9-1. 시작 frame 이전은 skip
        ########################################################

        if t < args.t0:
            continue

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

        droid.track(
            t,
            image,
            intrinsics=intrinsics,
            imu_prior=imu_prior,
        )

    ############################################################
    # 10. DROID 종료 및 trajectory 생성
    ############################################################

    if droid is None:
        raise RuntimeError("No images were processed. Check --imagedir and image files.")

    # terminate()는 backend refinement를 마무리하고,
    # keyframe이 아닌 skipped frame의 pose까지 채운 trajectory를 반환한다.
    #
    # image_stream을 다시 넘기는 이유:
    #   PoseTrajectoryFiller가 원본 stream의 전체 frame에 대한 pose를 채울 때
    #   다시 이미지 stream을 참고할 수 있기 때문.
    traj_est = droid.terminate(
        image_stream(args.imagedir, args.calib, args.stride, args.max_frames)
    )
    
    ############################################################
    # 11. reconstruction 저장
    ############################################################

    # reconstruction_path가 지정되어 있으면
    # pose, disparity, image, intrinsics 등을 .pth 파일로 저장
    if args.reconstruction_path is not None:
        save_reconstruction(
            droid,
            args.reconstruction_path
        )

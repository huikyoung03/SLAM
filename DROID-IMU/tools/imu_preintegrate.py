"""
프레임 사이 IMU 사전적분(preintegration) CSV 생성 스크립트.

이 파일은 SLAM 서버가 수집한 한 세션의 `frames.csv`와 `imu.csv`를 읽어서
각 이미지 프레임 i에 대해 "이전 프레임 i-1 시각부터 현재 프레임 i 시각까지"의
IMU 누적값을 `imu_prior.csv`로 저장한다.

출력 row의 의미:
    dt:
        이전 프레임과 현재 프레임 사이 시간 간격 [sec].

    dr_x, dr_y, dr_z:
        프레임 사이 gyro를 SO(3) 위에서 누적한 뒤 log-map으로 변환한 회전 벡터 [rad].
        단순히 gyro * dt를 더한 값이 아니라, 작은 quaternion step들을 순서대로 곱한
        최종 delta rotation을 다시 rotation vector로 표현한 값이다.

    dq_w, dq_x, dq_y, dq_z:
        같은 delta rotation의 quaternion 표현. 저장 순서는 w, x, y, z이다.

    dv_x, dv_y, dv_z:
        가속도를 시간 적분한 delta velocity. 각 IMU step의 중간 회전(q_mid)으로
        accel을 회전시킨 뒤 적분한다.

    dp_x, dp_y, dp_z:
        delta position. 위의 회전 보정된 accel을 속도와 함께 한 번 더 적분한다.

    imu_valid, imu_weight, imu_reason:
        후단에서 이 prior를 얼마나 믿어도 되는지 판단하기 위한 품질 플래그.

주의:
    이 스크립트는 선택적으로 IMU-camera extrinsic과 상수 bias를 적용할 수 있다. 다만
    gravity 제거와 bias 재추정은 아직 하지 않는다. 따라서 `dr/dq`는 비교적 바로 쓸 수
    있지만, `dv/dp`를 pose translation prior로 강하게 쓰려면 gravity, scale, bias 초기화를
    별도로 잡아야 한다.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


Matrix3 = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
Vector3 = Tuple[float, float, float]


IDENTITY_R: Matrix3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


@dataclass(frozen=True)
class ImuCalibration:
    """Static calibration applied before preintegration."""

    R_imu_to_camera: Matrix3 = IDENTITY_R
    extrinsic_applied: bool = False
    source: str = "identity"
    gyro_bias: Vector3 = (0.0, 0.0, 0.0)
    acc_bias: Vector3 = (0.0, 0.0, 0.0)
    gyro_noise_density: float = 0.0
    gyro_random_walk: float = 0.0
    accel_noise_density: float = 0.0
    accel_random_walk: float = 0.0


def to_float(value: Any, default: float = 0.0) -> float:
    """
    CSV에서 읽은 값을 안전하게 float로 변환한다.

    센서 로그는 브라우저/앱/중간 저장 과정에서 빈 문자열, None, nan, inf가 섞일 수 있다.
    preintegration 중 하나라도 비정상 숫자가 들어오면 전체 row가 망가지므로,
    이 함수에서 기본값으로 정리한다.
    """

    try:
        if value is None or value == "":
            return default

        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default

        return out

    except Exception:
        return default


def to_int_ns(row: Dict[str, Any]) -> int:
    """
    CSV row의 timestamp를 nanosecond 정수로 통일한다.

    우선순위:
        1. `timestamp_ns`가 있으면 그대로 사용
        2. 없으면 `timestamp_sec`를 1e9 배 해서 ns로 변환

    내부 적분은 시간 간격 차이를 많이 계산하므로 float second보다 int ns를 기준으로
    창을 선택하고, 필요한 순간에만 sec 단위 dt로 바꾼다.
    """

    if row.get("timestamp_ns") not in (None, ""):
        return int(float(row["timestamp_ns"]))

    if row.get("timestamp_sec") not in (None, ""):
        return int(to_float(row.get("timestamp_sec")) * 1_000_000_000)

    # EuRoC uses headers such as "#timestamp [ns]" for cam0/imu0 CSV files.
    # Accept timestamp-like external CSV headers without forcing a preprocessing
    # conversion step.
    for key, value in row.items():
        if value in (None, ""):
            continue

        normalized = key.lower().replace("#", "").strip()
        if "timestamp" not in normalized:
            continue

        stamp = to_float(value)
        if "[ns]" in normalized or stamp > 1.0e12:
            return int(stamp)
        return int(stamp * 1_000_000_000)

    return 0


def get_float_any(
    row: Dict[str, Any],
    names: Tuple[str, ...],
    default: float = 0.0,
) -> float:
    """
    같은 물리량에 대해 여러 CSV 컬럼명을 허용한다.

    현재 서버는 `gx, gy, gz, ax, ay, az`를 쓰지만, 과거 로그나 다른 수집 코드에서는
    `gyro_x`, `acc_x`, `wx` 같은 이름을 쓸 수 있다. preintegration 자체는 컬럼명보다
    값의 의미가 중요하므로, 가능한 후보를 순서대로 찾는다.
    """

    for name in names:
        if row.get(name) not in (None, ""):
            return to_float(row.get(name), default)

    def normalize(name: str) -> str:
        name = name.lower()
        name = re.sub(r"\[[^\]]*\]", "", name)
        name = re.sub(r"[^a-z0-9]+", "_", name)
        return name.strip("_")

    normalized_names = {normalize(name) for name in names}
    for key, value in row.items():
        if value in (None, ""):
            continue
        if normalize(key) in normalized_names:
            return to_float(value, default)

    return default


def mat3_transpose(R: Matrix3) -> Matrix3:
    return tuple(tuple(R[j][i] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def mat3_multiply(A: Matrix3, B: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def mat3_vec(R: Matrix3, v: Vector3) -> Vector3:
    return (
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    )


def parse_vector3(values: Optional[Sequence[float]]) -> Vector3:
    if values is None:
        return 0.0, 0.0, 0.0

    if len(values) != 3:
        raise ValueError("Expected exactly 3 values")

    return float(values[0]), float(values[1]), float(values[2])


def parse_matrix3(values: Optional[Sequence[float]]) -> Optional[Matrix3]:
    if values is None:
        return None

    if len(values) != 9:
        raise ValueError("Expected exactly 9 values for a 3x3 matrix")

    vals = [float(v) for v in values]
    return (
        (vals[0], vals[1], vals[2]),
        (vals[3], vals[4], vals[5]),
        (vals[6], vals[7], vals[8]),
    )


def read_yaml_matrix4(path: Path, key: str = "T_BS") -> List[List[float]]:
    """
    Read a small EuRoC-style 4x4 matrix from `sensor.yaml`.

    PyYAML is intentionally not required. EuRoC stores `T_BS` as:

        T_BS:
          cols: 4
          rows: 4
          data: [...]
    """

    text = path.read_text(encoding="utf-8")
    block_match = re.search(rf"{re.escape(key)}\s*:\s*(.*?)(?:\n\S|\Z)", text, re.S)
    search_text = block_match.group(1) if block_match else text
    data_match = re.search(r"data\s*:\s*\[([^\]]+)\]", search_text, re.S)
    if data_match is None:
        raise ValueError(f"Cannot find {key}.data in {path}")

    values = [
        float(v)
        for v in re.split(r"[,\s]+", data_match.group(1).strip())
        if v
    ]
    if len(values) != 16:
        raise ValueError(f"{path} {key}.data must contain 16 numbers, got {len(values)}")

    return [values[i:i + 4] for i in range(0, 16, 4)]


def read_yaml_scalar(path: Optional[Path], key: str, default: float = 0.0) -> float:
    if path is None or not path.exists():
        return default

    text = path.read_text(encoding="utf-8")
    match = re.search(rf"^\s*{re.escape(key)}\s*:\s*([^\s#]+)", text, re.M)
    if match is None:
        return default

    return to_float(match.group(1), default)


def rotation_from_t_bs(path: Path) -> Matrix3:
    T = read_yaml_matrix4(path, "T_BS")
    return (
        (T[0][0], T[0][1], T[0][2]),
        (T[1][0], T[1][1], T[1][2]),
        (T[2][0], T[2][1], T[2][2]),
    )


def build_imu_calibration(
    cam_sensor_yaml: Optional[Path] = None,
    imu_sensor_yaml: Optional[Path] = None,
    imu_to_cam_rotation: Optional[Sequence[float]] = None,
    gyro_bias: Optional[Sequence[float]] = None,
    acc_bias: Optional[Sequence[float]] = None,
) -> ImuCalibration:
    """
    Build the static transform used to express IMU measurements in camera frame.

    EuRoC's `T_BS` is sensor-to-body. For vectors:

        v_B = R_BI v_I
        v_C = R_BC^T v_B

    so `R_CI = R_BC^T R_BI`.
    """

    R_cli = parse_matrix3(imu_to_cam_rotation)
    source = "identity"
    extrinsic_applied = False

    if R_cli is not None:
        R_imu_to_camera = R_cli
        source = "cli_imu_to_cam_rotation"
        extrinsic_applied = True
    elif cam_sensor_yaml is not None and imu_sensor_yaml is not None:
        R_body_cam = rotation_from_t_bs(cam_sensor_yaml)
        R_body_imu = rotation_from_t_bs(imu_sensor_yaml)
        R_imu_to_camera = mat3_multiply(mat3_transpose(R_body_cam), R_body_imu)
        source = f"{cam_sensor_yaml}:{imu_sensor_yaml}"
        extrinsic_applied = True
    else:
        R_imu_to_camera = IDENTITY_R

    imu_yaml = imu_sensor_yaml if imu_sensor_yaml is not None else None
    return ImuCalibration(
        R_imu_to_camera=R_imu_to_camera,
        extrinsic_applied=extrinsic_applied,
        source=source,
        gyro_bias=parse_vector3(gyro_bias),
        acc_bias=parse_vector3(acc_bias),
        gyro_noise_density=read_yaml_scalar(imu_yaml, "gyroscope_noise_density"),
        gyro_random_walk=read_yaml_scalar(imu_yaml, "gyroscope_random_walk"),
        accel_noise_density=read_yaml_scalar(imu_yaml, "accelerometer_noise_density"),
        accel_random_walk=read_yaml_scalar(imu_yaml, "accelerometer_random_walk"),
    )


def estimate_preintegration_uncertainty(
    dt: float,
    calibration: ImuCalibration,
) -> Dict[str, float]:
    """
    Approximate diagonal covariance/info for future IMU factors.

    This is a first-order metadata estimate, not full Forster-style covariance
    propagation. It is intentionally conservative and only meant to provide a
    stable initial `w^u`/information source before learned confidence exists.
    """

    dt = max(float(dt), 0.0)
    gyro_sigma = max(float(calibration.gyro_noise_density), 0.0)
    acc_sigma = max(float(calibration.accel_noise_density), 0.0)

    rot_var = gyro_sigma * gyro_sigma * dt if gyro_sigma > 0.0 and dt > 0.0 else 0.0
    vel_var = acc_sigma * acc_sigma * dt if acc_sigma > 0.0 and dt > 0.0 else 0.0
    pos_var = (
        acc_sigma * acc_sigma * dt * dt * dt / 3.0
        if acc_sigma > 0.0 and dt > 0.0
        else 0.0
    )

    return {
        "rot_var": rot_var,
        "vel_var": vel_var,
        "pos_var": pos_var,
        "rot_info": 1.0 / rot_var if rot_var > 1e-18 else 0.0,
        "vel_info": 1.0 / vel_var if vel_var > 1e-18 else 0.0,
        "pos_info": 1.0 / pos_var if pos_var > 1e-18 else 0.0,
    }


def convert_gyro_device_to_camera(
    gx: float,
    gy: float,
    gz: float,
) -> Tuple[float, float, float]:
    """
    Android/browser device gyro -> camera gyro conversion hook.

    현재는 identity mapping이다.

    즉, 입력 gyro 축을 그대로 DROID 쪽 camera 축이라고 가정한다. 실제 스마트폰/브라우저
    DeviceMotionEvent 축과 카메라 optical frame은 보통 완전히 같지 않을 수 있으므로,
    나중에 IMU prior가 반대 방향으로 작동하거나 특정 축이 뒤집혀 보이면 이 함수에서
    축 변환을 고정하거나 `demo.py`의 axis sign 옵션으로 보정해야 한다.
    """

    return gx, gy, gz


def convert_accel_device_to_camera(
    ax: float,
    ay: float,
    az: float,
) -> Tuple[float, float, float]:
    """
    Android/browser device accel -> camera accel conversion hook.

    현재는 identity mapping이다.

    accel도 gyro와 같은 축 변환 문제가 있다. 특히 accel은 gravity와 bias 영향까지 같이
    받기 때문에, `dp/dv`를 강한 translation prior로 쓰기 전에는 축 정렬, gravity 처리,
    bias 보정 여부를 반드시 확인해야 한다.
    """

    return ax, ay, az


def rotvec_to_quat_wxyz(
    rx: float,
    ry: float,
    rz: float,
) -> Tuple[float, float, float, float]:
    """
    rotation vector(axis-angle)를 quaternion(w, x, y, z)으로 변환한다.

    입력 `r = [rx, ry, rz]`의 방향은 회전축, 크기 `||r||`는 회전각 [rad]이다.
    gyro 한 step을 `omega * dt` 형태의 작은 rotation vector로 만든 뒤 이 함수로
    quaternion step을 생성한다.
    """

    theta = math.sqrt(rx * rx + ry * ry + rz * rz)

    if theta < 1e-12:
        return 1.0, 0.0, 0.0, 0.0

    ax = rx / theta
    ay = ry / theta
    az = rz / theta

    half = 0.5 * theta
    s = math.sin(half)

    return math.cos(half), ax * s, ay * s, az * s


def quat_normalize_wxyz(
    q: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """
    quaternion을 unit quaternion으로 정규화한다.

    작은 회전 step을 수백/수천 번 곱하면 부동소수점 오차 때문에 norm이 1에서 조금씩
    벗어날 수 있다. 회전 quaternion은 항상 unit norm이어야 하므로 곱셈 직후 정규화한다.
    """

    w, x, y, z = q
    norm = math.sqrt(w * w + x * x + y * y + z * z)

    if norm < 1e-12:
        return 1.0, 0.0, 0.0, 0.0

    return w / norm, x / norm, y / norm, z / norm


def quat_multiply_wxyz(
    q1: Tuple[float, float, float, float],
    q2: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """
    두 quaternion의 Hamilton product를 계산한다.

    여기서는 `q1` 이후에 작은 delta rotation `q2`를 적용하는 형태로 사용한다.
    적분 루프에서 `dq = dq * step`을 반복하면 프레임 구간 전체의 누적 회전이 된다.
    """

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return quat_normalize_wxyz((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ))


def quat_rotate_vector_wxyz(
    q: Tuple[float, float, float, float],
    v: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """
    quaternion q로 3D vector v를 회전시킨다.

    수식은 `q * [0, v] * q^-1`이다. accel은 센서/body 좌표계에서 측정되므로,
    preintegration 구간의 누적 회전 중간값으로 accel을 회전시킨 뒤 dv/dp에 반영한다.
    """

    w, x, y, z = quat_normalize_wxyz(q)
    vx, vy, vz = v

    # q * [0, v] * q^-1, expanded to avoid temporary quaternion objects.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def quat_to_rotvec_wxyz(
    q: Tuple[float, float, float, float],
) -> Tuple[float, float, float]:
    """
    unit quaternion을 rotation vector(log-map)로 변환한다.

    후단의 `MotionFilter`는 `dr_x/y/z`의 norm과 축 성분을 바로 사용하므로,
    누적 quaternion `dq`를 다시 axis-angle vector로 저장한다. q와 -q는 같은 회전을
    뜻하기 때문에 w가 음수이면 부호를 뒤집어 가장 짧은 회전 표현을 사용한다.
    """

    w, x, y, z = quat_normalize_wxyz(q)

    # Use the shortest equivalent rotation for a stable log map.
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z

    sin_half = math.sqrt(x * x + y * y + z * z)
    if sin_half < 1e-12:
        return 0.0, 0.0, 0.0

    angle = 2.0 * math.atan2(sin_half, w)
    scale = angle / sin_half

    return x * scale, y * scale, z * scale


def compute_imu_weight(
    dt: float,
    imu_count: int,
    used_steps: int,
    dr_norm_deg: float,
    has_nan: bool,
    window_status: str,
) -> Tuple[float, int, str]:
    """
    preintegrated IMU row의 사용 가능 여부와 기본 가중치를 결정한다.

    반환값:
        weight:
            후단 residual/prior에서 사용할 수 있는 기본 신뢰도. 현재 DROID 연결부에서는
            주로 valid flag와 reason이 중요하고, weight는 추후 residual/factor에서 쓸 수
            있도록 같이 저장한다.

        valid:
            1이면 사용할 수 있는 row, 0이면 후단에서 무시해야 하는 row.

        reason:
            왜 valid/invalid가 되었는지 사람이 CSV만 보고도 알 수 있게 남기는 문자열.

    이 함수는 "수치적으로 말이 안 되는 row"를 막는 방어막이다. 예를 들어 프레임 사이
    시간이 너무 길거나, IMU window가 프레임 구간 밖에 있거나, 회전량이 비정상적으로
    크면 후단 SLAM pose를 망칠 수 있어서 invalid 처리한다.
    """

    # select_imu_window() 단계에서 이미 window 자체가 불완전하다고 판정된 경우.
    if window_status != "ok":
        return 0.0, 0, window_status

    # nan/inf는 CSV에 한 번 들어가면 torch/tensor 연산까지 퍼질 수 있으므로 차단한다.
    if has_nan:
        return 0.0, 0, "nan"

    # 보간된 시작/끝 샘플 포함 최소 2개, 실제 적분 step 최소 1개가 필요하다.
    if imu_count < 2 or used_steps < 1:
        return 0.0, 0, "too_few_imu"

    # 프레임 timestamp가 뒤집혔거나 같은 시각인 경우.
    if dt <= 0.0:
        return 0.0, 0, "bad_dt"

    # 모바일/웹 수집에서는 큰 timestamp jump가 생길 수 있다. 너무 긴 구간은 prior로 위험하다.
    if dt > 0.5:
        return 0.0, 0, "dt_too_large"

    # 한 프레임 간 45도 이상 회전은 일반 FPS 영상에서는 이상치일 가능성이 높다.
    if dr_norm_deg > 45.0:
        return 0.0, 0, "too_large_rotation"

    # 작은 회전은 유효하지만 motion forcing에는 큰 의미가 없으므로 reason만 남긴다.
    if dr_norm_deg < 0.05:
        return 0.0, 1, "small_rotation"

    # 큰 회전이지만 완전히 버리지는 않는다. 단, 후단에서 쓸 때는 낮은 weight로 다룬다.
    if dr_norm_deg > 20.0:
        return 0.0003, 1, "large_but_accepted"

    return 0.001, 1, "ok"


def load_frames(frames_csv: Path) -> List[Dict[str, Any]]:
    """
    `frames.csv`를 읽어서 timestamp 순서로 정렬한다.

    서버가 저장하는 frame_id는 수신 순서를 의미한다. 일반적으로 timestamp 순서와 같지만,
    네트워크/브라우저 환경에서는 아주 드물게 순서가 틀어질 수 있으므로, preintegration은
    timestamp 기준으로 다시 정렬한다.

    정렬 후 `frame_index`를 다시 부여한다. `demo.py`는 이미지 스트림의 index와
    `imu_prior.csv`의 frame_index를 맞춰서 prior를 찾는다.
    """

    frames: List[Dict[str, Any]] = []

    with open(frames_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame_index = len(frames)
            frame_id = int(float(row.get("frame_id", frame_index)))
            timestamp_ns = to_int_ns(row)
            timestamp_sec = to_float(row.get("timestamp_sec"), timestamp_ns / 1e9)

            # filename이 없는 오래된 CSV도 처리할 수 있게 기본 파일명을 만든다.
            frames.append({
                "frame_id": frame_id,
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "timestamp_ns": timestamp_ns,
                "filename": row.get("filename", f"{frame_id:06d}.jpg"),
            })

    frames.sort(key=lambda x: x["timestamp_ns"])

    # timestamp 정렬 이후의 순번을 frame_index로 사용한다.
    for frame_index, frame in enumerate(frames):
        frame["frame_index"] = frame_index

    return frames


def load_imu(
    imu_csv: Path,
    calibration: Optional[ImuCalibration] = None,
) -> List[Dict[str, Any]]:
    """
    `imu.csv`를 읽어서 gyro/accel 샘플을 timestamp 순서로 정렬한다.

    입력 단위 가정:
        gyro:
            rad/s. 웹 프론트에서는 DeviceMotionEvent.rotationRate를 deg/s에서 rad/s로
            변환해서 서버에 보낸다.

        accel:
            m/s^2. 가능하면 linear acceleration을 쓰고, 없으면 gravity 포함 accel이
            들어올 수 있다. gravity 포함 값이면 `dv/dp`에는 중력 성분이 섞인다.

    반환 전 같은 timestamp의 중복 샘플은 평균으로 병합한다. 같은 ns 안에 여러 row가
    있으면 dt=0 step이 생겨 적분에는 도움이 되지 않고 품질 판단만 헷갈릴 수 있기 때문이다.
    """

    calibration = calibration or ImuCalibration()
    samples: List[Dict[str, Any]] = []

    with open(imu_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # 현재 컬럼명과 과거/외부 로그 컬럼명을 모두 허용한다.
            gx = get_float_any(row, ("gx", "gyro_x", "wx", "w_x", "w_RS_S_x")) - calibration.gyro_bias[0]
            gy = get_float_any(row, ("gy", "gyro_y", "wy", "w_y", "w_RS_S_y")) - calibration.gyro_bias[1]
            gz = get_float_any(row, ("gz", "gyro_z", "wz", "w_z", "w_RS_S_z")) - calibration.gyro_bias[2]

            ax = get_float_any(row, ("ax", "acc_x", "accel_x", "linear_accel_x", "a_RS_S_x")) - calibration.acc_bias[0]
            ay = get_float_any(row, ("ay", "acc_y", "accel_y", "linear_accel_y", "a_RS_S_y")) - calibration.acc_bias[1]
            az = get_float_any(row, ("az", "acc_z", "accel_z", "linear_accel_z", "a_RS_S_z")) - calibration.acc_bias[2]

            gx, gy, gz = convert_gyro_device_to_camera(gx, gy, gz)
            ax, ay, az = convert_accel_device_to_camera(ax, ay, az)

            gx, gy, gz = mat3_vec(calibration.R_imu_to_camera, (gx, gy, gz))
            ax, ay, az = mat3_vec(calibration.R_imu_to_camera, (ax, ay, az))

            timestamp_ns = to_int_ns(row)
            timestamp_sec = to_float(row.get("timestamp_sec"), timestamp_ns / 1e9)

            samples.append({
                "timestamp_sec": timestamp_sec,
                "timestamp_ns": timestamp_ns,
                "gx": gx,
                "gy": gy,
                "gz": gz,
                "ax": ax,
                "ay": ay,
                "az": az,
            })

    samples.sort(key=lambda x: x["timestamp_ns"])
    return merge_duplicate_timestamps(samples)


def merge_duplicate_timestamps(
    samples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    같은 timestamp_ns를 가진 IMU 샘플들을 하나로 합친다.

    여러 센서 이벤트가 같은 ns로 기록되면 그 사이 dt가 0이므로 적분 step으로 사용할 수
    없다. 버리면 정보가 사라지니 같은 timestamp 안의 값을 평균내서 하나의 대표 샘플로
    만든다.
    """

    if not samples:
        return []

    merged: List[Dict[str, Any]] = []
    current_ts = samples[0]["timestamp_ns"]
    bucket = []

    def flush_bucket() -> None:
        """
        현재 timestamp bucket을 평균 샘플 하나로 확정한다.

        nested function인 이유는 current_ts와 bucket을 바로 공유하면서, 루프 중 timestamp가
        바뀔 때와 마지막 종료 시점에 같은 평균 처리 코드를 재사용하기 위해서다.
        """

        if not bucket:
            return

        count = len(bucket)
        merged.append({
            "timestamp_sec": current_ts / 1e9,
            "timestamp_ns": current_ts,
            "gx": sum(float(s["gx"]) for s in bucket) / count,
            "gy": sum(float(s["gy"]) for s in bucket) / count,
            "gz": sum(float(s["gz"]) for s in bucket) / count,
            "ax": sum(float(s["ax"]) for s in bucket) / count,
            "ay": sum(float(s["ay"]) for s in bucket) / count,
            "az": sum(float(s["az"]) for s in bucket) / count,
        })

    for sample in samples:
        sample_ts = sample["timestamp_ns"]
        if sample_ts != current_ts:
            flush_bucket()
            bucket = []
            current_ts = sample_ts

        bucket.append(sample)

    flush_bucket()
    return merged


def interpolate_imu_sample(
    imu: List[Dict[str, Any]],
    timestamps: List[int],
    target_ns: int,
) -> Optional[Dict[str, Any]]:
    """
    원하는 timestamp의 IMU 값을 선형 보간한다.

    프레임 timestamp가 실제 IMU 샘플 timestamp와 정확히 일치하는 경우는 드물다. 그래서
    프레임 경계 시각(start/end)에 해당하는 가상의 IMU 샘플을 양쪽 실제 샘플로 보간해서
    만든다. 이 보간이 있어야 `[prev_frame_time, curr_frame_time]` 구간 전체를 빠짐없이
    적분할 수 있다.

    target_ns가 IMU 로그 범위 밖이면 None을 반환한다. 이 경우 해당 프레임 prior는
    `outside_imu_range`로 invalid 처리된다.
    """

    # timestamps는 정렬된 int ns 리스트다. bisect로 O(log N)에 target 위치를 찾는다.
    index = bisect.bisect_left(timestamps, target_ns)

    # 정확히 같은 timestamp 샘플이 있으면 보간 없이 그대로 복사한다.
    if index < len(imu) and timestamps[index] == target_ns:
        sample = dict(imu[index])
        sample["timestamp_ns"] = target_ns
        sample["timestamp_sec"] = target_ns / 1e9
        return sample

    # target이 첫 샘플보다 이전이거나 마지막 샘플보다 이후면 보간할 양쪽 이웃이 없다.
    if index == 0 or index >= len(imu):
        return None

    left = imu[index - 1]
    right = imu[index]
    left_ns = int(left["timestamp_ns"])
    right_ns = int(right["timestamp_ns"])

    if right_ns <= left_ns:
        return None

    # alpha=0이면 left, alpha=1이면 right. 각 센서 축을 독립적으로 선형 보간한다.
    alpha = (target_ns - left_ns) / float(right_ns - left_ns)

    def lerp(name: str) -> float:
        return (1.0 - alpha) * float(left[name]) + alpha * float(right[name])

    return {
        "timestamp_sec": target_ns / 1e9,
        "timestamp_ns": target_ns,
        "gx": lerp("gx"),
        "gy": lerp("gy"),
        "gz": lerp("gz"),
        "ax": lerp("ax"),
        "ay": lerp("ay"),
        "az": lerp("az"),
    }


def integrate_imu_window(
    imu_window: List[Dict[str, Any]],
    start_ns: Optional[int],
    end_ns: int,
    window_status: str = "ok",
) -> Dict[str, Any]:
    """
    하나의 프레임 구간에 해당하는 IMU window를 사전적분한다.

    입력:
        imu_window:
            select_imu_window()가 만든 IMU 샘플 목록. 정상 구간이면 첫 샘플은 이전 프레임
            timestamp에 보간된 값이고, 마지막 샘플은 현재 프레임 timestamp에 보간된 값이다.

        start_ns, end_ns:
            이전 프레임과 현재 프레임 timestamp [ns].

        window_status:
            IMU window 선택 단계의 상태. "ok"가 아니면 적분하지 않고 invalid row용
            zero prior를 반환한다.

    적분 방식:
        1. 각 인접 IMU 샘플 사이 dt_sample을 구한다.
        2. gyro/accel은 구간 양 끝 값을 평균내 midpoint 값으로 사용한다.
        3. gyro midpoint로 작은 quaternion step을 만들고 `dq = dq * step`으로 누적한다.
        4. accel은 현재 누적 회전의 중간값(q_mid)으로 회전시킨 뒤 dv/dp를 적분한다.

    한계:
        이 함수는 bias와 gravity를 추정하지 않는다. 따라서 `dv/dp`는 "회전 보정된 raw
        acceleration 적분값"에 가깝고, 물리적으로 완전한 world-frame translation prior는
        아니다.
    """

    # 첫 프레임에는 이전 프레임이 없어서 delta를 정의할 수 없다.
    # 또한 window_status가 ok가 아니거나 샘플 수가 부족하면 후단이 사용할 수 없도록
    # zero prior와 reason만 남긴다.
    if start_ns is None or len(imu_window) < 2 or window_status != "ok":
        dt = 0.0 if start_ns is None else (end_ns - start_ns) / 1e9
        dq = rotvec_to_quat_wxyz(0.0, 0.0, 0.0)

        return {
            "dt": dt,
            "imu_count": len(imu_window),
            "imu_used_steps": 0,
            "dr_x": 0.0,
            "dr_y": 0.0,
            "dr_z": 0.0,
            "dr_norm": 0.0,
            "dr_norm_deg": 0.0,
            "dq_w": dq[0],
            "dq_x": dq[1],
            "dq_y": dq[2],
            "dq_z": dq[3],
            "dv_x": 0.0,
            "dv_y": 0.0,
            "dv_z": 0.0,
            "dp_x": 0.0,
            "dp_y": 0.0,
            "dp_z": 0.0,
            "gyro_norm_mean": 0.0,
            "acc_norm_mean": 0.0,
            "has_nan": False,
            "window_status": window_status,
        }

    # 누적 delta rotation. 프레임 시작 시점에서 현재 step까지의 상대 회전이다.
    # wxyz 순서이며 초기값은 identity rotation이다.
    dq = (1.0, 0.0, 0.0, 0.0)

    # delta velocity. 현재는 bias/gravity 제거 전 accel을 적분한 값이다.
    dv_x = 0.0
    dv_y = 0.0
    dv_z = 0.0

    # delta position. 아래의 vx/vy/vz를 이용해 한 번 더 적분한다.
    dp_x = 0.0
    dp_y = 0.0
    dp_z = 0.0

    # 프레임 구간 시작을 기준으로 한 누적 속도. dp 적분을 위해 내부에서만 사용한다.
    vx = 0.0
    vy = 0.0
    vz = 0.0

    # 품질/로그용 평균 norm 계산에 사용한다.
    gyro_norm_sum = 0.0
    acc_norm_sum = 0.0
    used_steps = 0

    # 인접한 두 IMU 샘플 사이를 하나의 작은 적분 step으로 본다.
    for prev, curr in zip(imu_window[:-1], imu_window[1:]):
        dt_sample = (curr["timestamp_ns"] - prev["timestamp_ns"]) / 1e9

        # dt가 0/음수면 timestamp가 깨진 것이고, 0.2초 이상이면 센서 로그가 끊긴 것으로 본다.
        # 큰 gap을 억지로 적분하면 gyro/accel 평균 하나로 긴 구간을 대표하게 되어 prior가 위험해진다.
        if dt_sample <= 0.0 or dt_sample > 0.2:
            continue

        # midpoint integration. 양 끝 샘플을 평균내 현재 작은 구간의 대표 센서값으로 사용한다.
        gx = 0.5 * (prev["gx"] + curr["gx"])
        gy = 0.5 * (prev["gy"] + curr["gy"])
        gz = 0.5 * (prev["gz"] + curr["gz"])

        ax = 0.5 * (prev["ax"] + curr["ax"])
        ay = 0.5 * (prev["ay"] + curr["ay"])
        az = 0.5 * (prev["az"] + curr["az"])

        # accel을 어느 방향으로 적분할지 정하려면 step 중간 시점의 회전이 필요하다.
        # 현재 누적 회전 dq에 half gyro step을 곱해서 q_mid를 만든다.
        half_step = rotvec_to_quat_wxyz(
            0.5 * gx * dt_sample,
            0.5 * gy * dt_sample,
            0.5 * gz * dt_sample,
        )
        q_mid = quat_multiply_wxyz(dq, half_step)

        # 센서/body frame accel을 q_mid 기준으로 회전시킨다.
        # gravity 제거는 여기서 하지 않으므로, 입력 accel이 gravity 포함이면 결과에도 섞인다.
        acc_x, acc_y, acc_z = quat_rotate_vector_wxyz(q_mid, (ax, ay, az))

        # 등가속도 가정으로 position을 먼저 적분한다.
        # p_{k+1} = p_k + v_k * dt + 0.5 * a * dt^2
        dp_x += vx * dt_sample + 0.5 * acc_x * dt_sample * dt_sample
        dp_y += vy * dt_sample + 0.5 * acc_y * dt_sample * dt_sample
        dp_z += vz * dt_sample + 0.5 * acc_z * dt_sample * dt_sample

        # v_{k+1} = v_k + a * dt
        vx += acc_x * dt_sample
        vy += acc_y * dt_sample
        vz += acc_z * dt_sample

        # delta velocity도 CSV에 바로 저장하기 위해 별도로 누적한다.
        dv_x += acc_x * dt_sample
        dv_y += acc_y * dt_sample
        dv_z += acc_z * dt_sample

        # gyro midpoint로 현재 step의 delta quaternion을 만들고, 누적 회전에 곱한다.
        # 이 방식이 단순 `dr += gyro * dt`보다 회전축이 변하는 상황에서 더 안정적이다.
        step = rotvec_to_quat_wxyz(
            gx * dt_sample,
            gy * dt_sample,
            gz * dt_sample,
        )
        dq = quat_multiply_wxyz(dq, step)

        gyro_norm_sum += math.sqrt(gx * gx + gy * gy + gz * gz)
        acc_norm_sum += math.sqrt(acc_x * acc_x + acc_y * acc_y + acc_z * acc_z)
        used_steps += 1

    # 누적 quaternion은 후단에서 쓰기 편하도록 rotation vector도 같이 저장한다.
    dr_x, dr_y, dr_z = quat_to_rotvec_wxyz(dq)
    dr_norm = math.sqrt(dr_x * dr_x + dr_y * dr_y + dr_z * dr_z)
    dr_norm_deg = math.degrees(dr_norm)

    # CSV로 내보내기 전에 비정상 숫자가 있는지 한 번 더 확인한다.
    values = [
        dr_x,
        dr_y,
        dr_z,
        dr_norm,
        dr_norm_deg,
        *dq,
        dv_x,
        dv_y,
        dv_z,
        dp_x,
        dp_y,
        dp_z,
    ]
    has_nan = any(math.isnan(v) or math.isinf(v) for v in values)

    # 반환 dict의 key는 build_imu_prior()에서 그대로 CSV 컬럼으로 변환된다.
    return {
        "dt": (end_ns - start_ns) / 1e9,
        "imu_count": len(imu_window),
        "imu_used_steps": used_steps,
        "dr_x": dr_x,
        "dr_y": dr_y,
        "dr_z": dr_z,
        "dr_norm": dr_norm,
        "dr_norm_deg": dr_norm_deg,
        "dq_w": dq[0],
        "dq_x": dq[1],
        "dq_y": dq[2],
        "dq_z": dq[3],
        "dv_x": dv_x,
        "dv_y": dv_y,
        "dv_z": dv_z,
        "dp_x": dp_x,
        "dp_y": dp_y,
        "dp_z": dp_z,
        "gyro_norm_mean": gyro_norm_sum / max(used_steps, 1),
        "acc_norm_mean": acc_norm_sum / max(used_steps, 1),
        "has_nan": has_nan,
        "window_status": window_status,
    }


def select_imu_window(
    imu: List[Dict[str, Any]],
    timestamps: List[int],
    start_ns: Optional[int],
    end_ns: int,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    한 프레임 구간 `[start_ns, end_ns]`에 들어갈 IMU 샘플 window를 만든다.

    중요한 점:
        단순히 `start <= sample_time <= end`인 실제 샘플만 고르면, start와 첫 IMU 샘플
        사이, 마지막 IMU 샘플과 end 사이의 시간이 빠질 수 있다. 그래서 이 함수는
        start/end 시각에 보간 샘플을 추가해서 정확히 프레임 경계부터 경계까지 적분하게 한다.

    반환:
        (window, "ok"):
            정상 window. window[0]은 start_ns 보간 샘플, window[-1]은 end_ns 보간 샘플.

        ([], reason):
            적분할 수 없는 상태. reason은 imu_prior.csv의 imu_reason으로 저장된다.
    """

    if start_ns is None:
        return [], "first_frame"

    if end_ns <= start_ns:
        return [], "bad_dt"

    # 프레임 경계 시각의 센서값을 보간해서 만든다.
    start_sample = interpolate_imu_sample(imu, timestamps, start_ns)
    end_sample = interpolate_imu_sample(imu, timestamps, end_ns)

    # 프레임 경계가 IMU 로그 범위 밖이면 보간 자체가 불가능하다.
    if start_sample is None or end_sample is None:
        return [], "outside_imu_range"

    # 실제 IMU 샘플 중 start/end 사이에 있는 샘플만 가져온다.
    # start/end는 이미 보간 샘플로 직접 넣으므로 내부 실제 샘플은 열린 구간으로 고른다.
    left = bisect.bisect_right(timestamps, start_ns)
    right = bisect.bisect_left(timestamps, end_ns)
    window = [start_sample] + imu[left:right] + [end_sample]

    return window, "ok"


def build_imu_prior(
    frames_csv: Path,
    imu_csv: Path,
    output_csv: Path,
    calibration: Optional[ImuCalibration] = None,
) -> Dict[str, Any]:
    """
    frames.csv와 imu.csv를 읽어서 imu_prior.csv를 생성한다.

    전체 흐름:
        1. frame/IMU 로그를 timestamp 기준으로 정렬한다.
        2. 각 frame i에 대해 frame i-1 시각부터 frame i 시각까지의 IMU window를 만든다.
        3. integrate_imu_window()로 `dr/dq/dv/dp`를 계산한다.
        4. compute_imu_weight()로 valid/weight/reason을 붙인다.
        5. frame별 row를 CSV로 저장한다.

    이 함수가 서버에서 실제로 호출되는 entry point이다. `/home/ubuntu/SLAM/main.py`의
    run_imu_preintegration()이 이 스크립트를 subprocess로 실행한다.
    """

    calibration = calibration or ImuCalibration()

    frames = load_frames(frames_csv)
    imu = load_imu(imu_csv, calibration=calibration)

    # 보간/범위 검색을 빠르게 하기 위한 timestamp 전용 리스트.
    imu_timestamps = [int(sample["timestamp_ns"]) for sample in imu]

    if len(frames) == 0:
        raise ValueError(f"frames.csv has no frame rows: {frames_csv}")

    if len(imu) == 0:
        raise ValueError(f"imu.csv has no imu rows: {imu_csv}")

    rows = []

    # frame 0은 이전 프레임이 없으므로 identity prior가 되고, frame 1부터 실제 delta가 생긴다.
    for i, frame in enumerate(frames):
        curr_ns = frame["timestamp_ns"]
        curr_sec = frame["timestamp_sec"]

        prev_ns = None if i == 0 else frames[i - 1]["timestamp_ns"]
        prev_sec = 0.0 if i == 0 else frames[i - 1]["timestamp_sec"]

        # 현재 frame에 대응되는 IMU 구간을 정확히 잘라낸다.
        imu_window, window_status = select_imu_window(
            imu,
            imu_timestamps,
            prev_ns,
            curr_ns,
        )
        integ = integrate_imu_window(
            imu_window,
            prev_ns,
            curr_ns,
            window_status=window_status,
        )

        # 후단에서 잘못된 prior를 쓰지 않도록 품질 정보를 같이 계산한다.
        weight, valid, reason = compute_imu_weight(
            dt=float(integ["dt"]),
            imu_count=int(integ["imu_count"]),
            used_steps=int(integ["imu_used_steps"]),
            dr_norm_deg=float(integ["dr_norm_deg"]),
            has_nan=bool(integ["has_nan"]),
            window_status=str(integ["window_status"]),
        )
        uncertainty = estimate_preintegration_uncertainty(
            float(integ["dt"]),
            calibration,
        )
        R = calibration.R_imu_to_camera

        # CSV row. demo.py는 frame_index로 이 row를 찾아 MotionFilter에 넘긴다.
        # dr/dq는 rotation prior, dv/dp는 실험적 translation prior의 입력으로 사용할 수 있다.
        rows.append({
            "frame_id": frame["frame_id"],
            "frame_index": frame["frame_index"],
            "filename": frame["filename"],
            "timestamp_sec": f"{curr_sec:.9f}",
            "timestamp_ns": curr_ns,
            "prev_timestamp_sec": f"{prev_sec:.9f}",
            "prev_timestamp_ns": "" if prev_ns is None else prev_ns,
            "dt": f"{float(integ['dt']):.9f}",
            "imu_count": int(integ["imu_count"]),
            "imu_used_steps": int(integ["imu_used_steps"]),
            "dr_x": f"{float(integ['dr_x']):.12f}",
            "dr_y": f"{float(integ['dr_y']):.12f}",
            "dr_z": f"{float(integ['dr_z']):.12f}",
            "dr_norm": f"{float(integ['dr_norm']):.12f}",
            "dr_norm_deg": f"{float(integ['dr_norm_deg']):.9f}",
            "dq_w": f"{float(integ['dq_w']):.12f}",
            "dq_x": f"{float(integ['dq_x']):.12f}",
            "dq_y": f"{float(integ['dq_y']):.12f}",
            "dq_z": f"{float(integ['dq_z']):.12f}",
            "dv_x": f"{float(integ['dv_x']):.12f}",
            "dv_y": f"{float(integ['dv_y']):.12f}",
            "dv_z": f"{float(integ['dv_z']):.12f}",
            "dp_x": f"{float(integ['dp_x']):.12f}",
            "dp_y": f"{float(integ['dp_y']):.12f}",
            "dp_z": f"{float(integ['dp_z']):.12f}",
            "gyro_norm_mean": f"{float(integ['gyro_norm_mean']):.12f}",
            "acc_norm_mean": f"{float(integ['acc_norm_mean']):.12f}",
            "imu_weight": f"{weight:.9f}",
            "imu_valid": valid,
            "imu_reason": reason,
            "imu_extrinsic_applied": 1 if calibration.extrinsic_applied else 0,
            "imu_calibration_source": calibration.source,
            "gyro_bias_x": f"{calibration.gyro_bias[0]:.12f}",
            "gyro_bias_y": f"{calibration.gyro_bias[1]:.12f}",
            "gyro_bias_z": f"{calibration.gyro_bias[2]:.12f}",
            "acc_bias_x": f"{calibration.acc_bias[0]:.12f}",
            "acc_bias_y": f"{calibration.acc_bias[1]:.12f}",
            "acc_bias_z": f"{calibration.acc_bias[2]:.12f}",
            "gyro_noise_density": f"{calibration.gyro_noise_density:.12e}",
            "gyro_random_walk": f"{calibration.gyro_random_walk:.12e}",
            "accel_noise_density": f"{calibration.accel_noise_density:.12e}",
            "accel_random_walk": f"{calibration.accel_random_walk:.12e}",
            "rot_var": f"{uncertainty['rot_var']:.12e}",
            "vel_var": f"{uncertainty['vel_var']:.12e}",
            "pos_var": f"{uncertainty['pos_var']:.12e}",
            "rot_info": f"{uncertainty['rot_info']:.12e}",
            "vel_info": f"{uncertainty['vel_info']:.12e}",
            "pos_info": f"{uncertainty['pos_info']:.12e}",
            "R_cam_imu_00": f"{R[0][0]:.12f}",
            "R_cam_imu_01": f"{R[0][1]:.12f}",
            "R_cam_imu_02": f"{R[0][2]:.12f}",
            "R_cam_imu_10": f"{R[1][0]:.12f}",
            "R_cam_imu_11": f"{R[1][1]:.12f}",
            "R_cam_imu_12": f"{R[1][2]:.12f}",
            "R_cam_imu_20": f"{R[2][0]:.12f}",
            "R_cam_imu_21": f"{R[2][1]:.12f}",
            "R_cam_imu_22": f"{R[2][2]:.12f}",
            # 이 컬럼은 나중에 여러 적분 구현을 비교할 때 어떤 방식으로 생성된 CSV인지
            # 추적하기 위한 메타데이터다.
            "integration_method": "boundary_interp_midpoint_so3",
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # rows[0]의 key 순서를 그대로 CSV header 순서로 사용한다.
    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 첫 프레임은 원래 delta가 없으므로 통계에서 제외한다.
    valid_rows = [row for row in rows[1:] if int(row["imu_valid"]) == 1]
    zero_imu = sum(1 for row in rows[1:] if int(row["imu_count"]) == 0)
    avg_imu = (
        sum(int(row["imu_count"]) for row in rows[1:]) / max(len(rows) - 1, 1)
    )

    print(f"[OK] imu_prior saved: {output_csv}")
    print(f"[INFO] frames={len(frames)}, imu={len(imu)}, priors={len(rows)}")
    print(f"[INFO] valid_priors={len(valid_rows)}, zero_imu_windows={zero_imu}")
    print(f"[INFO] avg_imu_per_frame={avg_imu:.3f}")
    print(f"[INFO] frame_time={frames[0]['timestamp_sec']:.6f}~{frames[-1]['timestamp_sec']:.6f}")
    print(f"[INFO] imu_time={imu[0]['timestamp_sec']:.6f}~{imu[-1]['timestamp_sec']:.6f}")
    print(
        "[INFO] imu_calibration="
        f"{calibration.source}, extrinsic_applied={int(calibration.extrinsic_applied)}"
    )

    return {
        "frames": len(frames),
        "imu": len(imu),
        "priors": len(rows),
        "valid_priors": len(valid_rows),
        "zero_imu_windows": zero_imu,
        "imu_calibration_source": calibration.source,
        "imu_extrinsic_applied": calibration.extrinsic_applied,
        "output": str(output_csv),
    }


def parse_args() -> argparse.Namespace:
    """
    CLI argument 정의.

    사용 예:
        python tools/imu_preintegrate.py --session_dir /path/to/session

    또는:
        python tools/imu_preintegrate.py \
            --frames /path/to/frames.csv \
            --imu /path/to/imu.csv \
            --output /path/to/imu_prior.csv
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--session_dir", type=str, default=None)
    parser.add_argument("--frames", type=str, default=None)
    parser.add_argument("--imu", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--cam_sensor_yaml",
        type=str,
        default=None,
        help="EuRoC-style camera sensor.yaml containing T_BS",
    )
    parser.add_argument(
        "--imu_sensor_yaml",
        type=str,
        default=None,
        help="EuRoC-style IMU sensor.yaml containing T_BS and noise values",
    )
    parser.add_argument(
        "--imu_to_cam_rotation",
        type=float,
        nargs=9,
        default=None,
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
        help="optional row-major 3x3 rotation from IMU frame to camera frame",
    )
    parser.add_argument(
        "--gyro_bias",
        type=float,
        nargs=3,
        default=None,
        metavar=("BGX", "BGY", "BGZ"),
        help="constant gyro bias to subtract before integration",
    )
    parser.add_argument(
        "--acc_bias",
        type=float,
        nargs=3,
        default=None,
        metavar=("BAX", "BAY", "BAZ"),
        help="constant accelerometer bias to subtract before integration",
    )
    return parser.parse_args()


def main() -> None:
    """
    CLI entry point.

    서버에서는 session_dir, frames, imu, output을 모두 넘겨 실행한다. 사람이 직접 실행할 때는
    `--session_dir`만 줘도 해당 폴더의 `frames.csv`, `imu.csv`를 읽고 `imu_prior.csv`를 쓴다.
    """

    args = parse_args()

    session_dir_arg = args.session_dir or args.session
    if session_dir_arg is None and (args.frames is None or args.imu is None):
        raise ValueError("Provide --session_dir/--session or both --frames and --imu")

    session_dir = Path(session_dir_arg).resolve() if session_dir_arg else None

    # 명시 경로가 있으면 그것을 우선하고, 없으면 session_dir 안의 표준 파일명을 사용한다.
    frames_csv = Path(args.frames).resolve() if args.frames else session_dir / "frames.csv"
    imu_csv = Path(args.imu).resolve() if args.imu else session_dir / "imu.csv"
    output_csv = Path(args.output).resolve() if args.output else session_dir / "imu_prior.csv"

    if not frames_csv.exists():
        raise FileNotFoundError(f"frames.csv not found: {frames_csv}")

    if not imu_csv.exists():
        raise FileNotFoundError(f"imu.csv not found: {imu_csv}")

    cam_sensor_yaml = Path(args.cam_sensor_yaml).resolve() if args.cam_sensor_yaml else None
    imu_sensor_yaml = Path(args.imu_sensor_yaml).resolve() if args.imu_sensor_yaml else None

    calibration = build_imu_calibration(
        cam_sensor_yaml=cam_sensor_yaml,
        imu_sensor_yaml=imu_sensor_yaml,
        imu_to_cam_rotation=args.imu_to_cam_rotation,
        gyro_bias=args.gyro_bias,
        acc_bias=args.acc_bias,
    )

    build_imu_prior(frames_csv, imu_csv, output_csv, calibration=calibration)


if __name__ == "__main__":
    main()

import csv
import math
import torch


IMU_BIAS_JACOBIAN_ORDER = ("J_R_bg", "J_v_ba", "J_v_bg", "J_p_ba", "J_p_bg")


def _safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _quat_normalize_xyzw(q):
    x, y, z, w = [_safe_float(v) for v in q]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / norm, y / norm, z / norm, w / norm]


def _quat_multiply_xyzw(q1, q2):
    x1, y1, z1, w1 = _quat_normalize_xyzw(q1)
    x2, y2, z2, w2 = _quat_normalize_xyzw(q2)

    return _quat_normalize_xyzw([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def _quat_rotate_xyzw(q, v):
    x, y, z, w = _quat_normalize_xyzw(q)
    vx, vy, vz = [_safe_float(x) for x in v]

    # q * [v, 0] * q^-1, expanded.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    return [
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    ]


def _rotvec_to_quat_xyzw(rx, ry, rz):
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]

    half = 0.5 * theta
    scale = math.sin(half) / theta
    return _quat_normalize_xyzw([
        rx * scale,
        ry * scale,
        rz * scale,
        math.cos(half),
    ])


def _quat_to_rotvec_xyzw(q):
    x, y, z, w = _quat_normalize_xyzw(q)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w

    sin_half = math.sqrt(x * x + y * y + z * z)
    if sin_half < 1e-12:
        return [2.0 * x, 2.0 * y, 2.0 * z]

    angle = 2.0 * math.atan2(sin_half, w)
    scale = angle / sin_half
    return [x * scale, y * scale, z * scale]


def _row_delta_quat_xyzw(row):
    if row is None:
        return [0.0, 0.0, 0.0, 1.0]

    has_dq = all(row.get(k) not in (None, "") for k in ("dq_w", "dq_x", "dq_y", "dq_z"))
    if has_dq:
        return _quat_normalize_xyzw([
            _safe_float(row.get("dq_x")),
            _safe_float(row.get("dq_y")),
            _safe_float(row.get("dq_z")),
            _safe_float(row.get("dq_w"), 1.0),
        ])

    return _rotvec_to_quat_xyzw(
        _safe_float(row.get("dr_x")),
        _safe_float(row.get("dr_y")),
        _safe_float(row.get("dr_z")),
    )


def _row_vec3(row, prefix):
    return [
        _safe_float(row.get(f"{prefix}_x")),
        _safe_float(row.get(f"{prefix}_y")),
        _safe_float(row.get(f"{prefix}_z")),
    ]


def _add3(a, b):
    return [a[i] + b[i] for i in range(3)]


def _scale3(a, s):
    return [a[i] * s for i in range(3)]


def _zero_mat3():
    return [[0.0, 0.0, 0.0] for _ in range(3)]


def _eye_mat3(scale=1.0):
    return [
        [float(scale), 0.0, 0.0],
        [0.0, float(scale), 0.0],
        [0.0, 0.0, float(scale)],
    ]


def _skew3(v):
    x, y, z = [_safe_float(a) for a in v]
    return [
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ]


def _mat3_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(3)] for i in range(3)]


def _mat3_scale(A, s):
    return [[A[i][j] * s for j in range(3)] for i in range(3)]


def _mat3_multiply(A, B):
    return [
        [sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def _quat_to_rotmat_xyzw(q):
    x, y, z, w = _quat_normalize_xyzw(q)
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _row_has_mat3(row, prefix):
    return all(row.get(f"{prefix}_{i}{j}") not in (None, "") for i in range(3) for j in range(3))


def _fallback_bias_jacobians(dt):
    dt = max(_safe_float(dt), 0.0)
    return {
        "J_R_bg": _eye_mat3(-dt),
        "J_v_ba": _eye_mat3(-dt),
        "J_v_bg": _zero_mat3(),
        "J_p_ba": _eye_mat3(-0.5 * dt * dt),
        "J_p_bg": _zero_mat3(),
    }


def _row_mat3(row, prefix, fallback):
    if not _row_has_mat3(row, prefix):
        return fallback

    return [
        [_safe_float(row.get(f"{prefix}_{i}{j}")) for j in range(3)]
        for i in range(3)
    ]


def _matrix_columns(prefix, matrix):
    return {
        f"{prefix}_{i}{j}": matrix[i][j]
        for i in range(3)
        for j in range(3)
    }


def _bias_jacobian_dict(row):
    fallback = _fallback_bias_jacobians(row.get("dt", 0.0) if row is not None else 0.0)
    return {
        prefix: _row_mat3(row or {}, prefix, fallback[prefix])
        for prefix in IMU_BIAS_JACOBIAN_ORDER
    }


def _variance_from_info(row, var_key, info_key):
    var = _safe_float(row.get(var_key), 0.0)
    if var > 0.0:
        return var

    info = _safe_float(row.get(info_key), 0.0)
    if info > 1e-18:
        return 1.0 / info

    return 0.0


def compose_imu_prior_rows(rows, prev_frame_index=None, frame_index=None):
    """
    Compose consecutive preintegrated IMU rows into one interval.

    Each input row describes k-1 -> k in the starting frame of that row.
    For a skipped visual interval A -> C through B, the correct first-order
    composition is:

        dR_AC = dR_AB dR_BC
        dv_AC = dv_AB + dR_AB dv_BC
        dp_AC = dp_AB + dv_AB dt_BC + dR_AB dp_BC

    This avoids the incorrect `dv += row.dv`, `dp += row.dp` shortcut, which
    breaks as soon as rotation accumulates between selected keyframes.

    The stored uncertainty is diagonal rather than a full Forster covariance.
    For rows produced from white-noise densities, we recover the nominal gyro
    and accelerometer variance rates and rebuild the total-duration diagonal
    approximation. In particular, position variance grows with `dt^3`; simply
    summing each short-row `pos_var` would incorrectly make it grow only
    linearly when MotionFilter skips camera frames.
    """

    rows = [row for row in rows if row is not None]
    if len(rows) == 0:
        return None

    q_total = [0.0, 0.0, 0.0, 1.0]
    dv_total = [0.0, 0.0, 0.0]
    dp_total = [0.0, 0.0, 0.0]
    dt_total = 0.0

    imu_count = 0
    imu_used_steps = 0
    valid = True
    imu_weight = 1.0
    reasons = []
    rot_var = 0.0
    vel_var = 0.0
    pos_var = 0.0
    gyro_variance_rate_sum = 0.0
    gyro_variance_rate_count = 0
    accel_variance_rate_sum = 0.0
    accel_variance_rate_count = 0
    J_R_bg_total = _zero_mat3()
    J_v_ba_total = _zero_mat3()
    J_v_bg_total = _zero_mat3()
    J_p_ba_total = _zero_mat3()
    J_p_bg_total = _zero_mat3()

    for row in rows:
        q_before = q_total
        R_before = _quat_to_rotmat_xyzw(q_before)
        dt = _safe_float(row.get("dt"), 0.0)
        dv = _row_vec3(row, "dv")
        dp = _row_vec3(row, "dp")
        jac = _bias_jacobian_dict(row)

        R_dp_skew = _mat3_multiply(R_before, _skew3(dp))
        R_dv_skew = _mat3_multiply(R_before, _skew3(dv))

        J_p_ba_total = _mat3_add(
            _mat3_add(J_p_ba_total, _mat3_scale(J_v_ba_total, dt)),
            _mat3_multiply(R_before, jac["J_p_ba"]),
        )
        J_p_bg_total = _mat3_add(
            _mat3_add(
                _mat3_add(J_p_bg_total, _mat3_scale(J_v_bg_total, dt)),
                _mat3_multiply(R_before, jac["J_p_bg"]),
            ),
            _mat3_scale(_mat3_multiply(R_dp_skew, J_R_bg_total), -1.0),
        )
        J_v_ba_total = _mat3_add(
            J_v_ba_total,
            _mat3_multiply(R_before, jac["J_v_ba"]),
        )
        J_v_bg_total = _mat3_add(
            _mat3_add(J_v_bg_total, _mat3_multiply(R_before, jac["J_v_bg"])),
            _mat3_scale(_mat3_multiply(R_dv_skew, J_R_bg_total), -1.0),
        )
        J_R_bg_total = _mat3_add(
            J_R_bg_total,
            _mat3_multiply(R_before, jac["J_R_bg"]),
        )

        dp_total = _add3(
            dp_total,
            _add3(_scale3(dv_total, dt), _quat_rotate_xyzw(q_before, dp)),
        )
        dv_total = _add3(dv_total, _quat_rotate_xyzw(q_before, dv))
        q_total = _quat_multiply_xyzw(q_total, _row_delta_quat_xyzw(row))
        dt_total += dt

        imu_count += _safe_int(row.get("imu_count"), 0)
        imu_used_steps += _safe_int(row.get("imu_used_steps", row.get("imu_count")), 0)
        valid = valid and (_safe_int(row.get("imu_valid"), 1) != 0)
        imu_weight = min(imu_weight, _safe_float(row.get("imu_weight"), 1.0))
        reasons.append(str(row.get("imu_reason", "ok")))

        row_rot_var = _variance_from_info(row, "rot_var", "rot_info")
        row_vel_var = _variance_from_info(row, "vel_var", "vel_info")
        row_pos_var = _variance_from_info(row, "pos_var", "pos_info")
        rot_var += row_rot_var
        vel_var += row_vel_var
        pos_var += row_pos_var

        if dt > 1e-12 and row_rot_var > 0.0:
            gyro_variance_rate_sum += row_rot_var / dt
            gyro_variance_rate_count += 1
        if dt > 1e-12 and row_vel_var > 0.0:
            accel_variance_rate_sum += row_vel_var / dt
            accel_variance_rate_count += 1

    # `estimate_preintegration_uncertainty()` uses the same first-order white
    # noise model for raw rows. Rebuilding it over the composed duration is a
    # substantially better diagonal approximation than adding short-interval
    # position variances without their velocity-position cross terms.
    if dt_total > 1e-12 and gyro_variance_rate_count > 0:
        gyro_variance_rate = gyro_variance_rate_sum / gyro_variance_rate_count
        rot_var = gyro_variance_rate * dt_total
    if dt_total > 1e-12 and accel_variance_rate_count > 0:
        accel_variance_rate = accel_variance_rate_sum / accel_variance_rate_count
        vel_var = accel_variance_rate * dt_total
        pos_var = accel_variance_rate * dt_total * dt_total * dt_total / 3.0

    dr = _quat_to_rotvec_xyzw(q_total)
    dr_norm = math.sqrt(sum(x * x for x in dr))
    out = dict(rows[-1])

    if prev_frame_index is None:
        prev_frame_index = rows[0].get("prev_frame_index")
    if frame_index is None:
        frame_index = rows[-1].get("frame_index", rows[-1].get("frame_id"))

    out.update({
        "prev_frame_index": prev_frame_index,
        "frame_index": frame_index,
        "dt": dt_total,
        "imu_count": imu_count,
        "imu_used_steps": imu_used_steps,
        "imu_valid": 1 if valid else 0,
        "imu_weight": imu_weight,
        "imu_reason": "ok" if all(r == "ok" for r in reasons) else "composed_with_invalid_reason",
        "dr_x": dr[0],
        "dr_y": dr[1],
        "dr_z": dr[2],
        "dr_norm": dr_norm,
        "dr_norm_deg": dr_norm * 180.0 / math.pi,
        "dq_w": q_total[3],
        "dq_x": q_total[0],
        "dq_y": q_total[1],
        "dq_z": q_total[2],
        "dv_x": dv_total[0],
        "dv_y": dv_total[1],
        "dv_z": dv_total[2],
        "dp_x": dp_total[0],
        "dp_y": dp_total[1],
        "dp_z": dp_total[2],
        "rot_var": rot_var,
        "vel_var": vel_var,
        "pos_var": pos_var,
        "rot_info": 1.0 / rot_var if rot_var > 1e-18 else 0.0,
        "vel_info": 1.0 / vel_var if vel_var > 1e-18 else 0.0,
        "pos_info": 1.0 / pos_var if pos_var > 1e-18 else 0.0,
    })
    for prefix, matrix in {
        "J_R_bg": J_R_bg_total,
        "J_v_ba": J_v_ba_total,
        "J_v_bg": J_v_bg_total,
        "J_p_ba": J_p_ba_total,
        "J_p_bg": J_p_bg_total,
    }.items():
        out.update(_matrix_columns(prefix, matrix))

    return out


def load_imu_prior_csv(path):
    """
    imu_prior.csv를 frame_index 기준 dictionary로 로드한다.

    expected columns:
        frame_index/frame_id, timestamp_sec/timestamp_ns, dt,
        dr_x, dr_y, dr_z,
        dq_w, dq_x, dq_y, dq_z,
        dv_x, dv_y, dv_z,
        dp_x, dp_y, dp_z,
        imu_count, imu_used_steps, imu_valid, imu_weight,
        optional rot/vel/pos variance or information and calibration metadata
    """

    imu_priors = {}

    if path is None:
        return None

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            def get_float(key, default=0.0):
                value = row.get(key, default)
                if value is None or value == "":
                    return default
                try:
                    return float(value)
                except Exception:
                    return default

            def get_int(key, default=0):
                try:
                    return int(float(row.get(key, default)))
                except Exception:
                    return default

            if row.get("frame_index") not in (None, ""):
                frame_index = get_int("frame_index", -1)
            elif row.get("frame_id") not in (None, ""):
                frame_index = get_int("frame_id", -1)
            else:
                frame_index = len(imu_priors)

            if frame_index < 0:
                continue

            frame_id = get_int("frame_id", frame_index)

            parsed = {
                "frame_id": frame_id,
                "frame_index": frame_index,
                "timestamp_sec": get_float("timestamp_sec", 0.0),
                "timestamp_ns": get_int("timestamp_ns", 0),
                "prev_timestamp_sec": get_float("prev_timestamp_sec", 0.0),
                "dt": get_float("dt", 0.0),

                "dr_x": get_float("dr_x", 0.0),
                "dr_y": get_float("dr_y", 0.0),
                "dr_z": get_float("dr_z", 0.0),

                "dq_w": get_float("dq_w", 1.0),
                "dq_x": get_float("dq_x", 0.0),
                "dq_y": get_float("dq_y", 0.0),
                "dq_z": get_float("dq_z", 0.0),

                "dv_x": get_float("dv_x", 0.0),
                "dv_y": get_float("dv_y", 0.0),
                "dv_z": get_float("dv_z", 0.0),

                "dp_x": get_float("dp_x", 0.0),
                "dp_y": get_float("dp_y", 0.0),
                "dp_z": get_float("dp_z", 0.0),

                "imu_count": get_float("imu_count", 0.0),
                "imu_used_steps": get_float(
                    "imu_used_steps",
                    get_float("imu_count", 0.0),
                ),
                "imu_valid": get_int("imu_valid", 1),
                "imu_weight": get_float("imu_weight", 1.0),
                "imu_reason": row.get("imu_reason", ""),

                "rot_var": get_float("rot_var", 0.0),
                "vel_var": get_float("vel_var", 0.0),
                "pos_var": get_float("pos_var", 0.0),
                "rot_info": get_float("rot_info", 0.0),
                "vel_info": get_float("vel_info", 0.0),
                "pos_info": get_float("pos_info", 0.0),
                # Bias random-walk densities are static sensor-calibration
                # values, but keep them on each row so composed keyframe
                # intervals can carry the same uncertainty model at runtime.
                "gyro_random_walk": get_float("gyro_random_walk", 0.0),
                "accel_random_walk": get_float("accel_random_walk", 0.0),
                "imu_extrinsic_applied": get_int("imu_extrinsic_applied", 0),
                "imu_calibration_source": row.get("imu_calibration_source", ""),
            }
            fallback_jac = _fallback_bias_jacobians(parsed["dt"])
            for prefix in IMU_BIAS_JACOBIAN_ORDER:
                parsed.update(
                    _matrix_columns(
                        prefix,
                        _row_mat3(row, prefix, fallback_jac[prefix]),
                    )
                )

            imu_priors[frame_index] = parsed

    return imu_priors


def get_imu_prior_for_frame(imu_priors, frame_index):
    if not imu_priors:
        return None

    return imu_priors.get(int(frame_index))


def rotvec_to_quat_torch(rotvec, device="cuda", dtype=torch.float32):
    """
    rotation vector [rx, ry, rz]를 quaternion [qx, qy, qz, qw]로 변환한다.
    rotvec 단위는 rad 기준이다.
    """

    if not torch.is_tensor(rotvec):
        rotvec = torch.tensor(rotvec, device=device, dtype=dtype)
    else:
        rotvec = rotvec.to(device=device, dtype=dtype)

    angle = torch.linalg.norm(rotvec)

    eps = torch.tensor(1e-8, device=device, dtype=dtype)

    if angle.item() < 1e-8:
        return torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)

    axis = rotvec / torch.maximum(angle, eps)
    half = 0.5 * angle

    sin_half = torch.sin(half)
    cos_half = torch.cos(half)

    qx = axis[0] * sin_half
    qy = axis[1] * sin_half
    qz = axis[2] * sin_half
    qw = cos_half

    quat = torch.stack([qx, qy, qz, qw])
    quat = quat / torch.linalg.norm(quat).clamp(min=1e-8)

    return quat


def quat_multiply_torch(q1, q2):
    """
    quaternion multiply.
    q format: [qx, qy, qz, qw]
    return: q1 * q2
    """

    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    q = torch.stack([x, y, z, w])
    q = q / torch.linalg.norm(q).clamp(min=1e-8)

    return q


def imu_prior_to_delta_quat(imu_prior, rotation_weight=1.0, device="cuda"):
    """
    imu_prior row에서 dr_x, dr_y, dr_z를 읽어 delta quaternion으로 변환한다.
    rotation_weight는 IMU 회전 prior 강도 조절용이다.
    """

    if imu_prior is None:
        return torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32)

    dr = torch.tensor([
        float(imu_prior.get("dr_x", 0.0)),
        float(imu_prior.get("dr_y", 0.0)),
        float(imu_prior.get("dr_z", 0.0)),
    ], device=device, dtype=torch.float32)

    dr = dr * float(rotation_weight)

    return rotvec_to_quat_torch(dr, device=device, dtype=torch.float32)


class ImuRotationAccumulator:
    """
    프레임별 IMU 회전량을 누적해서 absolute-like rotation prior를 만든다.

    DROID-SLAM pose는 SE3 형태 [tx, ty, tz, qx, qy, qz, qw]로 들어간다.
    여기서는 translation은 직접 신뢰하지 않고, rotation만 누적한다.
    """

    def __init__(self, rotation_weight=1.0, device="cuda"):
        self.rotation_weight = rotation_weight
        self.device = device
        self.reset()

    def reset(self):
        self.q = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            device=self.device,
            dtype=torch.float32
        )

    def update(self, imu_prior):
        if imu_prior is None:
            return self.q

        imu_count = float(imu_prior.get("imu_count", 0.0))
        dt = float(imu_prior.get("dt", 0.0))

        # IMU가 없거나 dt가 비정상인 구간은 누적하지 않는다.
        if imu_count <= 0 or dt <= 0:
            return self.q

        dq = imu_prior_to_delta_quat(
            imu_prior,
            rotation_weight=self.rotation_weight,
            device=self.device
        )

        self.q = quat_multiply_torch(self.q, dq)
        self.q = self.q / torch.linalg.norm(self.q).clamp(min=1e-8)

        return self.q

    def get_se3_vec(self):
        """
        lietorch.SE3 입력용 7-vector 반환.
        [tx, ty, tz, qx, qy, qz, qw]
        translation은 0으로 둔다.
        """

        t = torch.zeros(3, device=self.device, dtype=torch.float32)
        return torch.cat([t, self.q], dim=0)

# droid_slam/imu_residual.py

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


DEBUG_FIELDNAMES = [
    "stage",
    "pose_ix",
    "prev_stamp",
    "curr_stamp",
    "prev_frame_index",
    "curr_frame_index",
    "gap",
    "residual_deg",
    "alpha",
    "alpha_base",
    "row_scale",
    "confidence",
    "applied",
    "reason",
]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def q_normalize(q: Any) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def q_inverse(q: Any) -> np.ndarray:
    q = q_normalize(q)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def q_mul(q1: Any, q2: Any) -> np.ndarray:
    """
    Quaternion multiplication.

    Input/output format is DROID pose format: [x, y, z, w].
    """

    x1, y1, z1, w1 = q_normalize(q1)
    x2, y2, z2, w2 = q_normalize(q2)

    return q_normalize(np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64))


def q_slerp(q0: Any, q1: Any, alpha: float) -> np.ndarray:
    q0 = q_normalize(q0)
    q1 = q_normalize(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        return q_normalize(q0 + alpha * (q1 - q0))

    theta_0 = math.acos(dot)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return q_normalize((s0 * q0) + (s1 * q1))


def q_angle_deg(q_err: Any) -> float:
    q_err = q_normalize(q_err)
    w = abs(float(q_err[3]))
    w = float(np.clip(w, -1.0, 1.0))
    return math.degrees(2.0 * math.acos(w))


def drvec_to_quat_xyzw(rx: float, ry: float, rz: float) -> np.ndarray:
    r = np.array([rx, ry, rz], dtype=np.float64)
    theta = np.linalg.norm(r)

    if theta < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    axis = r / theta
    half = theta * 0.5
    xyz = axis * math.sin(half)
    w = math.cos(half)

    return q_normalize(np.array([xyz[0], xyz[1], xyz[2], w], dtype=np.float64))


class ImuRotationResidual:
    """
    Rotation-only inertial residual regularizer.

    This is the first, lightweight form of E^u:

        r_u = Log(DeltaR_imu^-1 * DeltaR_visual)

    It does not rewrite DROID's CUDA DBA objective. Instead, it runs after visual
    BA and softly pulls the current pose rotation toward the IMU-predicted
    rotation. This keeps the code generic for EuRoC, phone captures, and any
    future data source that can produce the same `imu_prior.csv` schema.
    """

    def __init__(
        self,
        imu_prior_path: str,
        weight: float = 0.02,
        window: int = 12,
        compose_order: str = "prev_dq",
        inverse_imu: bool = False,
        max_residual_deg: float = 45.0,
        max_alpha: float = 0.05,
        max_frame_gap: int = 30,
        debug: bool = False,
        debug_path: Optional[str] = None,
    ):
        self.imu_prior_path = Path(imu_prior_path)
        self.weight = float(weight)
        self.window = int(window)
        self.compose_order = str(compose_order)
        self.inverse_imu = bool(inverse_imu)
        self.max_residual_deg = float(max_residual_deg)
        self.max_alpha = float(max_alpha)
        self.max_frame_gap = int(max_frame_gap)
        self.debug = bool(debug)

        if debug_path:
            self.debug_path = Path(debug_path)
        else:
            self.debug_path = self.imu_prior_path.parent / "imu_residual_debug.csv"

        self.priors_by_index: Dict[int, Dict[str, Any]] = {}
        self.priors_by_timestamp_ns: Dict[int, Dict[str, Any]] = {}
        self.debug_rows: List[Dict[str, Any]] = []

        self._load()

    def _load(self) -> None:
        if not self.imu_prior_path.exists():
            raise FileNotFoundError(f"imu_prior.csv not found: {self.imu_prior_path}")

        with open(self.imu_prior_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if _to_int(row.get("imu_valid"), 1) == 0:
                    continue

                if _to_int(row.get("imu_used_steps"), _to_int(row.get("imu_count"))) < 1:
                    continue

                frame_index = _to_int(
                    row.get("frame_index"),
                    _to_int(row.get("frame_id"), -1),
                )
                if frame_index < 0:
                    continue

                if all(row.get(c) not in (None, "") for c in ("dq_x", "dq_y", "dq_z", "dq_w")):
                    q = np.array([
                        _to_float(row.get("dq_x")),
                        _to_float(row.get("dq_y")),
                        _to_float(row.get("dq_z")),
                        _to_float(row.get("dq_w"), 1.0),
                    ], dtype=np.float64)
                elif all(row.get(c) not in (None, "") for c in ("dr_x", "dr_y", "dr_z")):
                    q = drvec_to_quat_xyzw(
                        _to_float(row.get("dr_x")),
                        _to_float(row.get("dr_y")),
                        _to_float(row.get("dr_z")),
                    )
                else:
                    continue

                if np.any(~np.isfinite(q)):
                    continue

                imu_weight_raw = _to_float(row.get("imu_weight"), 1.0)
                if row.get("imu_weight") in (None, ""):
                    row_scale = 1.0
                else:
                    # tools/imu_preintegrate.py writes 0.001 for a normal usable row.
                    row_scale = float(np.clip(imu_weight_raw / 0.001, 0.0, 1.0))

                timestamp_ns = row.get("timestamp_ns")
                timestamp_ns_int = None
                if timestamp_ns not in (None, ""):
                    timestamp_ns_int = _to_int(timestamp_ns, -1)
                    if timestamp_ns_int < 0:
                        timestamp_ns_int = None

                prior = {
                    "frame_index": frame_index,
                    "timestamp_ns": timestamp_ns_int,
                    "timestamp_sec": _to_float(row.get("timestamp_sec")),
                    "q": q_normalize(q),
                    "row_scale": row_scale,
                    "imu_weight_raw": imu_weight_raw,
                    "imu_reason": row.get("imu_reason", ""),
                }

                self.priors_by_index[frame_index] = prior
                if timestamp_ns_int is not None:
                    self.priors_by_timestamp_ns[timestamp_ns_int] = prior

        print(
            f"[IMU-RESIDUAL] loaded {len(self.priors_by_index)} priors "
            f"from {self.imu_prior_path}"
        )
        print(
            f"[IMU-RESIDUAL] weight={self.weight}, window={self.window}, "
            f"compose_order={self.compose_order}, inverse_imu={self.inverse_imu}"
        )

    def _get_pose_count(self, video: Any) -> int:
        if hasattr(video, "counter"):
            counter = video.counter
            if hasattr(counter, "value"):
                return int(counter.value)
            try:
                return int(counter)
            except Exception:
                pass

        return int(video.poses.shape[0])

    def _stamp_to_frame_index(self, stamp: Any) -> Optional[int]:
        value = _to_float(stamp, default=float("nan"))
        if not math.isfinite(value):
            return None

        # Normal DROID demo/test paths store frame index in video.tstamp.
        rounded = int(round(value))
        if abs(value - rounded) < 1e-3:
            if rounded in self.priors_by_index:
                return rounded

            # The first frame usually has no valid prior because there is no
            # previous frame, but it still needs to match as the start of the
            # first valid IMU interval 0 -> 1.
            if rounded + 1 in self.priors_by_index:
                return rounded

        # If a caller stores ns timestamps and the precision survived, allow exact lookup.
        if abs(value) > 1e12:
            prior = self.priors_by_timestamp_ns.get(int(round(value)))
            if prior is not None:
                return int(prior["frame_index"])

        return None

    def _compose_prior_between(
        self,
        prev_stamp: Any,
        curr_stamp: Any,
    ) -> Tuple[Optional[np.ndarray], float, Dict[str, Any]]:
        prev_index = self._stamp_to_frame_index(prev_stamp)
        curr_index = self._stamp_to_frame_index(curr_stamp)

        info = {
            "prev_frame_index": prev_index,
            "curr_frame_index": curr_index,
            "gap": None,
            "reason": "ok",
        }

        if prev_index is None or curr_index is None:
            info["reason"] = "stamp_not_matched"
            return None, 0.0, info

        if curr_index <= prev_index:
            info["gap"] = curr_index - prev_index
            info["reason"] = "non_forward_gap"
            return None, 0.0, info

        gap = curr_index - prev_index
        info["gap"] = gap

        if gap > self.max_frame_gap:
            info["reason"] = "gap_too_large"
            return None, 0.0, info

        q_total = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        row_scale = 1.0

        for frame_index in range(prev_index + 1, curr_index + 1):
            prior = self.priors_by_index.get(frame_index)
            if prior is None:
                info["reason"] = "missing_prior"
                return None, 0.0, info

            q_total = q_mul(q_total, prior["q"])
            row_scale = min(row_scale, float(prior["row_scale"]))

        if self.inverse_imu:
            q_total = q_inverse(q_total)

        return q_total, row_scale, info

    def _predict_current_q(self, q_prev: Any, q_imu: Any) -> np.ndarray:
        if self.compose_order == "prev_dq":
            return q_mul(q_prev, q_imu)
        if self.compose_order == "dq_prev":
            return q_mul(q_imu, q_prev)

        raise ValueError(f"Unknown compose_order: {self.compose_order}")

    def _record_debug(self, row: Dict[str, Any]) -> None:
        if self.debug:
            self.debug_rows.append(row)

    def _confidence_for_pose(self, imu_confidence: Any, pose_ix: int) -> float:
        if imu_confidence is None:
            return 1.0

        try:
            if torch.is_tensor(imu_confidence):
                conf = imu_confidence.detach()
                if conf.ndim == 2:
                    value = conf[0, pose_ix]
                else:
                    value = conf.reshape(-1)[pose_ix]
                out = float(value.detach().cpu().item())
            else:
                conf = np.asarray(imu_confidence, dtype=np.float64)
                if conf.ndim == 2:
                    out = float(conf[0, pose_ix])
                else:
                    out = float(conf.reshape(-1)[pose_ix])
        except Exception:
            return 1.0

        if not math.isfinite(out):
            return 1.0

        return float(np.clip(out, 0.0, 1.0))

    @torch.no_grad()
    def apply(self, video: Any, stage: str = "", imu_confidence: Any = None) -> int:
        n = self._get_pose_count(video)
        if n < 2:
            return 0

        start = max(1, n - self.window)
        poses = video.poses
        applied = 0

        for pose_ix in range(start, n):
            prev_stamp = video.tstamp[pose_ix - 1].detach().cpu().item()
            curr_stamp = video.tstamp[pose_ix].detach().cpu().item()

            q_imu, row_scale, info = self._compose_prior_between(
                prev_stamp,
                curr_stamp,
            )
            if q_imu is None:
                self._record_debug({
                    "stage": stage,
                    "pose_ix": pose_ix,
                    "prev_stamp": prev_stamp,
                    "curr_stamp": curr_stamp,
                    "applied": 0,
                    **info,
                })
                continue

            q_prev = q_normalize(poses[pose_ix - 1, 3:7].detach().cpu().numpy())
            q_curr = q_normalize(poses[pose_ix, 3:7].detach().cpu().numpy())
            q_pred = self._predict_current_q(q_prev, q_imu)

            q_err = q_mul(q_inverse(q_pred), q_curr)
            residual_deg = q_angle_deg(q_err)

            if not math.isfinite(residual_deg):
                self._record_debug({
                    "stage": stage,
                    "pose_ix": pose_ix,
                    "prev_stamp": prev_stamp,
                    "curr_stamp": curr_stamp,
                    "residual_deg": residual_deg,
                    "applied": 0,
                    "reason": "non_finite_residual",
                    **info,
                })
                continue

            if residual_deg > self.max_residual_deg:
                self._record_debug({
                    "stage": stage,
                    "pose_ix": pose_ix,
                    "prev_stamp": prev_stamp,
                    "curr_stamp": curr_stamp,
                    "residual_deg": residual_deg,
                    "applied": 0,
                    "reason": "too_large_residual",
                    "confidence": self._confidence_for_pose(imu_confidence, pose_ix),
                    **info,
                })
                continue

            confidence = self._confidence_for_pose(imu_confidence, pose_ix)
            alpha_base = self.weight * row_scale
            alpha = float(np.clip(alpha_base * confidence, 0.0, self.max_alpha))
            if alpha <= 0.0:
                self._record_debug({
                    "stage": stage,
                    "pose_ix": pose_ix,
                    "prev_stamp": prev_stamp,
                    "curr_stamp": curr_stamp,
                    "residual_deg": residual_deg,
                    "alpha": alpha,
                    "alpha_base": alpha_base,
                    "applied": 0,
                    "row_scale": row_scale,
                    "confidence": confidence,
                    "reason": "zero_alpha",
                    **info,
                })
                continue

            q_new = q_slerp(q_curr, q_pred, alpha)
            poses[pose_ix, 3:7] = torch.as_tensor(
                q_new,
                device=poses.device,
                dtype=poses.dtype,
            )

            if hasattr(video, "dirty"):
                try:
                    video.dirty[pose_ix] = True
                except Exception:
                    pass

            applied += 1
            self._record_debug({
                "stage": stage,
                "pose_ix": pose_ix,
                "prev_stamp": prev_stamp,
                "curr_stamp": curr_stamp,
                "residual_deg": residual_deg,
                "alpha": alpha,
                "alpha_base": alpha_base,
                "row_scale": row_scale,
                "confidence": confidence,
                "applied": 1,
                **info,
            })

        return applied

    def flush_debug(self) -> None:
        if not self.debug or not self.debug_rows:
            return

        self.debug_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.debug_path.exists()

        with open(self.debug_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=DEBUG_FIELDNAMES,
                extrasaction="ignore",
            )
            if write_header:
                writer.writeheader()
            writer.writerows(self.debug_rows)

        print(f"[IMU-RESIDUAL] debug appended: {self.debug_path}")
        self.debug_rows = []


def build_imu_rotation_residual_from_args(
    args: Any,
    stage: str = "",
) -> Optional[ImuRotationResidual]:
    needs_imu_prior = (
        getattr(args, "use_imu_residual", False)
        or getattr(args, "use_imu_ba_prior", False)
        or getattr(args, "use_full_imu_ba", False)
    )
    if not needs_imu_prior:
        return None

    imu_prior_path = getattr(args, "imu_prior", None)
    if imu_prior_path in (None, ""):
        print("[IMU-RESIDUAL] IMU prior path is missing; inertial terms will be disabled")
        return None

    debug_path = getattr(args, "imu_residual_debug_path", None)
    if debug_path in (None, ""):
        debug_path = None

    return ImuRotationResidual(
        imu_prior_path=imu_prior_path,
        weight=getattr(args, "imu_residual_weight", 0.02),
        window=getattr(args, "imu_residual_window", 12),
        compose_order=getattr(args, "imu_residual_compose_order", "prev_dq"),
        inverse_imu=getattr(args, "imu_residual_inverse", False),
        max_residual_deg=getattr(args, "imu_residual_max_deg", 45.0),
        max_alpha=getattr(args, "imu_residual_max_alpha", 0.05),
        max_frame_gap=getattr(args, "imu_residual_max_frame_gap", 30),
        debug=getattr(args, "imu_residual_debug", False),
        debug_path=debug_path,
    )

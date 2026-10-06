"""Shared input contract for camera-frame manifests and IMU priors.

The preintegration tool assigns ``frame_index`` after timestamp sorting.  The
runtime must consume the very same ordered manifest; independently enumerating
an image directory can silently attach an IMU interval to the wrong image when
a phone capture has a missing frame or non-timestamp filenames.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _finite_float(value: Any, field: str, row_number: int) -> float:
    try:
        if value is None or str(value).strip() == "":
            raise ValueError("missing value")
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"frames.csv row {row_number}: invalid {field}={value!r}"
        ) from exc

    if not math.isfinite(parsed):
        raise ValueError(
            f"frames.csv row {row_number}: non-finite {field}={value!r}"
        )
    return parsed


def _frame_filename(value: Any, row_number: int) -> str:
    filename = str(value or "").strip().replace("\\", "/")
    if not filename:
        raise ValueError(f"frames.csv row {row_number}: missing filename")

    path = PurePosixPath(filename)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"frames.csv row {row_number}: filename must be relative to --imagedir, "
            f"got {filename!r}"
        )
    return str(path)


def load_frame_manifest(frames_csv: Path) -> List[Dict[str, Any]]:
    """Load the timestamp-ordered image manifest used by preintegration.

    ``frame_index`` is deliberately reassigned *after* sorting.  This is the
    contract used by ``imu_prior.csv`` and must also be the timestamp passed to
    DROID at runtime.
    """

    frames_csv = Path(frames_csv).expanduser().resolve()
    if not frames_csv.is_file():
        raise FileNotFoundError(f"frames.csv does not exist: {frames_csv}")

    frames: List[Dict[str, Any]] = []
    with frames_csv.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"frames.csv has no header: {frames_csv}")

        for source_index, row in enumerate(reader):
            row_number = source_index + 2
            frame_id_value = row.get("frame_id", source_index)
            frame_id = int(_finite_float(frame_id_value, "frame_id", row_number))

            timestamp_ns_value = row.get("timestamp_ns")
            if timestamp_ns_value not in (None, ""):
                timestamp_ns = int(
                    _finite_float(timestamp_ns_value, "timestamp_ns", row_number)
                )
            else:
                timestamp_sec = _finite_float(
                    row.get("timestamp_sec"), "timestamp_sec", row_number
                )
                timestamp_ns = int(timestamp_sec * 1_000_000_000)

            if timestamp_ns <= 0:
                raise ValueError(
                    f"frames.csv row {row_number}: timestamp must be positive"
                )

            timestamp_sec_value = row.get("timestamp_sec")
            timestamp_sec = (
                _finite_float(timestamp_sec_value, "timestamp_sec", row_number)
                if timestamp_sec_value not in (None, "")
                else timestamp_ns / 1_000_000_000.0
            )

            frames.append(
                {
                    "frame_id": frame_id,
                    "frame_index": source_index,
                    "timestamp_ns": timestamp_ns,
                    "timestamp_sec": timestamp_sec,
                    "filename": _frame_filename(row.get("filename"), row_number),
                    "_source_index": source_index,
                }
            )

    if not frames:
        raise ValueError(f"frames.csv has no frame rows: {frames_csv}")

    # ``list.sort`` is stable; the source index also makes the tie-breaker
    # explicit and identical for preintegration and runtime.
    frames.sort(key=lambda frame: (frame["timestamp_ns"], frame["_source_index"]))
    for frame_index, frame in enumerate(frames):
        frame["frame_index"] = frame_index
        frame.pop("_source_index", None)

    return frames


def validate_frame_manifest_images(
    frames: Sequence[Mapping[str, Any]],
    image_dir: Path,
) -> None:
    """Fail before SLAM if a manifest image is absent or escapes image_dir."""

    image_dir = Path(image_dir).expanduser()
    errors = []
    for frame in frames:
        filename = _frame_filename(
            frame.get("filename"), int(frame.get("frame_index", -1)) + 2
        )
        image_path = image_dir / Path(filename)
        if not image_path.is_file():
            errors.append(
                f"frame_index={frame.get('frame_index')}, filename={filename!r}"
            )
            if len(errors) >= 8:
                break

    if errors:
        raise FileNotFoundError(
            "frames.csv/image directory mismatch; refusing to renumber frames after "
            "missing images:\n  - "
            + "\n  - ".join(errors)
        )


def validate_frame_prior_mapping(
    frames: Sequence[Mapping[str, Any]],
    imu_priors: Mapping[int, Mapping[str, Any]],
) -> None:
    """Verify that every manifest frame maps to its exact IMU-prior row."""

    errors = []
    expected_indices = {int(frame["frame_index"]) for frame in frames}
    extra_indices = sorted(set(int(index) for index in imu_priors) - expected_indices)
    if extra_indices:
        preview = ", ".join(str(index) for index in extra_indices[:8])
        errors.append(f"imu_prior has frame_index values absent from frames.csv: {preview}")

    for frame in frames:
        frame_index = int(frame["frame_index"])
        prior = imu_priors.get(frame_index)
        if prior is None:
            errors.append(f"frame_index={frame_index}: missing imu_prior row")
            if len(errors) >= 8:
                break
            continue

        checks = (
            ("frame_id", int(frame["frame_id"]), int(prior.get("frame_id", -1))),
            (
                "timestamp_ns",
                int(frame["timestamp_ns"]),
                int(prior.get("timestamp_ns", -1)),
            ),
            (
                "filename",
                str(frame["filename"]),
                str(prior.get("filename", "")),
            ),
        )
        for name, expected, actual in checks:
            if expected != actual:
                errors.append(
                    f"frame_index={frame_index}: {name} expected={expected!r}, "
                    f"imu_prior={actual!r}"
                )
                break
        if len(errors) >= 8:
            break

    if errors:
        raise ValueError(
            "frames.csv and imu_prior.csv do not describe the same ordered "
            "camera sequence. Rebuild imu_prior.csv from this frames.csv:\n  - "
            + "\n  - ".join(errors)
        )


def selected_frame_records(
    frames: Sequence[Mapping[str, Any]],
    stride: int,
    max_frames: Optional[int] = None,
) -> List[Mapping[str, Any]]:
    """Apply DROID sampling without changing original camera frame indices."""

    stride = max(1, int(stride))
    selected = list(frames)[::stride]
    if max_frames is not None and int(max_frames) > 0:
        selected = selected[: int(max_frames)]
    return selected


def frame_manifest_digest(frames: Iterable[Mapping[str, Any]]) -> str:
    """Return a deterministic digest of the runtime frame-to-IMU mapping."""

    digest = hashlib.sha256()
    digest.update(b"droid-frame-manifest-v1\0")
    for frame in frames:
        digest.update(
            (
                f"{int(frame['frame_index'])}:"
                f"{int(frame['frame_id'])}:"
                f"{int(frame['timestamp_ns'])}:"
                f"{frame['filename']}\n"
            ).encode("utf-8", "surrogateescape")
        )
    return digest.hexdigest()


def make_invalid_imu_prior(
    reason: str,
    *,
    frame_index: Optional[int] = None,
    prev_frame_index: Optional[int] = None,
    fallback: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Make an explicit unusable IMU edge instead of substituting another one."""

    prior: Dict[str, Any] = dict(fallback or {})
    if frame_index is not None:
        prior["frame_index"] = int(frame_index)
    if prev_frame_index is not None:
        prior["prev_frame_index"] = int(prev_frame_index)

    prior.update(
        {
            "dt": 0.0,
            "imu_count": 0,
            "imu_used_steps": 0,
            "imu_valid": 0,
            "imu_weight": 0.0,
            "imu_reason": str(reason),
            "dr_x": 0.0,
            "dr_y": 0.0,
            "dr_z": 0.0,
            "dr_norm": 0.0,
            "dr_norm_deg": 0.0,
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
            "rot_var": 0.0,
            "vel_var": 0.0,
            "pos_var": 0.0,
            "rot_info": 0.0,
            "vel_info": 0.0,
            "pos_info": 0.0,
        }
    )
    return prior

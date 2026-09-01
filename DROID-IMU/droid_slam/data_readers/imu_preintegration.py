"""
Importable IMU preintegration adapter for DROID-SLAM data readers.

The actual preintegration implementation lives in `tools/imu_preintegrate.py`
because the server and terminal workflow run it as a script. This module keeps a
stable Python API for dataset loaders and experiments without duplicating the
integration algorithm.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Union

try:
    from imu_prior import load_imu_prior_csv
except ImportError:
    from droid_slam.imu_prior import load_imu_prior_csv


_TOOL_MODULE: Optional[ModuleType] = None


def _load_preintegration_tool() -> ModuleType:
    global _TOOL_MODULE

    if _TOOL_MODULE is not None:
        return _TOOL_MODULE

    tool_path = Path(__file__).resolve().parents[2] / "tools" / "imu_preintegrate.py"
    if not tool_path.exists():
        raise FileNotFoundError(f"imu_preintegrate.py not found: {tool_path}")

    spec = importlib.util.spec_from_file_location(
        "droid_slam_tools_imu_preintegrate",
        str(tool_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load imu_preintegrate.py: {tool_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _TOOL_MODULE = module
    return module


def build_imu_prior_csv(
    frames_csv: Union[str, Path],
    imu_csv: Union[str, Path],
    output_csv: Union[str, Path],
    cam_sensor_yaml: Optional[Union[str, Path]] = None,
    imu_sensor_yaml: Optional[Union[str, Path]] = None,
    imu_to_cam_rotation=None,
    gyro_bias=None,
    acc_bias=None,
) -> Dict[str, Any]:
    """
    Build `imu_prior.csv` from generic `frames.csv` and `imu.csv` files.

    This is intentionally dataset-neutral. EuRoC, phone captures, or any future
    source only need to be converted to the same CSV schema before calling this.
    """

    module = _load_preintegration_tool()
    calibration = module.build_imu_calibration(
        cam_sensor_yaml=Path(cam_sensor_yaml).resolve() if cam_sensor_yaml else None,
        imu_sensor_yaml=Path(imu_sensor_yaml).resolve() if imu_sensor_yaml else None,
        imu_to_cam_rotation=imu_to_cam_rotation,
        gyro_bias=gyro_bias,
        acc_bias=acc_bias,
    )
    return module.build_imu_prior(
        Path(frames_csv).resolve(),
        Path(imu_csv).resolve(),
        Path(output_csv).resolve(),
        calibration=calibration,
    )


def build_imu_prior_for_session(
    session_dir: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    cam_sensor_yaml: Optional[Union[str, Path]] = None,
    imu_sensor_yaml: Optional[Union[str, Path]] = None,
    imu_to_cam_rotation=None,
    gyro_bias=None,
    acc_bias=None,
) -> Dict[str, Any]:
    """Build `imu_prior.csv` for a session containing `frames.csv` and `imu.csv`."""

    session_dir = Path(session_dir).resolve()
    output = Path(output_csv).resolve() if output_csv else session_dir / "imu_prior.csv"

    return build_imu_prior_csv(
        session_dir / "frames.csv",
        session_dir / "imu.csv",
        output,
        cam_sensor_yaml=cam_sensor_yaml,
        imu_sensor_yaml=imu_sensor_yaml,
        imu_to_cam_rotation=imu_to_cam_rotation,
        gyro_bias=gyro_bias,
        acc_bias=acc_bias,
    )


def load_preintegrated_imu(path: Union[str, Path]):
    """Load an `imu_prior.csv` file keyed by frame index."""

    return load_imu_prior_csv(path)

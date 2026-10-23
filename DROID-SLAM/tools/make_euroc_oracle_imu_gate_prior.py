#!/usr/bin/env python3
"""Make a *diagnostic-only* GT-oracle mask for an EuRoC IMU prior.

This program must never be used for a reportable SLAM result.  It uses the
held-out trajectory's ground truth *after* a completed run to answer a narrow
upper-bound question:

    Would the existing 15DoF objective help if it received only physically
    accurate preintegrated IMU intervals?

The input edge CSV is produced by ``audit_euroc_preintegration.py`` with
``--apply-reconstruction-bias``.  Each audited edge is one final global-BA
keyframe interval.  If any of rotation / velocity / position error exceeds the
chosen bound, every raw prior row in that interval is marked invalid.  At
runtime ``compose_imu_prior_rows()`` therefore disables an arbitrary composed
factor whenever it spans one of those rows.

The resulting CSV is intentionally labelled ``oracle_gt_*`` and is valid only
as a one-off diagnostic.  It leaks held-out GT and cannot train, select, or
claim the final learned model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np
import torch


def _as_int(value: object, name: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if not np.isfinite(numeric) or abs(numeric - round(numeric)) > 1e-6:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(round(numeric))


def _as_float(value: object, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite, got {value!r}") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return numeric


def _load_keyframes(path: Path) -> List[int]:
    blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict) or "tstamps" not in blob:
        raise ValueError(f"reconstruction lacks a tstamps tensor: {path}")
    values = np.asarray(blob["tstamps"], dtype=np.float64).reshape(-1)
    keyframes = [_as_int(value, "reconstruction tstamp") for value in values]
    if len(keyframes) < 2:
        raise ValueError("reconstruction needs at least two keyframes")
    if any(current <= previous for previous, current in zip(keyframes[:-1], keyframes[1:])):
        raise ValueError("reconstruction tstamps must be strictly increasing")
    return keyframes


def _load_audit_edges(path: Path) -> Mapping[int, Mapping[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "frame_index",
            "rotation_error_deg",
            "dv_error_norm_mps",
            "dp_error_norm_m",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"audit CSV must contain {sorted(required)}, got {reader.fieldnames}: {path}"
            )
        result: Dict[int, Mapping[str, str]] = {}
        for row in reader:
            index = _as_int(row["frame_index"], "audit frame_index")
            if index in result:
                raise ValueError(f"duplicate audit edge endpoint {index}: {path}")
            result[index] = row
    return result


def _reason_with_oracle_tag(existing: object) -> str:
    previous = str(existing or "").strip()
    tag = "oracle_gt_rvp_gate"
    if tag in previous.split("|"):
        return previous
    return f"{previous}|{tag}" if previous else tag


def make_gate(args: argparse.Namespace) -> Dict[str, object]:
    input_prior = Path(args.input_prior).expanduser().resolve()
    reconstruction = Path(args.reconstruction).expanduser().resolve()
    audit_edges = Path(args.audit_edges_csv).expanduser().resolve()
    output_prior = Path(args.output_prior).expanduser().resolve()
    for path in (input_prior, reconstruction, audit_edges):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_prior.exists():
        raise FileExistsError(f"refusing to overwrite existing oracle prior: {output_prior}")

    keyframes = _load_keyframes(reconstruction)
    audit_by_endpoint = _load_audit_edges(audit_edges)
    thresholds = {
        "rotation_error_deg": float(args.max_rotation_error_deg),
        "dv_error_norm_mps": float(args.max_velocity_error_mps),
        "dp_error_norm_m": float(args.max_position_error_m),
    }

    interval_records: List[Dict[str, object]] = []
    gated_raw_indices: Set[int] = set()
    for previous, current in zip(keyframes[:-1], keyframes[1:]):
        audit = audit_by_endpoint.get(current)
        if audit is None:
            good = False
            status = "missing_gt_audit"
            errors = {name: None for name in thresholds}
        else:
            errors = {name: _as_float(audit[name], name) for name in thresholds}
            good = all(errors[name] <= threshold for name, threshold in thresholds.items())
            status = "kept" if good else "rvp_threshold_exceeded"
        if not good:
            gated_raw_indices.update(range(previous + 1, current + 1))
        interval_records.append(
            {
                "previous_frame_index": previous,
                "frame_index": current,
                "oracle_keep": int(good),
                "status": status,
                **errors,
            }
        )

    with input_prior.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"prior has no CSV header: {input_prior}")
        required_prior = {"frame_index", "imu_valid", "imu_weight", "imu_reason"}
        if not required_prior.issubset(reader.fieldnames):
            raise ValueError(
                f"prior must contain {sorted(required_prior)}, got {reader.fieldnames}"
            )
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    prior_indices = {_as_int(row["frame_index"], "prior frame_index") for row in rows}
    missing_prior = sorted(gated_raw_indices - prior_indices)
    if missing_prior:
        raise ValueError(
            "oracle intervals reference raw prior rows that do not exist; first missing: "
            f"{missing_prior[:10]}"
        )

    changed_rows = 0
    originally_invalid = 0
    for row in rows:
        frame_index = _as_int(row["frame_index"], "prior frame_index")
        if frame_index not in gated_raw_indices:
            continue
        if _as_int(row["imu_valid"], "imu_valid") == 0:
            originally_invalid += 1
        else:
            changed_rows += 1
        row["imu_valid"] = "0"
        row["imu_weight"] = "0"
        row["imu_reason"] = _reason_with_oracle_tag(row.get("imu_reason"))

    output_prior.parent.mkdir(parents=True, exist_ok=True)
    with output_prior.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

    summary: Dict[str, object] = {
        "diagnostic_only": True,
        "warning": (
            "This prior uses held-out ground truth after the fact. It is an oracle upper-bound "
            "diagnostic only and must not be reported as a learned or deployable result."
        ),
        "input_prior": str(input_prior),
        "reconstruction_used_to_define_keyframe_intervals": str(reconstruction),
        "audit_edges_csv": str(audit_edges),
        "output_prior": str(output_prior),
        "thresholds": {
            "rotation_error_deg_max": thresholds["rotation_error_deg"],
            "velocity_error_mps_max": thresholds["dv_error_norm_mps"],
            "position_error_m_max": thresholds["dp_error_norm_m"],
        },
        "keyframes": len(keyframes),
        "keyframe_intervals": len(interval_records),
        "oracle_kept_intervals": int(sum(record["oracle_keep"] for record in interval_records)),
        "oracle_gated_intervals": int(sum(not record["oracle_keep"] for record in interval_records)),
        "gated_raw_prior_rows": len(gated_raw_indices),
        "changed_valid_rows": changed_rows,
        "already_invalid_gated_rows": originally_invalid,
        "runtime_note": (
            "The static raw-row mask is conservative if a rerun chooses a different keyframe "
            "interval: any composed factor spanning a gated raw row is disabled."
        ),
    }
    summary_path = output_prior.with_suffix(".oracle_summary.json")
    intervals_path = output_prior.with_suffix(".oracle_intervals.csv")
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    with intervals_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(interval_records[0]))
        writer.writeheader()
        writer.writerows(interval_records)

    print("[ORACLE-GATE]", json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-prior", required=True)
    parser.add_argument("--reconstruction", required=True)
    parser.add_argument("--audit-edges-csv", required=True)
    parser.add_argument("--output-prior", required=True)
    parser.add_argument("--max-rotation-error-deg", type=float, default=0.2)
    parser.add_argument("--max-velocity-error-mps", type=float, default=0.1)
    parser.add_argument("--max-position-error-m", type=float, default=0.02)
    args = parser.parse_args()
    for name in (
        "max_rotation_error_deg",
        "max_velocity_error_mps",
        "max_position_error_m",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and > 0")
    return args


if __name__ == "__main__":
    make_gate(parse_args())

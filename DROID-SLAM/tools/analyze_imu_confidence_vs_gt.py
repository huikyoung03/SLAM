#!/usr/bin/env python3
"""Offline oracle audit of learned IMU confidence against GT factor error.

This is deliberately *not* an inference or ATE evaluation path.  It joins:

1. a reconstruction's keyframe ordering;
2. the runtime IMU-BA debug CSV (confidence actually passed to BA); and
3. ``audit_euroc_preintegration.py``'s GT-only per-interval residual audit.

The result answers one diagnostic question: does a lower learned confidence
correspond to a physically less accurate preintegrated IMU interval?  GT is
used only after a run has finished and is never written to a runtime prior.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch


def _finite(value: str, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks without a scipy dependency."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    begin = 0
    while begin < values.size:
        end = begin + 1
        while end < values.size and sorted_values[end] == sorted_values[begin]:
            end += 1
        ranks[order[begin:end]] = 0.5 * (begin + end - 1)
        begin = end
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> Dict[str, float]:
    if first.size < 3:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": float(np.corrcoef(first, second)[0, 1]),
        "spearman": float(np.corrcoef(_rank(first), _rank(second))[0, 1]),
    }


def _quantile_groups(confidence: np.ndarray, error: np.ndarray, fraction: float):
    low_threshold = float(np.quantile(confidence, fraction))
    high_threshold = float(np.quantile(confidence, 1.0 - fraction))
    low = error[confidence <= low_threshold]
    high = error[confidence >= high_threshold]
    return {
        "fraction": float(fraction),
        "low_confidence_count": int(low.size),
        "high_confidence_count": int(high.size),
        "low_confidence_error_mean": float(np.mean(low)),
        "high_confidence_error_mean": float(np.mean(high)),
        "low_confidence_threshold": low_threshold,
        "high_confidence_threshold": high_threshold,
    }


def _load_keyframe_lookup(path: Path) -> Dict[int, int]:
    blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict) or "tstamps" not in blob:
        raise ValueError(f"reconstruction needs a tstamps tensor: {path}")
    tstamps = np.asarray(blob["tstamps"], dtype=np.float64).reshape(-1)
    lookup: Dict[int, int] = {}
    for index, value in enumerate(tstamps):
        frame_index = int(round(float(value)))
        if abs(float(value) - frame_index) > 1e-4:
            raise ValueError(f"non-integer reconstruction timestamp: {value}")
        lookup[frame_index] = index
    return lookup


def _load_runtime_confidence(path: Path, stage: str):
    required = {
        "curr_ix",
        "reason",
        "stage",
        "confidence_rot",
        "confidence_vel",
        "confidence_pos",
        "confidence",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"debug CSV is missing required fields {sorted(required)}: {path}"
            )
        buckets = defaultdict(lambda: [[], [], [], []])
        for row in reader:
            if row.get("reason") != "used" or row.get("stage") != stage:
                continue
            keyframe_index = int(_finite(row["curr_ix"], "curr_ix"))
            values = [
                _finite(row["confidence_rot"], "confidence_rot"),
                _finite(row["confidence_vel"], "confidence_vel"),
                _finite(row["confidence_pos"], "confidence_pos"),
                _finite(row["confidence"], "confidence"),
            ]
            for bucket, value in zip(buckets[keyframe_index], values):
                bucket.append(value)
    return {
        key: np.asarray([np.mean(values) for values in channels], dtype=np.float64)
        for key, channels in buckets.items()
        if all(channels)
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    reconstruction = Path(args.reconstruction).expanduser().resolve()
    debug_csv = Path(args.debug_csv).expanduser().resolve()
    audit_csv = Path(args.audit_edges_csv).expanduser().resolve()
    for path in (reconstruction, debug_csv, audit_csv):
        if not path.is_file():
            raise FileNotFoundError(path)

    frame_to_keyframe = _load_keyframe_lookup(reconstruction)
    confidence_by_keyframe = _load_runtime_confidence(debug_csv, args.stage)
    records: List[Dict[str, float]] = []
    with audit_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "frame_index",
            "rotation_error_deg",
            "dv_error_norm_mps",
            "dp_error_norm_m",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"audit CSV is missing required fields {sorted(required)}: {audit_csv}"
            )
        for row in reader:
            frame_index = int(_finite(row["frame_index"], "frame_index"))
            keyframe_index = frame_to_keyframe.get(frame_index)
            if keyframe_index is None:
                continue
            confidence = confidence_by_keyframe.get(keyframe_index)
            if confidence is None:
                continue
            records.append(
                {
                    "frame_index": frame_index,
                    "keyframe_index": keyframe_index,
                    "confidence_rot": float(confidence[0]),
                    "confidence_vel": float(confidence[1]),
                    "confidence_pos": float(confidence[2]),
                    "confidence_mean": float(confidence[3]),
                    "rotation_error_deg": _finite(row["rotation_error_deg"], "rotation_error_deg"),
                    "velocity_error_mps": _finite(row["dv_error_norm_mps"], "dv_error_norm_mps"),
                    "position_error_m": _finite(row["dp_error_norm_m"], "dp_error_norm_m"),
                }
            )
    if len(records) < 3:
        raise RuntimeError("fewer than three matching runtime/GT intervals")

    pairs = {
        "rotation": ("confidence_rot", "rotation_error_deg"),
        "velocity": ("confidence_vel", "velocity_error_mps"),
        "position": ("confidence_pos", "position_error_m"),
    }
    summary: Dict[str, object] = {
        "purpose": "offline_GT_only_confidence_quality_audit",
        "reconstruction": str(reconstruction),
        "debug_csv": str(debug_csv),
        "audit_edges_csv": str(audit_csv),
        "stage": args.stage,
        "matched_intervals": int(len(records)),
        "expected_sign": "negative: higher confidence should mean lower GT factor error",
        "components": {},
    }
    for name, (confidence_key, error_key) in pairs.items():
        confidence = np.asarray([record[confidence_key] for record in records])
        error = np.asarray([record[error_key] for record in records])
        summary["components"][name] = {
            "confidence_mean": float(np.mean(confidence)),
            "confidence_p05_p50_p95": [float(value) for value in np.quantile(confidence, [0.05, 0.5, 0.95])],
            "error_p50_p90_p95": [float(value) for value in np.quantile(error, [0.5, 0.9, 0.95])],
            **_correlation(confidence, error),
            **_quantile_groups(confidence, error, args.group_fraction),
        }

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "confidence_vs_gt_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        with (output_dir / "confidence_vs_gt_edges.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    print("[CONFIDENCE-VS-GT]", json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", required=True)
    parser.add_argument("--debug-csv", required=True)
    parser.add_argument("--audit-edges-csv", required=True)
    parser.add_argument("--stage", choices=("frontend", "backend"), default="backend")
    parser.add_argument("--group-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if not 0.0 < args.group_fraction < 0.5:
        parser.error("--group-fraction must be in (0, 0.5)")
    return args


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""Summarize dense-depth and IMU-factor health from saved reconstructions.

The report is intentionally diagnostic rather than an accuracy evaluator.
It identifies frames whose disparity distribution is disproportionate to the
sequence median and aggregates the IMU residual rows produced by --imu_ba_debug.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def _float(value, default=np.nan):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _percentile(values, q):
    return float(np.percentile(values, q)) if values.size else float("nan")


def _load_imu_debug(path, timestamps):
    """Aggregate repeated full-IMU factor rows by destination keyframe."""

    if not path:
        return {}
    debug_path = Path(path)
    if not debug_path.is_file():
        return {}

    values = defaultdict(lambda: defaultdict(list))
    skipped = defaultdict(int)
    with debug_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            curr_ix = int(_float(row.get("curr_ix"), -1))
            if curr_ix < 0 or curr_ix >= len(timestamps):
                continue
            if row.get("reason") != "used":
                skipped[curr_ix] += 1
                continue
            for key in (
                "r_R_deg",
                "r_v_norm",
                "r_p_norm",
                "r_ba_norm",
                "r_bg_norm",
                "confidence",
                "confidence_rot",
                "confidence_vel",
                "confidence_pos",
                "weight",
            ):
                value = _float(row.get(key))
                if np.isfinite(value):
                    values[curr_ix][key].append(value)

    result = {}
    for index in set(values) | set(skipped):
        row = {"imu_factor_samples": sum(len(v) for v in values[index].values())}
        row["imu_factor_skipped_rows"] = skipped[index]
        for key, samples in values[index].items():
            row[f"imu_{key}_mean"] = float(np.mean(samples))
            row[f"imu_{key}_p95"] = _percentile(np.asarray(samples), 95)
        result[index] = row
    return result


def _summarize_reconstruction(label, path, debug_path, output_dir):
    blob = torch.load(path, map_location="cpu")
    disparities = torch.as_tensor(blob["disps"]).detach().cpu().numpy()
    timestamps = torch.as_tensor(blob["tstamps"]).detach().cpu().reshape(-1).numpy()
    if disparities.shape[0] != len(timestamps):
        raise ValueError(f"{path}: disparity/timestamp count mismatch")

    finite_positive = np.isfinite(disparities) & (disparities > 0.0)
    all_values = disparities[finite_positive]
    if all_values.size == 0:
        raise ValueError(f"{path}: no finite positive disparities")

    sequence_median = _percentile(all_values, 50)
    threshold_10x = 10.0 * sequence_median
    debug = _load_imu_debug(debug_path, timestamps)
    frame_rows = []
    for index, frame in enumerate(disparities):
        values = frame[np.isfinite(frame) & (frame > 0.0)]
        row = {
            "frame_index": index,
            "frame_id": int(round(float(timestamps[index]))),
            "valid_disparity_pixels": int(values.size),
            "disp_q01": _percentile(values, 1),
            "disp_median": _percentile(values, 50),
            "disp_q95": _percentile(values, 95),
            "disp_q99": _percentile(values, 99),
            "disp_max": float(np.max(values)) if values.size else float("nan"),
            "disp_max_over_sequence_median": (
                float(np.max(values) / sequence_median) if values.size else float("nan")
            ),
            "disp_pixels_over_10x_sequence_median_ratio": (
                float(np.mean(values > threshold_10x)) if values.size else float("nan")
            ),
        }
        row.update(debug.get(index, {}))
        frame_rows.append(row)

    frame_path = output_dir / f"{label}_frame_health.csv"
    fields = sorted({key for row in frame_rows for key in row})
    with frame_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frame_rows)

    state = blob.get("imu_state_diagnostics", {})
    summary = {
        "label": label,
        "reconstruction": str(Path(path).resolve()),
        "keyframes": int(len(timestamps)),
        "imu_alignment_ready": int(torch.as_tensor(blob.get("imu_alignment_ready", 0)).reshape(-1)[0]),
        "disparity_q01": _percentile(all_values, 1),
        "disparity_median": sequence_median,
        "disparity_q95": _percentile(all_values, 95),
        "disparity_q99": _percentile(all_values, 99),
        "disparity_max": float(np.max(all_values)),
        "disparity_max_over_median": float(np.max(all_values) / sequence_median),
        "disparity_over_10x_median_ratio": float(np.mean(all_values > threshold_10x)),
        "top_outlier_frame_ids": [
            row["frame_id"]
            for row in sorted(
                frame_rows,
                key=lambda row: row["disp_max_over_sequence_median"],
                reverse=True,
            )[:10]
        ],
        "imu_state_diagnostics": state,
        "frame_health_csv": str(frame_path.resolve()),
    }
    with (output_dir / f"{label}_health.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recon",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="saved reconstruction to summarize; repeat for each stage",
    )
    parser.add_argument(
        "--imu-debug",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="optional --imu_ba_debug CSV for a matching reconstruction",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    def split(items):
        result = {}
        for item in items:
            if "=" not in item:
                raise ValueError(f"expected LABEL=PATH, got: {item}")
            label, path = item.split("=", 1)
            if not label or not path:
                raise ValueError(f"expected LABEL=PATH, got: {item}")
            result[label] = path
        return result

    reconstructions = split(args.recon)
    debug_paths = split(args.imu_debug)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        _summarize_reconstruction(label, path, debug_paths.get(label), output_dir)
        for label, path in reconstructions.items()
    ]
    fields = [
        "label",
        "keyframes",
        "imu_alignment_ready",
        "disparity_q01",
        "disparity_median",
        "disparity_q95",
        "disparity_q99",
        "disparity_max",
        "disparity_max_over_median",
        "disparity_over_10x_median_ratio",
        "top_outlier_frame_ids",
        "frame_health_csv",
    ]
    csv_path = output_dir / "health_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row = dict(summary)
            row["top_outlier_frame_ids"] = " ".join(
                str(frame_id) for frame_id in summary["top_outlier_frame_ids"]
            )
            writer.writerow({key: row.get(key) for key in fields})

    md_path = output_dir / "health_summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Reconstruction Health\n\n")
        handle.write(
            "| label | keyframes | aligned | disp median | disp p99 | "
            "disp max/median | pixels >10x median |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for summary in summaries:
            handle.write(
                f"| {summary['label']} | {summary['keyframes']} | "
                f"{summary['imu_alignment_ready']} | "
                f"{summary['disparity_median']:.6g} | "
                f"{summary['disparity_q99']:.6g} | "
                f"{summary['disparity_max_over_median']:.6g} | "
                f"{summary['disparity_over_10x_median_ratio']:.6g} |\n"
            )
        handle.write("\nTop outlier frame IDs and per-frame details are in the matching JSON/CSV files.\n")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()

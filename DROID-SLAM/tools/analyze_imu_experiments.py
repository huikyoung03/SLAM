#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METRICS = [
    ("rot_error", "lower"),
    ("tr_error", "lower"),
    ("residual", "lower"),
    ("f_error", "lower"),
    ("1px", "higher"),
    ("imu_rot_loss", "lower"),
    ("imu_full_loss", "lower"),
    ("imu_pos_loss", "lower"),
    ("imu_vel_loss", "lower"),
    ("imu_bias_loss", "lower"),
    ("imu_pos_error", "lower"),
    ("imu_vel_error", "lower"),
    ("imu_rot_error", "lower"),
    ("imu_bias_error", "lower"),
    ("imu_conf_mean", "info"),
    ("imu_conf_min", "info"),
    ("imu_conf_max", "info"),
    ("imu_ba_weight", "info"),
    ("imu_bias_norm", "info"),
    ("imu_acc_bias_norm", "info"),
]


def parse_run(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=RUN_DIR_NAME")

    label, name = value.split("=", 1)
    label = label.strip()
    name = name.strip()
    if not label or not name:
        raise argparse.ArgumentTypeError("run must be LABEL=RUN_DIR_NAME")

    return label, name


def load_scalars(run_dir):
    if not run_dir.exists():
        return {}

    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return {}

    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))

    out = {}
    for metric, _ in METRICS:
        if metric not in tags:
            continue
        values = [event.value for event in accumulator.Scalars(metric)]
        if not values:
            continue
        out[metric] = {
            "n": len(values),
            "mean": sum(values) / len(values),
            "last5": sum(values[-5:]) / min(5, len(values)),
            "last": values[-1],
        }

    return out


def fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.6f}"


def best_label(summary, metric, mode, aggregate):
    if mode not in {"lower", "higher"}:
        return None

    candidates = []
    for label, data in summary.items():
        if metric in data:
            candidates.append((label, data[metric][aggregate]))

    if not candidates:
        return None

    if mode == "lower":
        return min(candidates, key=lambda item: item[1])[0]
    return max(candidates, key=lambda item: item[1])[0]


def write_csv(path, labels, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["metric"]
        for aggregate in ("mean", "last5", "last"):
            header.extend(f"{label}_{aggregate}" for label in labels)
        writer.writerow(header)

        for metric, _ in METRICS:
            row = [metric]
            for aggregate in ("mean", "last5", "last"):
                for label in labels:
                    row.append(summary.get(label, {}).get(metric, {}).get(aggregate, ""))
            writer.writerow(row)


def write_markdown(path, labels, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# IMU Experiment Comparison")
    lines.append("")
    lines.append("Lower is better for error metrics. Higher is better for `1px`.")
    lines.append("")

    for aggregate in ("mean", "last5", "last"):
        lines.append(f"## {aggregate}")
        lines.append("")
        lines.append("| metric | " + " | ".join(labels) + " | best |")
        lines.append("|---|" + "|".join(["---:"] * len(labels)) + "|---|")
        for metric, mode in METRICS:
            values = [
                summary.get(label, {}).get(metric, {}).get(aggregate, float("nan"))
                for label in labels
            ]
            best = best_label(summary, metric, mode, aggregate)
            lines.append(
                "| "
                + metric
                + " | "
                + " | ".join(fmt(value) for value in values)
                + " | "
                + (best or "-")
                + " |"
            )
        lines.append("")

    baseline = summary.get("baseline", {})
    full = summary.get("full_long", {})
    rot = summary.get("rotation_only", {})
    if baseline and full:
        lines.append("## Quick Read")
        lines.append("")
        for metric in ("rot_error", "tr_error", "f_error", "1px"):
            base_value = baseline.get(metric, {}).get("mean")
            full_value = full.get(metric, {}).get("mean")
            rot_value = rot.get(metric, {}).get("mean") if rot else None
            if base_value is None or full_value is None:
                continue
            if metric == "1px":
                direction = "higher"
                delta = full_value - base_value
            else:
                direction = "lower"
                delta = base_value - full_value
            lines.append(
                f"- `{metric}` mean: baseline={fmt(base_value)}, "
                f"rotation_only={fmt(rot_value)}, full_long={fmt(full_value)} "
                f"({direction} is better, full-baseline delta={delta:.6f})"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-md", default="logs/imu_comparison_report.md")
    parser.add_argument("--output-csv", default="logs/imu_comparison_summary.csv")
    args = parser.parse_args()

    labels = [label for label, _ in args.run]
    summary = {}
    runs_root = Path(args.runs_root)
    for label, name in args.run:
        summary[label] = load_scalars(runs_root / name)

    write_markdown(Path(args.output_md), labels, summary)
    write_csv(Path(args.output_csv), labels, summary)

    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()

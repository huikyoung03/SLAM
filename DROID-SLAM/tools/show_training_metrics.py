#!/usr/bin/env python3
"""Print the newest scalar values from a DROID TensorBoard training run.

Examples:
  python tools/show_training_metrics.py runs/full15dof_confidence_only_3000 --total-steps 3000
  python tools/show_training_metrics.py runs/full15dof_confidence_only_3000 --total-steps 3000 --watch 10
"""

import argparse
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_TAGS = (
    "loss_total",
    "imu_full_loss",
    "imu_conf_mean",
    "imu_conf_min",
    "imu_conf_max",
    "imu_precision_mean",
    "imu_precision_rot_mean",
    "imu_precision_vel_mean",
    "imu_precision_pos_mean",
    "imu_precision_clean_mean",
    "imu_precision_augmented_mean",
    "imu_precision_target_mean",
    "imu_precision_augmentation_calibration",
    "loss_imu_precision_augmentation_calibration_weighted",
    "imu_confidence_nll",
    "imu_ba_weight",
    "imu_ba_checkpoint_weight_ignored",
    "imu_pos_weight",
    "imu_vel_weight",
    "imu_residual_weight_prior_loss",
    "loss_residual_weight_prior_weighted",
)


def latest_scalars(run_dir, tags):
    files = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no TensorBoard event file under {run_dir}")

    accumulator = EventAccumulator(str(files[-1]), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    result = {}
    for tag in tags:
        if tag in available:
            values = accumulator.Scalars(tag)
            if values:
                result[tag] = values[-1]
    return files[-1], result


def print_snapshot(run_dir, total_steps, tags):
    event_file, values = latest_scalars(run_dir, tags)
    if not values:
        print(f"[WAIT] {event_file.name}: no requested scalar has been written yet")
        return

    step = max(value.step for value in values.values())
    progress = ""
    if total_steps is not None:
        progress = f" ({100.0 * step / total_steps:.1f}% of {total_steps})"
    print(f"[RUN] step={step}{progress}, event={event_file.name}")
    for tag in tags:
        if tag in values:
            value = values[tag]
            print(f"  {tag:38s} {value.value:.8g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="TensorBoard run directory")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="refresh repeatedly at this interval; omit for one snapshot",
    )
    args = parser.parse_args()

    if args.total_steps is not None and args.total_steps <= 0:
        parser.error("--total-steps must be positive")
    if args.watch < 0.0:
        parser.error("--watch must be non-negative")

    while True:
        try:
            print_snapshot(args.run_dir, args.total_steps, DEFAULT_TAGS)
        except FileNotFoundError as error:
            print(f"[WAIT] {error}")

        if args.watch == 0.0:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()

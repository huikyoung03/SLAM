#!/usr/bin/env python3
"""Compare one fixed-validation checkpoint against prior 2k checkpoints.

Exit status 42 is deliberate: it means the checkpoint was evaluated
successfully but training must not advance to the next segment.  Other
non-zero statuses are evaluation/contract errors rather than an early-stop
decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


STOP_EXIT_CODE = 42


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid or missing {key!r} in evaluation summary") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key!r} in evaluation summary")
    return value


def checkpoint_row(rows: list[dict[str, str]], checkpoint: str) -> dict[str, str]:
    candidates = [row for row in rows if row.get("label", "").startswith("learned_")]
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one learned checkpoint row in evaluation summary "
            f"for {checkpoint}, found {len(candidates)}"
        )
    return candidates[0]


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"format": "droid-early-stop-v1", "records": []}
    with path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("format") != "droid-early-stop-v1":
        raise ValueError(f"unexpected early-stop state format: {path}")
    if not isinstance(state.get("records"), list):
        raise ValueError(f"early-stop state lacks record list: {path}")
    return state


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-common", required=True, type=Path)
    parser.add_argument("--summary-full", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state-path", required=True, type=Path)
    parser.add_argument("--metric", default="sim3_ate_rmse_m")
    parser.add_argument("--min-coverage-ratio", type=float, default=0.95)
    parser.add_argument("--deterioration-tolerance-m", type=float, default=0.0)
    args = parser.parse_args()

    if not 0.0 < args.min_coverage_ratio <= 1.0:
        raise ValueError("--min-coverage-ratio must be in (0, 1]")
    if args.deterioration_tolerance_m < 0.0:
        raise ValueError("--deterioration-tolerance-m must be non-negative")

    common = checkpoint_row(read_rows(args.summary_common), args.checkpoint)
    full = checkpoint_row(read_rows(args.summary_full), args.checkpoint)
    metric = number(common, args.metric)
    coverage = number(full, "raw_n_poses")

    state_path = args.state_path.expanduser().resolve()
    state = load_state(state_path)
    records = state["records"]
    if any(record.get("checkpoint") == args.checkpoint for record in records):
        raise ValueError(f"checkpoint already exists in early-stop state: {args.checkpoint}")

    action = "continue"
    reason = "first_validation"
    if records:
        best_metric = min(float(record["metric"]) for record in records if record["action"] == "continue")
        reference_coverage = max(float(record["coverage"]) for record in records)
        required_coverage = args.min_coverage_ratio * reference_coverage
        if coverage < required_coverage:
            action = "stop"
            reason = (
                f"coverage_declined: current={coverage:.0f}, "
                f"required={required_coverage:.0f}"
            )
        elif metric > best_metric + args.deterioration_tolerance_m:
            action = "stop"
            reason = (
                f"metric_declined: current={metric:.9f}, "
                f"best={best_metric:.9f}, "
                f"tolerance={args.deterioration_tolerance_m:.9f}"
            )
        else:
            reason = "metric_not_worse_than_best"

    record = {
        "checkpoint": args.checkpoint,
        "metric_name": args.metric,
        "metric": metric,
        "coverage": coverage,
        "action": action,
        "reason": reason,
    }
    records.append(record)
    write_state(state_path, state)
    print(
        "[EARLY-STOP] "
        f"checkpoint={args.checkpoint}, {args.metric}={metric:.9f}, "
        f"coverage={coverage:.0f}, action={action}, reason={reason}"
    )
    return STOP_EXIT_CODE if action == "stop" else 0


if __name__ == "__main__":
    raise SystemExit(main())

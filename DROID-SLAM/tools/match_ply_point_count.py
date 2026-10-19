#!/usr/bin/env python3
"""Create a deterministic, equal-point-count PLY pair for visual comparison.

This tool intentionally changes only sampling density.  It does not transform,
align, scale, or crop either reconstruction, so geometry coordinates remain in
their source coordinate systems.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import open3d as o3d


def ply_vertex_count(path: Path) -> int:
    """Read the vertex count without loading a (potentially multi-GB) PLY."""
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            fields = line.decode("ascii", errors="strict").strip().split()
            if fields[:2] == ["element", "vertex"] and len(fields) == 3:
                return int(fields[2])
            if fields == ["end_header"]:
                break
    raise ValueError(f"Could not find an 'element vertex' header in {path}")


def sampled_cloud(path: Path, target_count: int, seed: int) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    actual_count = len(cloud.points)
    if actual_count < target_count:
        raise ValueError(f"{path} has {actual_count} points, less than requested {target_count}")
    if actual_count == target_count:
        return cloud

    indices = np.sort(
        np.random.default_rng(seed).choice(actual_count, size=target_count, replace=False)
    )
    sampled = o3d.geometry.PointCloud()
    sampled.points = o3d.utility.Vector3dVector(np.asarray(cloud.points)[indices])
    if cloud.has_colors():
        sampled.colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors)[indices])
    if cloud.has_normals():
        sampled.normals = o3d.utility.Vector3dVector(np.asarray(cloud.normals)[indices])
    return sampled


def output_path(output_dir: Path, source: Path, target_count: int) -> Path:
    return output_dir / f"{source.stem}_equal_{target_count}{source.suffix}"


def write_output(
    source: Path,
    destination: Path,
    source_count: int,
    target_count: int,
    seed: int,
    force: bool,
) -> str:
    if destination.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite {destination}; use --force")
        destination.unlink()

    # Preserve an already-matching source byte-for-byte and avoid a needless
    # second ~150 MB file. A hard link remains a normal readable PLY file.
    if source_count == target_count:
        os.link(source, destination)
        return "hardlink (already at target count)"

    cloud = sampled_cloud(source, target_count, seed)
    # Open3D determines the writer from the final extension, so keep `.ply`.
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    if not o3d.io.write_point_cloud(str(temporary), cloud, write_ascii=False, compressed=False):
        raise RuntimeError(f"Open3D failed to write {temporary}")
    os.replace(temporary, destination)
    return f"uniform random sample (seed={seed})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply_a", type=Path)
    parser.add_argument("ply_b", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=None,
                        help="Default: the smaller input point count")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    inputs = [args.ply_a.resolve(), args.ply_b.resolve()]
    for path in inputs:
        if not path.is_file():
            parser.error(f"Input PLY does not exist: {path}")
    if inputs[0] == inputs[1]:
        parser.error("Inputs must be two distinct PLY files")

    counts = [ply_vertex_count(path) for path in inputs]
    target_count = args.target_count if args.target_count is not None else min(counts)
    if target_count <= 0 or target_count > min(counts):
        parser.error(f"--target-count must be in [1, {min(counts)}], got {target_count}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [output_path(args.output_dir, path, target_count) for path in inputs]
    methods = []
    try:
        for source, destination, count in zip(inputs, destinations, counts):
            methods.append(write_output(source, destination, count, target_count, args.seed, args.force))
    except Exception:
        # Keep any successfully created sibling; it is valid and avoids silently
        # deleting a result the caller may already be inspecting.
        raise

    manifest = {
        "purpose": "Equal point-count PLY comparison; source coordinates unchanged.",
        "target_point_count": target_count,
        "seed": args.seed,
        "inputs": [
            {
                "source": str(source),
                "source_point_count": count,
                "output": str(destination),
                "method": method,
            }
            for source, count, destination, method in zip(inputs, counts, destinations, methods)
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for item in manifest["inputs"]:
        print(f"{item['output']}: {target_count} points ({item['method']})")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

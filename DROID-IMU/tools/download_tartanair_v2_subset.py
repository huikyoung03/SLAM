#!/usr/bin/env python3
"""Download a small TartanAir V2 subset and extract it with Python zipfile.

The official `tartanair` package shells out to the system `unzip` binary when
`unzip=True`. Some minimal Linux environments do not have that binary installed,
so this helper downloads zip files first and then extracts them using the Python
standard library.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def as_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def expected_zip_files(
    root: Path,
    envs: list[str],
    difficulties: list[str],
    modalities: list[str],
    camera_name: str,
) -> list[Path]:
    files: list[Path] = []

    for env in envs:
        for difficulty in difficulties:
            base = root / env / f"Data_{difficulty}"
            for modality in modalities:
                if modality in {"image", "depth", "seg"}:
                    files.append(base / f"{modality}_{camera_name}.zip")
                elif modality in {"imu", "lidar"}:
                    files.append(base / f"{modality}.zip")
                else:
                    raise ValueError(f"unsupported modality for this helper: {modality}")

    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/ubuntu/SLAM/datasets/tartanair_v2")
    parser.add_argument("--env", default="ArchVizTinyHouseNight")
    parser.add_argument("--difficulty", default="hard")
    parser.add_argument("--modality", default="image,depth,imu")
    parser.add_argument("--camera_name", default="lcam_front")
    parser.add_argument("--data_source", default="huggingface", choices=["huggingface", "airlab"])
    parser.add_argument("--delete_zip", action="store_true")
    args = parser.parse_args()

    import tartanair as ta

    root = Path(args.root)
    envs = as_list(args.env)
    difficulties = as_list(args.difficulty)
    modalities = as_list(args.modality)

    ta.init(str(root))
    ta.download(
        env=envs,
        difficulty=difficulties,
        modality=modalities,
        camera_name=[args.camera_name],
        unzip=False,
        delete_zip=False,
        num_workers=1,
        data_source=args.data_source,
    )

    for zip_path in expected_zip_files(root, envs, difficulties, modalities, args.camera_name):
        if not zip_path.is_file():
            print(f"[WARN] missing downloaded zip: {zip_path}")
            continue

        print(f"[EXTRACT] {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)

        if args.delete_zip:
            zip_path.unlink()

    print("[OK] TartanAir V2 subset ready")


if __name__ == "__main__":
    main()

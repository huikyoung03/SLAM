import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def q_normalize_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def q_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q_normalize_wxyz(q1)
    w2, x2, y2, z2 = q_normalize_wxyz(q2)

    return q_normalize_wxyz(np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64))


def q_to_rotvec_wxyz(q):
    q = q_normalize_wxyz(q)
    w, x, y, z = q
    w = float(np.clip(w, -1.0, 1.0))

    angle = 2.0 * math.acos(w)
    if angle > math.pi:
        angle -= 2.0 * math.pi

    s = math.sqrt(max(1.0 - w*w, 0.0))
    if s < 1e-12:
        return np.zeros(3, dtype=np.float64)

    axis = np.array([x, y, z], dtype=np.float64) / s
    return axis * angle


def find_image(row_filename, image_dir):
    """
    imu_prior.csv의 filename과 실제 image_dir 파일을 맞춘다.
    WebP를 우선 사용한다.
    """
    name = Path(str(row_filename)).name
    stem = Path(name).stem

    candidates = [
        image_dir / name,
        image_dir / f"{stem}.webp",
        image_dir / f"{stem}.WEBP",
        image_dir / f"{stem}.jpg",
        image_dir / f"{stem}.jpeg",
        image_dir / f"{stem}.png",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def compose_prior_segment(df, prev_orig_idx, cur_orig_idx):
    """
    선택된 프레임 사이에 포함된 기존 imu_prior row들을 quaternion으로 누적한다.
    기존 row는 frame i-1 -> frame i 구간의 IMU delta라고 가정한다.
    """
    if cur_orig_idx <= prev_orig_idx:
        return {
            "dt": 0.0,
            "imu_count": 0,
            "dq": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "imu_valid": 0,
            "imu_weight": 0.0,
            "invalid_reason": "zero_interval",
        }

    seg = df[(df["frame_index"] > prev_orig_idx) & (df["frame_index"] <= cur_orig_idx)].copy()

    if len(seg) == 0:
        return {
            "dt": 0.0,
            "imu_count": 0,
            "dq": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "imu_valid": 0,
            "imu_weight": 0.0,
            "invalid_reason": "empty_segment",
        }

    q_total = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    valid_all = True

    for _, r in seg.iterrows():
        q = np.array([
            float(r.get("dq_w", 1.0)),
            float(r.get("dq_x", 0.0)),
            float(r.get("dq_y", 0.0)),
            float(r.get("dq_z", 0.0)),
        ], dtype=np.float64)

        if not np.all(np.isfinite(q)):
            valid_all = False
            continue

        q_total = q_mul_wxyz(q_total, q)

        if "imu_valid" in r and int(r["imu_valid"]) == 0:
            valid_all = False

    dt = float(seg["dt"].sum()) if "dt" in seg.columns else 0.0
    imu_count = int(seg["imu_count"].sum()) if "imu_count" in seg.columns else len(seg)

    if "imu_weight" in seg.columns:
        weights = seg["imu_weight"].dropna().astype(float).values
        imu_weight = float(np.clip(np.nanmean(weights) if len(weights) else 1.0, 0.0, 1.0))
    else:
        imu_weight = 1.0

    return {
        "dt": dt,
        "imu_count": imu_count,
        "dq": q_normalize_wxyz(q_total),
        "imu_valid": 1 if valid_all and imu_count > 0 else 0,
        "imu_weight": imu_weight,
        "invalid_reason": "ok" if valid_all and imu_count > 0 else "invalid_segment",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_stride", type=int, default=3)
    parser.add_argument("--turn_percentile", type=float, default=70.0)
    parser.add_argument("--extreme_percentile", type=float, default=98.0)
    parser.add_argument("--drop_extreme", action="store_true")
    parser.add_argument("--min_keep", type=int, default=40)
    args = parser.parse_args()

    session = Path(args.session)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_img = out_dir / "images"

    out_img.mkdir(parents=True, exist_ok=True)

    prior_path = session / "imu_prior.csv"
    df = pd.read_csv(prior_path)

    if "frame_index" not in df.columns:
        df["frame_index"] = df["frame_id"].astype(int)

    df = df.sort_values("frame_index").reset_index(drop=True)

    dr = df["dr_norm_deg"].fillna(0).astype(float).values
    valid_dr = dr[np.isfinite(dr)]

    turn_th = float(np.percentile(valid_dr, args.turn_percentile)) if len(valid_dr) else 5.0
    extreme_th = float(np.percentile(valid_dr, args.extreme_percentile)) if len(valid_dr) else 25.0

    keep = set()

    if len(df) > 0:
        keep.add(int(df.iloc[0]["frame_index"]))
        keep.add(int(df.iloc[-1]["frame_index"]))

    for _, r in df.iterrows():
        ix = int(r["frame_index"])
        deg = float(r.get("dr_norm_deg", 0.0))
        imu_valid = int(r.get("imu_valid", 1))

        if imu_valid == 0:
            continue

        # 직선 구간은 기본 stride로 선택
        if ix % args.base_stride == 0:
            keep.add(ix)

        # 회전량 큰 구간은 프레임을 촘촘히 유지
        if deg >= turn_th:
            keep.add(ix)
            keep.add(max(0, ix - 1))
            keep.add(ix + 1)

    if args.drop_extreme:
        drop = set()
        for _, r in df.iterrows():
            ix = int(r["frame_index"])
            deg = float(r.get("dr_norm_deg", 0.0))
            if deg >= extreme_th:
                drop.add(ix)
                keep.add(max(0, ix - 1))
                keep.add(ix + 1)
        keep = keep - drop

    selected = []
    for ix in sorted(keep):
        rows = df[df["frame_index"] == ix]
        if len(rows) == 0:
            continue

        row = rows.iloc[0]
        src = find_image(row["filename"], image_dir)

        if src is None:
            continue

        selected.append((ix, row, src))

    if len(selected) < args.min_keep:
        print("[WARN] selected frames too few. fallback to base stride only.")
        selected = []
        for _, row in df.iterrows():
            ix = int(row["frame_index"])
            if ix % args.base_stride == 0 or ix == int(df.iloc[0]["frame_index"]) or ix == int(df.iloc[-1]["frame_index"]):
                src = find_image(row["filename"], image_dir)
                if src is not None:
                    selected.append((ix, row, src))

    mapping_rows = []
    prior_rows = []
    prev_orig_ix = None

    for new_ix, (orig_ix, row, src) in enumerate(selected):
        # WebP 그대로 유지. DROID가 WebP 사용 가능하다고 했으므로 변환하지 않음.
        new_name = f"{new_ix:06d}.webp"
        dst = out_img / new_name
        shutil.copy2(src, dst)

        mapping_rows.append({
            "new_frame_index": new_ix,
            "orig_frame_index": orig_ix,
            "src": str(src),
            "filename": f"images/{new_name}",
            "timestamp_sec": float(row.get("timestamp_sec", np.nan)),
            "dr_norm_deg_orig": float(row.get("dr_norm_deg", 0.0)),
        })

        if prev_orig_ix is None:
            seg_info = {
                "dt": 0.0,
                "imu_count": 0,
                "dq": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                "imu_valid": 0,
                "imu_weight": 0.0,
                "invalid_reason": "first_frame",
            }
        else:
            seg_info = compose_prior_segment(df, prev_orig_ix, orig_ix)

        dq = seg_info["dq"]
        rv = q_to_rotvec_wxyz(dq)
        dr_norm = float(np.linalg.norm(rv))
        dr_norm_deg = float(dr_norm * 180.0 / math.pi)

        prior_rows.append({
            "frame_id": new_ix,
            "frame_index": new_ix,
            "timestamp_sec": float(row.get("timestamp_sec", np.nan)),
            "timestamp_ns": row.get("timestamp_ns", np.nan),
            "filename": f"images/{new_name}",
            "dt": seg_info["dt"],
            "imu_count": seg_info["imu_count"],
            "dr_x": rv[0],
            "dr_y": rv[1],
            "dr_z": rv[2],
            "dr_norm": dr_norm,
            "dr_norm_deg": dr_norm_deg,
            "dq_w": dq[0],
            "dq_x": dq[1],
            "dq_y": dq[2],
            "dq_z": dq[3],
            "imu_valid": seg_info["imu_valid"],
            "imu_weight": seg_info["imu_weight"],
            "invalid_reason": seg_info["invalid_reason"],
            "orig_frame_index": orig_ix,
        })

        prev_orig_ix = orig_ix

    mapping = pd.DataFrame(mapping_rows)
    prior_new = pd.DataFrame(prior_rows)

    mapping.to_csv(out_dir / "selected_mapping.csv", index=False)
    prior_new.to_csv(out_dir / "imu_prior.csv", index=False)

    timestamps = mapping[["new_frame_index", "timestamp_sec", "filename", "orig_frame_index"]].copy()
    timestamps.to_csv(out_dir / "timestamps.csv", index=False)

    if (session / "calib.txt").exists():
        shutil.copy2(session / "calib.txt", out_dir / "calib.txt")

    print("=== IMU SELECTED WEBP SESSION CREATED ===")
    print("session:", session)
    print("image_dir:", image_dir)
    print("out_dir:", out_dir)
    print("selected frames:", len(selected), "/", len(df))
    print("turn_th:", turn_th)
    print("extreme_th:", extreme_th)
    print("drop_extreme:", args.drop_extreme)
    print("output images:", out_img)
    print("output prior:", out_dir / "imu_prior.csv")
    print("mapping:", out_dir / "selected_mapping.csv")
    print()
    print(mapping.head())
    print()
    print(mapping.tail())


if __name__ == "__main__":
    main()

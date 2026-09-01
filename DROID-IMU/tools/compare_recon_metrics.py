import argparse
import os
import torch
import numpy as np


def q_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, 1e-12, None)


def quat_angle_deg(q1, q2):
    q1 = q_normalize(q1)
    q2 = q_normalize(q2)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot) * 180.0 / np.pi


def pairwise_nn(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.nan, np.nan, np.nan

    # small trajectory라서 numpy brute-force로 충분함
    dists = []
    for a in A:
        d = np.linalg.norm(B - a[None, :], axis=1)
        dists.append(np.min(d))
    dists = np.asarray(dists)

    return float(np.mean(dists)), float(np.median(dists)), float(np.percentile(dists, 90))


def load_recon(path):
    blob = torch.load(path, map_location="cpu")

    poses = blob["poses"].detach().cpu().float().numpy()

    if "tstamps" in blob:
        tstamps = blob["tstamps"].detach().cpu().numpy()
    else:
        tstamps = np.arange(len(poses))

    valid = np.isfinite(poses).all(axis=1)
    poses = poses[valid]
    tstamps = tstamps[valid]

    t = poses[:, 0:3]
    q = poses[:, 3:7]

    return tstamps, t, q


def analyze(path, turn_frame):
    tstamps, t, q = load_recon(path)

    if len(t) < 5:
        return None

    trans_step = np.linalg.norm(np.diff(t, axis=0), axis=1)
    rot_step = quat_angle_deg(q[:-1], q[1:])

    path_len = float(np.sum(trans_step))
    start_end = float(np.linalg.norm(t[-1] - t[0]))
    bbox = np.ptp(t, axis=0)
    bbox_diag = float(np.linalg.norm(bbox))

    # turn frame 근처 찾기
    turn_idx = int(np.argmin(np.abs(tstamps - turn_frame)))

    # 회전 전/후 trajectory가 서로 얼마나 가까운지
    # out-and-back이면 이 값이 작을수록 좋음
    before = t[:max(0, turn_idx - 5)]
    after = t[min(len(t), turn_idx + 5):]

    nn_mean, nn_median, nn_p90 = pairwise_nn(after, before)

    # 회전 구간 주변 pose가 얼마나 튀는지
    a = max(0, turn_idx - 10)
    b = min(len(t), turn_idx + 10)
    turn_local = t[a:b]
    turn_spread = float(np.linalg.norm(np.ptp(turn_local, axis=0))) if len(turn_local) > 1 else np.nan

    # path_len으로 정규화한 gap
    norm_gap = nn_median / path_len if path_len > 1e-12 else np.nan
    norm_start_end = start_end / path_len if path_len > 1e-12 else np.nan

    return {
        "file": os.path.basename(path),
        "n_poses": len(t),
        "path_len": path_len,
        "start_end": start_end,
        "start_end/path": norm_start_end,
        "bbox_diag": bbox_diag,
        "step_mean": float(np.mean(trans_step)),
        "step_p95": float(np.percentile(trans_step, 95)),
        "step_max": float(np.max(trans_step)),
        "rot_mean_deg": float(np.mean(rot_step)),
        "rot_p95_deg": float(np.percentile(rot_step, 95)),
        "rot_max_deg": float(np.max(rot_step)),
        "turn_frame": turn_frame,
        "turn_idx": turn_idx,
        "turn_spread": turn_spread,
        "return_gap_mean": nn_mean,
        "return_gap_median": nn_median,
        "return_gap_p90": nn_p90,
        "return_gap_median/path": norm_gap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn_frame", type=int, default=223)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    rows = []
    for f in args.files:
        r = analyze(f, args.turn_frame)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No valid reconstructions.")
        return

    keys = [
        "file",
        "n_poses",
        "path_len",
        "start_end/path",
        "step_p95",
        "rot_p95_deg",
        "turn_spread",
        "return_gap_median/path",
        "return_gap_p90",
    ]

    print("\n=== Reconstruction trajectory metrics ===")
    print("Lower is generally better for: start_end/path, step_p95, rot_p95_deg, turn_spread, return_gap_median/path")
    print()

    header = " | ".join(keys)
    print(header)
    print("-" * len(header))

    for r in rows:
        vals = []
        for k in keys:
            v = r[k]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        print(" | ".join(vals))


if __name__ == "__main__":
    main()

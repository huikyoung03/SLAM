import math


def apply_learned_imu_ba_weight(args, net, source="runtime"):
    """Apply the checkpoint-learned global IMU BA weight to runtime args."""

    if not getattr(args, "use_learned_imu_ba_weight", False):
        return None

    loaded_keys = getattr(net, "_loaded_state_keys", None)
    if loaded_keys is not None and "imu_ba_log_weight" not in loaded_keys:
        raise ValueError(
            "--use_learned_imu_ba_weight was requested, but this checkpoint "
            "does not contain imu_ba_log_weight. Use an IMU-trained checkpoint."
        )

    if not hasattr(net, "get_imu_ba_weight"):
        raise ValueError(
            "--use_learned_imu_ba_weight requires a DroidNet with get_imu_ba_weight()."
        )

    learned = float(net.get_imu_ba_weight().detach().cpu().item())
    scale = float(getattr(args, "learned_imu_ba_weight_scale", 1.0))

    if not math.isfinite(learned) or learned < 0.0:
        raise ValueError(f"invalid learned IMU BA weight: {learned}")
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError(f"invalid learned IMU BA weight scale: {scale}")

    old_weight = float(getattr(args, "imu_ba_prior_weight", 0.0))
    runtime_weight = learned * scale
    args.imu_ba_prior_weight = runtime_weight

    print(
        "[IMU] using learned global IMU BA weight "
        f"source={source}, learned={learned:.6g}, scale={scale:.6g}, "
        f"runtime={runtime_weight:.6g}, previous_cli={old_weight:.6g}"
    )

    if not getattr(args, "use_imu_ba_prior", False):
        print(
            "[IMU WARNING] learned IMU BA weight has no effect without "
            "--use_imu_ba_prior or --use_full_imu_ba."
        )

    return runtime_weight

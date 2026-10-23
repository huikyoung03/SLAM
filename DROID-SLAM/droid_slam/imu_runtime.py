import math


def configure_learned_imu_precision_weighting(args, source="runtime"):
    """Configure learned R/v/p IMU attenuation without a global lambda.

    Each head output lies in ``(0, 1]`` and scales a covariance-normalized
    physical preintegration block. It may reject unreliable information but
    cannot strengthen an interval beyond the sensor covariance. It must not
    be combined with a second global learned/tuned multiplier or historical
    hand-set cross-block coefficients.
    """

    if not getattr(args, "use_learned_imu_precision", False):
        return False
    if not getattr(args, "use_full_imu_ba", False):
        raise ValueError("--use_learned_imu_precision requires --use_full_imu_ba")
    if not getattr(args, "use_imu_full_covariance_weighting", False):
        raise ValueError(
            "--use_learned_imu_precision requires "
            "--use_imu_full_covariance_weighting so R/v/p share a physical base"
        )
    if getattr(args, "use_learned_imu_ba_weight", False):
        raise ValueError(
            "--use_learned_imu_precision cannot be combined with "
            "--use_learned_imu_ba_weight"
        )
    if getattr(args, "use_learned_imu_residual_weights", False):
        raise ValueError(
            "--use_learned_imu_precision cannot be combined with "
            "--use_learned_imu_residual_weights"
        )
    if getattr(args, "imu_confidence_only_weighting", False):
        raise ValueError(
            "--use_learned_imu_precision cannot be combined with "
            "--imu_confidence_only_weighting"
        )
    if getattr(args, "imu_tuned_confidence_schedule", None) is not None:
        raise ValueError(
            "--use_learned_imu_precision cannot be combined with "
            "--imu_tuned_confidence_schedule"
        )

    old_weight = float(getattr(args, "imu_ba_prior_weight", 0.0))
    args.imu_ba_prior_weight = 1.0
    args.use_adaptive_imu_gate = False
    if hasattr(args, "imu_confidence_floor"):
        args.imu_confidence_floor = 0.0
    if hasattr(args, "imu_conf_floor"):
        args.imu_conf_floor = 0.0
    print(
        "[IMU] learned R/v/p covariance-relative attenuation "
        f"source={source}, global_weight=1, previous_cli={old_weight:.6g}, "
        "legacy_confidence_gate=disabled"
    )
    return True


def configure_confidence_only_imu_weighting(args, source="runtime"):
    """Use learned IMU confidence as the sole adaptive factor attenuation.

    ``imu_ba_prior_weight`` historically mixed two separate concerns: the
    visual-vs-IMU global scale and per-interval IMU reliability.  This mode
    fixes the former at one.  The preintegration covariance supplies the
    physical relative information and the temporal GRU confidence is the only
    learned multiplier that may reduce an individual IMU factor.
    """

    if not getattr(args, "imu_confidence_only_weighting", False):
        return False

    if getattr(args, "use_learned_imu_ba_weight", False):
        raise ValueError(
            "--imu_confidence_only_weighting cannot be combined with "
            "--use_learned_imu_ba_weight"
        )
    if getattr(args, "imu_tuned_confidence_schedule", None) is not None:
        raise ValueError(
            "--imu_confidence_only_weighting cannot be combined with "
            "--imu_tuned_confidence_schedule"
        )

    old_weight = float(getattr(args, "imu_ba_prior_weight", 0.0))
    args.imu_ba_prior_weight = 1.0

    # A learned confidence must be the only quality gate in this mode.  These
    # legacy mechanisms multiply the same factor a second time and obscure
    # what the confidence head learned.
    had_adaptive_gate = bool(getattr(args, "use_adaptive_imu_gate", False))
    args.use_adaptive_imu_gate = False
    if hasattr(args, "imu_confidence_floor"):
        args.imu_confidence_floor = 0.0
    if hasattr(args, "imu_conf_floor"):
        args.imu_conf_floor = 0.0

    print(
        "[IMU] confidence-only weighting "
        f"source={source}, global_weight=1, previous_cli={old_weight:.6g}, "
        "adaptive_gate=disabled, confidence_floor=disabled"
    )
    if had_adaptive_gate:
        print(
            "[IMU] disabled legacy adaptive IMU gate because learned "
            "confidence is the sole adaptive attenuation"
        )
    return True


def configure_imu_noise_model(args, source="runtime"):
    """Select covariance-relative weighting without an arbitrary info clip.

    Preintegration provides diagonal uncertainty metadata.  Its absolute scale
    cannot be directly combined with DROID's visual covariance, but the
    relative change between intervals is meaningful.  This mode keeps that
    relative information and lets the learned global/category weights set the
    cross-modal scale. The GRU confidence remains the only adaptive multiplier;
    this mode deliberately removes the legacy threshold-based heuristic gate
    and a hand-set confidence floor.
    """

    if not getattr(args, "use_imu_noise_model", False):
        return False
    if getattr(args, "_imu_noise_model_configured", False):
        return True

    args.use_imu_info_weighting = True
    args.imu_info_weight_clip = 0.0

    # A residual magnitude is not an IMU-quality measurement, so leave those
    # old heuristic gates disabled.  Duration is different: the present
    # keyframe-only factor carries one first-order bias linearization and only
    # a diagonal covariance.  It cannot faithfully represent an arbitrary
    # MotionFilter gap (observed gaps reach 7--10 s).  Preserve the caller's
    # physical maximum duration, whose parser default is 0.5 s, instead of
    # silently replacing it with zero.  A caller may explicitly set zero only
    # when it supplies a model with intervening inertial states/full covariance.
    args.imu_ba_prior_max_deg = 0.0
    args.imu_full_max_dv = 0.0
    args.imu_full_max_dp = 0.0
    # Training-side PyTorch DBA/full loss used this as a hard rotational
    # residual gate. A residual magnitude is not an IMU quality measurement;
    # in covariance mode it must remain available to the learned confidence and
    # visual objective instead of being removed at a hand-tuned angle.
    if hasattr(args, "imu_max_residual"):
        args.imu_max_residual = 0.0

    # These settings belong to the old heuristic gate rather than to the
    # preintegration noise model. Keeping either one active would mix a learned
    # confidence with a second, hand-tuned confidence mechanism.
    legacy_gate_enabled = bool(getattr(args, "use_adaptive_imu_gate", False))
    confidence_floor = float(getattr(args, "imu_confidence_floor", 0.0))
    args.use_adaptive_imu_gate = False
    if hasattr(args, "imu_confidence_floor"):
        args.imu_confidence_floor = 0.0
    if hasattr(args, "imu_conf_floor"):
        args.imu_conf_floor = 0.0

    args._imu_noise_model_configured = True
    max_dt = float(getattr(args, "imu_full_max_dt", 0.5))
    print(
        "[IMU] using covariance-relative IMU information "
        f"source={source}, info_clip=disabled, max_dt={max_dt:.6g}, "
        "dv_dp_residual_gates=disabled, adaptive_gate=disabled, "
        "confidence_floor=disabled"
    )
    if legacy_gate_enabled:
        print(
            "[IMU] disabled legacy adaptive IMU gate because "
            "--use_imu_noise_model uses GRU confidence directly."
        )
    if confidence_floor > 0.0:
        print(
            "[IMU] disabled --imu_confidence_floor because covariance mode "
            "must preserve the learned confidence value."
        )
    return True


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


def apply_learned_imu_residual_weights(args, net, source="runtime"):
    """Apply checkpoint-learned relative position/velocity IMU weights."""

    if not getattr(args, "use_learned_imu_residual_weights", False):
        return None

    loaded_keys = getattr(net, "_loaded_state_keys", None)
    required = {"imu_pos_log_weight", "imu_vel_log_weight"}
    if loaded_keys is not None and not required.issubset(loaded_keys):
        missing = ", ".join(sorted(required - set(loaded_keys)))
        raise ValueError(
            "--use_learned_imu_residual_weights was requested, but this checkpoint "
            f"does not contain {missing}. Fine-tune with learned residual weights first."
        )
    if not hasattr(net, "get_imu_full_residual_weights"):
        raise ValueError(
            "--use_learned_imu_residual_weights requires a DroidNet with "
            "get_imu_full_residual_weights()."
        )

    learned = net.get_imu_full_residual_weights().detach().cpu().tolist()
    if len(learned) != 2 or not all(math.isfinite(float(value)) and value >= 0.0 for value in learned):
        raise ValueError(f"invalid learned IMU residual weights: {learned}")

    old_pos = float(getattr(args, "imu_full_pos_weight", 0.0))
    old_vel = float(getattr(args, "imu_full_vel_weight", 0.0))
    args.imu_full_pos_weight = float(learned[0])
    args.imu_full_vel_weight = float(learned[1])
    print(
        "[IMU] using learned relative IMU residual weights "
        f"source={source}, pos={learned[0]:.6g}, vel={learned[1]:.6g}, "
        f"previous_cli=({old_pos:.6g}, {old_vel:.6g})"
    )
    return tuple(float(value) for value in learned)

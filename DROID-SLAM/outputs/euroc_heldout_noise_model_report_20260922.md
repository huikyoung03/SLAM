# 9DoF IMU Noise-Model Held-Out Validation

## Scope

The compared runs use the same frozen visual checkpoint and the same fixed-bias
pose-velocity 9DoF runtime. Visual-inertial initialization estimates the
initial scale, gravity, and global bias; window-level bias refinement is
disabled afterwards. Each sequence uses the first 450 input images with
`stride=2`. ATE is Sim(3)-aligned, so it measures reconstruction agreement but
does not by itself prove metric-scale accuracy.

The covariance-mode runs use raw-camera-rate information references, compose
the preintegration covariance across skipped frames, and disable the legacy
rotation/position/velocity residual limits and information clip. The IMU factor
is active on nearly every valid temporal edge.

## Results

| held-out sequence | visual-only ATE RMSE (m) | 9DoF covariance-mode ATE RMSE (m) | relative change | IMU edge use |
| --- | ---: | ---: | ---: | ---: |
| V2_01_easy | 0.030052 | 0.030048 | +0.01% | 2931 / 2931 |
| V2_03_difficult | 0.048491 | 0.048599 | -0.22% | 11155 / 11174 |

## Interpretation

The covariance composition correction is required for physical consistency:
for a composed interval of duration `T`, this implementation now uses rotation
and velocity variance proportional to `T`, and position variance proportional
to `T^3`. The prior implementation added position variance linearly, which
overstated position information for skipped-frame intervals.

The current IMU checkpoint does not produce a consistent held-out ATE gain.
V2_01 is effectively neutral and V2_03 is slightly worse. This means that any
earlier V1_03 improvement must be treated as sequence-specific until a newly
trained checkpoint succeeds on held-out sequences.

The learned IMU confidence is still high in both held-out runs: median 0.876
for V2_01 and 0.898 for V2_03, with the 95th percentile at the sigmoid ceiling.
It is connected to the DBA factor, but the old checkpoint was trained before
the corrected covariance/temporal-stride protocol and is not suitably
calibrated for this noise-model evaluation. It also predates the checkpointed
position/velocity weights, so these two held-out runs intentionally retain the
legacy 0.05 values only as a structural baseline, not as a final model result.

The available TartanAir V2 training root currently contains only 500 images
across seven short trajectories from one environment. It is enough for a
structural smoke test, but not enough evidence for a long fine-tuning run or a
generalization claim. The launcher therefore defaults to 2000 steps and saves
intermediate checkpoints every 500 steps; use held-out EuRoC evaluation to pick
the checkpoint instead of assuming the final step is best.

## Next Checkpoint Protocol

`tools/run_9dof_noise_finetune.sh` is the replacement fine-tuning launcher.
It resets only the final confidence projection to 0.5, keeps gravity and both
global biases fixed, samples frame spacings 1/2/4/8, uses configured IMU noise
densities, and learns the global plus relative position/velocity factor
weights. The old threshold gate, confidence floor, and information clip are
not part of the launcher.

After that checkpoint is trained, rerun both held-out sequences with
`USE_NOISE_MODEL=1`, `USE_LEGACY_ADAPTIVE_GATE=0`, and
`USE_LEARNED_IMU_RESIDUAL_WEIGHTS=1`. Accept the change only if it improves the
mean held-out ATE without a large regression on either sequence.

A two-step smoke checkpoint verified that the new `imu_pos_log_weight` and
`imu_vel_log_weight` fields are saved and replace runtime CLI values when
`USE_LEARNED_IMU_RESIDUAL_WEIGHTS=1`. It is only an integration test, not a
candidate model.

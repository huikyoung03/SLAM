# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 313 | 313 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.013653 | 0.015055 | 0.030638 |  |  |  |  |  |  |  | inv_center | 0.028025 | 20.355619 | 0.025096 | 0 |  |  |  |
| fixed_15d | 322 | 322 | imu_alignment_unit_scale | 0.578105 | 1 | live | 41 | 0 | 0 | 0.015431 | 0.014561 | 0.031467 | 0.350117 | 0.364449 | 0.207457 | 0.150925 | 0.460213 | 0.342891 | 0.726489 | inv_center | 0.028311 | 20.122183 | 0.028335 | 13648 | 0.999122 | 1.000000 | 0.002000 |
| tuned_15d | 322 | 322 | imu_alignment_unit_scale | 0.575078 | 1 | live | 41 | 0 | 0 | 0.015518 | 0.014498 | 0.031559 | 0.337560 | 0.351154 | 0.198587 | 0.145211 | 0.443680 | 0.340971 | 0.725803 | inv_center | 0.028323 | 20.121893 | 0.028351 | 13598 | 0.999118 | 0.516197 | 0.002000 |
| learned_old | 323 | 323 | imu_alignment_unit_scale | 0.581447 | 1 | live | 41 | 0 | 0 | 0.015403 | 0.014527 | 0.031419 | 0.364429 | 0.379654 | 0.218000 | 0.156618 | 0.478733 | 0.340579 | 0.726281 | inv_center | 0.028314 | 20.289335 | 0.028319 | 13470 | 0.999110 | 0.931566 | 0.002000 |
| learned_parity_2k | 322 | 322 | imu_alignment_unit_scale | 0.575205 | 1 | live | 41 | 0 | 0 | 0.015680 | 0.014636 | 0.031670 | 0.338157 | 0.351765 | 0.199004 | 0.145547 | 0.444550 | 0.342482 | 0.728110 | inv_center | 0.028330 | 20.121565 | 0.028348 | 13598 | 0.999118 | 0.854848 | 0.002000 |

Lowest metric SE(3) ATE: `tuned_15d` = 0.337560 m

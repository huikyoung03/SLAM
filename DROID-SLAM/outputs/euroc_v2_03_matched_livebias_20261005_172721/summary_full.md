# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 271 | 271 | monocular_scale_unobservable |  |  | unknown_nonmetric | 0 | 0 | 0 | 0.015015 | 0.016007 | 0.031525 |  |  |  |  |  |  |  | inv_center | 0.028378 | 20.588682 | 0.027932 | 0 |  |  |  |
| finetuned_visual_only | 271 | 271 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.015015 | 0.016007 | 0.031525 |  |  |  |  |  |  |  | inv_center | 0.028378 | 20.588682 | 0.027932 | 0 |  |  |  |
| imu_9dof_fixed | 324 | 324 | imu_alignment_unit_scale | 0.609494 |  | live | 41 | 0 | 0 | 0.014614 | 0.014290 | 0.030379 | 0.345422 | 0.360192 | 0.204391 | 0.150006 | 0.452673 | 0.341717 | 0.720223 | inv_center | 0.028324 | 20.281693 | 0.028309 | 0 |  |  |  |
| imu_9dof_lowrate_bias | 322 | 322 | imu_alignment_unit_scale | 0.609297 |  | live | 41 | 36 | 28 | 0.015063 | 0.014399 | 0.030420 | 0.337646 | 0.352439 | 0.199454 | 0.145043 | 0.442926 | 0.345736 | 0.719623 | inv_center | 0.028331 | 20.298765 | 0.028313 | 26080 | 0.999272 | 0.899941 | 0.053401 |

Lowest metric SE(3) ATE: `imu_9dof_lowrate_bias` = 0.337646 m

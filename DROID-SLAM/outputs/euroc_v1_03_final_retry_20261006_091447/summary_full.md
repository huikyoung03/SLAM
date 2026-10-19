# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 234 | 234 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.017735 | 0.028929 | 0.065636 |  |  |  |  |  |  |  | inv_center | 0.005676 | 19.552707 | 0.031189 | 0 |  |  |  |
| finetuned_visual_only | 234 | 234 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.017735 | 0.028929 | 0.065636 |  |  |  |  |  |  |  | inv_center | 0.005676 | 19.552707 | 0.031189 | 0 |  |  |  |
| imu_9dof_fixed | 289 | 289 | imu_alignment_unit_scale | 0.801499 | 1 | live | 65 | 0 | 0 | 0.017136 | 0.026131 | 0.062085 | 0.030663 | 0.089200 | 0.016846 | 0.028882 | 0.068698 | 0.232346 | 0.490706 | inv_center | 0.005588 | 19.082586 | 0.031110 | 25472 | 1.000000 | 0.913120 | 0.054184 |
| imu_9dof_lowrate_bias | 289 | 289 | imu_alignment_unit_scale | 0.801807 | 1 | live | 65 | 37 | 22 | 0.017148 | 0.026126 | 0.062068 | 0.030525 | 0.088911 | 0.016732 | 0.028835 | 0.068601 | 0.232176 | 0.490855 | inv_center | 0.005588 | 19.082533 | 0.031112 | 25472 | 1.000000 | 0.913016 | 0.054177 |

Lowest metric SE(3) ATE: `imu_9dof_lowrate_bias` = 0.030525 m

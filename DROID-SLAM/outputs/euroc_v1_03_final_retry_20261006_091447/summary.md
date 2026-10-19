# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 205 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 234 | 205 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.017757 | 0.028005 | 0.051294 |  |  |  |  |  |  |  | inv_center | 0.006115 | 20.008655 |  | 0 |  |  |  |
| finetuned_visual_only | 234 | 205 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.017757 | 0.028005 | 0.051294 |  |  |  |  |  |  |  | inv_center | 0.006115 | 20.008655 |  | 0 |  |  |  |
| imu_9dof_fixed | 289 | 205 | imu_alignment_unit_scale | 0.801499 | 1 | live | 65 | 0 | 0 | 0.017674 | 0.028062 | 0.051392 | 0.025934 | 0.104101 | 0.014032 | 0.030794 | 0.056501 | 0.260620 | 0.517645 | inv_center | 0.006083 | 20.012283 |  | 25472 | 1.000000 | 0.913120 | 0.054184 |
| imu_9dof_lowrate_bias | 289 | 205 | imu_alignment_unit_scale | 0.801807 | 1 | live | 65 | 37 | 22 | 0.017665 | 0.028055 | 0.051365 | 0.025795 | 0.103796 | 0.013899 | 0.030736 | 0.056398 | 0.260330 | 0.517847 | inv_center | 0.006084 | 20.010246 |  | 25472 | 1.000000 | 0.913016 | 0.054177 |

Lowest metric SE(3) ATE: `imu_9dof_lowrate_bias` = 0.025795 m

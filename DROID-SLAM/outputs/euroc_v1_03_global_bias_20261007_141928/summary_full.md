# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 98 | 98 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.015327 | 0.035534 | 0.096233 |  |  |  |  |  |  |  | inv_center | 0.131153 | 17.943485 | 0.081008 | 0 |  |  |  |
| visual_bootstrap_control | 134 | 134 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.013399 | 0.041539 | 0.099578 |  |  |  |  |  |  |  | inv_center | 0.125559 | 17.407338 | 0.087080 | 0 |  |  |  |
| two_stage_15dof_global_bias | 154 | 154 | imu_alignment_unit_scale | 0.754436 | 1 | live | 40 | 2 | 2 | 0.015417 | 0.028647 | 0.082981 | 0.021984 | 0.083303 | 0.009610 | 0.029110 | 0.085080 | 0.204385 | 0.490604 | inv_center | 0.128447 | 17.092927 | 0.080412 | 5476 | 0.886945 | 1.000000 | 0.002000 |

Lowest metric SE(3) ATE: `two_stage_15dof_global_bias` = 0.021984 m

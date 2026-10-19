# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 41 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 98 | 41 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.009562 | 0.043747 | 0.100463 |  |  |  |  |  |  |  | inv_center | 0.196077 | 15.510237 |  | 0 |  |  |  |
| visual_bootstrap_control | 134 | 41 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.008096 | 0.042472 | 0.096346 |  |  |  |  |  |  |  | inv_center | 0.195367 | 15.525750 |  | 0 |  |  |  |
| two_stage_15dof_global_bias | 154 | 41 | imu_alignment_unit_scale | 0.754436 | 1 | live | 40 | 2 | 2 | 0.009301 | 0.043564 | 0.100479 | 0.015676 | 0.116764 | 0.007090 | 0.044269 | 0.102641 | 0.269458 | 0.646747 | inv_center | 0.195750 | 15.491495 |  | 5476 | 0.886945 | 1.000000 | 0.002000 |

Lowest metric SE(3) ATE: `two_stage_15dof_global_bias` = 0.015676 m

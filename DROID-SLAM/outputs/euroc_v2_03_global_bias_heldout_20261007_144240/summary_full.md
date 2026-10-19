# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 120 | 120 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.016978 | 0.016093 | 0.028120 |  |  |  |  |  |  |  | inv_center | 0.077841 | 19.515377 | 0.071384 | 0 |  |  |  |
| visual_bootstrap_control | 161 | 161 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.016211 | 0.014287 | 0.029731 |  |  |  |  |  |  |  | inv_center | 0.075917 | 18.265894 | 0.062621 | 0 |  |  |  |
| two_stage_15dof_global_bias | 171 | 171 | imu_alignment_unit_scale | 0.610335 | 1 | live | 41 | 2 | 2 | 0.017772 | 0.013390 | 0.026307 | 0.187555 | 0.197449 | 0.103134 | 0.082901 | 0.260441 | 0.240858 | 0.525578 | inv_center | 0.077623 | 18.178302 | 0.075420 | 5622 | 0.797673 | 1.000000 | 0.002000 |

Lowest metric SE(3) ATE: `two_stage_15dof_global_bias` = 0.187555 m

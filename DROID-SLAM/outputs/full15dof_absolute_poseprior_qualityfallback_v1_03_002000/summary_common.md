# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 203 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 234 | 203 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.017711 | 0.028053 | 0.051065 |  |  |  |  |  |  |  | inv_center | 0.006140 | 19.636998 |  | 0 |  |  |  |
| learned_15dof_confidence | 290 | 203 | imu_alignment_unit_scale | 0.552517 | 1 | live | 40 | 0 | 0 | 0.017740 | 0.027956 | 0.050838 | 0.019119 | 0.094625 | 0.005332 | 0.028419 | 0.052019 | 0.260118 | 0.516047 | inv_center | 0.006121 | 19.612376 |  | 16188 | 1.000000 | 0.453418 | 1.000000 |

Lowest metric SE(3) ATE: `learned_15dof_confidence` = 0.019119 m

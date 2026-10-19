# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 203 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 234 | 203 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.017711 | 0.028053 | 0.051065 |  |  |  |  |  |  |  | inv_center | 0.006140 | 19.636998 |  | 0 |  |  |  |
| learned_15dof_confidence | 290 | 203 | imu_alignment_unit_scale | 0.552120 | 1 | live | 40 | 0 | 0 | 0.017725 | 0.027948 | 0.050818 | 0.019697 | 0.095513 | 0.006415 | 0.028589 | 0.052354 | 0.260878 | 0.515968 | inv_center | 0.006117 | 19.615321 |  | 16188 | 1.000000 | 0.748014 | 1.000000 |

Lowest metric SE(3) ATE: `learned_15dof_confidence` = 0.019697 m

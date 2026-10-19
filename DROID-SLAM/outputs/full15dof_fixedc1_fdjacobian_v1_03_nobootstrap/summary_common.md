# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 211 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 234 | 211 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.017774 | 0.029038 | 0.057688 |  |  |  |  |  |  |  | inv_center | 0.006025 | 19.628537 | 0.033134 | 0 |  |  |  |
| fixed_c1_15dof | 236 | 211 | imu_alignment_unit_scale | 0.456245 | 1 | history | 24 | 0 | 0 | 0.017850 | 0.028951 | 0.057694 | 0.036938 | 0.091636 | 0.024277 | 0.035140 | 0.065800 | 0.259699 | 0.514675 | inv_center | 0.006035 | 19.614191 | 0.033118 | 14694 | 1.000000 | 1.000000 | 1.000000 |

Lowest metric SE(3) ATE: `fixed_c1_15dof` = 0.036938 m

# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 237 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned_002000 | 256 | 237 | imu_alignment_unit_scale | 0.883018 | 1 | live | 33 | 0 | 0 | 0.009615 | 0.011816 | 0.032338 | 0.178762 | 0.189887 | 0.094239 | 0.078122 | 0.224253 | 0.265434 | 0.903296 | inv_center | 0.031314 | 21.862791 | 0.021819 | 14488 | 1.000000 | 0.873187 | 0.002000 |
| learned_008000 | 256 | 237 | imu_alignment_unit_scale | 0.884901 | 1 | live | 33 | 0 | 0 | 0.009440 | 0.011867 | 0.031952 | 0.184947 | 0.196322 | 0.097834 | 0.080703 | 0.231599 | 0.266879 | 0.906472 | inv_center | 0.031304 | 21.862360 | 0.021824 | 14518 | 1.000000 | 0.879226 | 0.002000 |

Lowest metric SE(3) ATE: `learned_002000` = 0.178762 m

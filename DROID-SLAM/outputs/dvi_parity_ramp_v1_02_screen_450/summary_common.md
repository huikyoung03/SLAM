# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 173 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned_002000 | 173 | 173 | imu_alignment_unit_scale | 1.598692 | 1 | live | 44 | 0 | 0 | 0.009309 | 0.027441 | 0.068695 | 0.049440 | 0.080949 | 0.031221 | 0.038533 | 0.093211 | 0.151907 | 0.242244 | inv_center | 0.080134 | 16.571671 | 0.049025 | 6920 | 1.000000 | 0.854735 | 0.002000 |
| learned_004000 | 173 | 173 | imu_alignment_unit_scale | 1.598728 | 1 | live | 44 | 0 | 0 | 0.009286 | 0.027439 | 0.068696 | 0.049578 | 0.081080 | 0.031317 | 0.038585 | 0.093337 | 0.151867 | 0.242054 | inv_center | 0.080132 | 16.572016 | 0.049025 | 6920 | 1.000000 | 0.859038 | 0.002000 |
| learned_006000 | 173 | 173 | imu_alignment_unit_scale | 1.598700 | 1 | live | 44 | 0 | 0 | 0.009311 | 0.027443 | 0.068710 | 0.049548 | 0.081032 | 0.031293 | 0.038575 | 0.093322 | 0.151938 | 0.242242 | inv_center | 0.080133 | 16.571702 | 0.049026 | 6920 | 1.000000 | 0.861574 | 0.002000 |
| learned_008000 | 173 | 173 | imu_alignment_unit_scale | 1.598719 | 1 | live | 44 | 0 | 0 | 0.009301 | 0.027444 | 0.068701 | 0.049532 | 0.081012 | 0.031284 | 0.038571 | 0.093305 | 0.151908 | 0.242284 | inv_center | 0.080133 | 16.571614 | 0.049025 | 6920 | 1.000000 | 0.861164 | 0.002000 |

Lowest metric SE(3) ATE: `learned_002000` = 0.049440 m

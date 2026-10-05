# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.

| label | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_droid | 47 | 1.042859 | 0.852511 | 1.405630 | pose_t | 0.171270 | 16.685997 | 0.166113 | 0 |  |  |  |
| full_fixed | 43 | 0.071904 | 0.045767 | 0.125108 | inv_center | 0.109272 | 20.954590 | 0.113060 | 247 | 0.309524 | 0.336067 | 0.000067 |
| full_info_fixed | 43 | 0.071891 | 0.045759 | 0.125089 | inv_center | 0.109271 | 20.954867 | 0.113057 | 247 | 0.309524 | 0.336033 | 0.000672 |
| learned_weight_info | 43 | 0.069190 | 0.046277 | 0.117036 | inv_center | 0.111299 | 20.920132 | 0.112877 | 247 | 0.309524 | 0.335020 | 0.000674 |

Best ATE RMSE: `learned_weight_info` = 0.069190 m

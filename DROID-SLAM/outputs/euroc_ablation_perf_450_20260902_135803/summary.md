# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.

| label | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 43 | 0.069182 | 0.046308 | 0.116937 | inv_center | 0.111294 | 20.920795 | 0.112866 | 0 |  |  |  |
| imu_data_only | 43 | 0.069182 | 0.046308 | 0.116937 | inv_center | 0.111294 | 20.920795 | 0.112866 | 0 |  |  |  |
| rot_ba_fixed_w0002 | 43 | 0.069185 | 0.046266 | 0.116973 | inv_center | 0.111294 | 20.920342 | 0.112855 | 0 |  |  |  |
| full_ba_fixed_w0002 | 43 | 0.069191 | 0.046272 | 0.116985 | inv_center | 0.111298 | 20.920225 | 0.112856 | 247 | 0.309524 | 1.000000 | 0.002000 |
| full_ba_conf_fixed_w0002 | 43 | 0.069205 | 0.046265 | 0.117047 | inv_center | 0.111296 | 20.920123 | 0.112873 | 247 | 0.309524 | 0.335003 | 0.000670 |
| full_ba_conf_learned_scale004_info | 43 | 0.069200 | 0.046262 | 0.116960 | inv_center | 0.111300 | 20.920511 | 0.112878 | 247 | 0.309524 | 0.335034 | 0.000674 |

Best ATE RMSE: `original_droid` = 0.069182 m

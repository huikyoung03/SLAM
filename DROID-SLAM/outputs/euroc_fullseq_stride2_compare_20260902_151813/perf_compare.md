# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.

| label | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 234 | 0.072375 | 0.064187 | 0.100136 | inv_center | 0.003559 | 19.552707 | 0.019286 | 0 |  |  |  |
| full_imu | 234 | 0.072392 | 0.064281 | 0.100148 | inv_center | 0.003558 | 19.551378 | 0.019283 | 513 | 0.115880 | 0.347841 | 0.000700 |

Best ATE RMSE: `original_droid` = 0.072375 m

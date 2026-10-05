# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.

| label | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 38 | 0.030052 | 0.026345 | 0.045065 | inv_center | 0.158908 | 19.851484 | 0.173414 | 0 |  |  |  |
| imu_9dof | 38 | 0.030048 | 0.026448 | 0.045052 | inv_center | 0.158876 | 19.851700 | 0.173370 | 2931 | 1.000000 | 0.883265 | 0.047668 |

Best ATE RMSE: `imu_9dof` = 0.030048 m

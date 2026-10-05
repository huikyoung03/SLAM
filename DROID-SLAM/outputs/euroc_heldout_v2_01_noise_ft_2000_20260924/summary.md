# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.

| label | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 38 | 0.030052 | 0.026345 | 0.045065 | inv_center | 0.158908 | 19.851484 | 0.173414 | 0 |  |  |  |
| imu_9dof | 38 | 0.030064 | 0.026526 | 0.045056 | inv_center | 0.158907 | 19.851702 | 0.173401 | 2931 | 1.000000 | 0.970079 | 0.052940 |

Best ATE RMSE: `original_droid` = 0.030052 m

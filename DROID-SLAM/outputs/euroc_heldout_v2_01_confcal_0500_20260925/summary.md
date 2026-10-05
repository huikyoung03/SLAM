# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.
Runs are compared on the same 38 reconstruction frame IDs before alignment.

| label | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 38 | 0.030052 | 0.026345 | 0.045065 | inv_center | 0.158908 | 19.851484 | 0.173414 | 0 |  |  |  |
| imu_9dof | 38 | 0.030031 | 0.026482 | 0.044982 | inv_center | 0.158888 | 19.851621 | 0.173386 | 2931 | 1.000000 | 0.798396 | 0.043893 |

Best ATE RMSE: `imu_9dof` = 0.030031 m

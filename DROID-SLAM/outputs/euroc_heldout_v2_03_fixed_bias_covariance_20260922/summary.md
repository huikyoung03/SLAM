# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.

| label | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 120 | 0.048491 | 0.047925 | 0.065391 | inv_center | 0.062993 | 19.515377 | 0.058363 | 0 |  |  |  |
| imu_9dof | 121 | 0.048599 | 0.048673 | 0.065363 | inv_center | 0.062219 | 19.717328 | 0.057260 | 11155 | 0.998300 | 0.906528 | 0.048923 |

Best ATE RMSE: `original_droid` = 0.048491 m

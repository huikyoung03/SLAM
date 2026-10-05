# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.
Runs are compared on the same 116 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 120 | 116 | 0.047936 | 0.047395 | 0.064477 | inv_center | 0.064380 | 19.560152 | 0.059647 | 0 |  |  |  |
| imu_9dof | 121 | 116 | 0.047958 | 0.048065 | 0.063602 | inv_center | 0.064369 | 19.553557 | 0.059715 | 11155 | 0.998300 | 0.846543 | 0.050241 |

Best ATE RMSE: `original_droid` = 0.047936 m

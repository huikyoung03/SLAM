# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.
Runs are compared on the same 116 reconstruction frame IDs before alignment.

| label | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 116 | 0.047936 | 0.047395 | 0.064477 | inv_center | 0.064380 | 19.560152 | 0.059647 | 0 |  |  |  |
| imu_9dof | 116 | 0.047857 | 0.047449 | 0.063596 | inv_center | 0.064404 | 19.554724 | 0.059711 | 11155 | 0.998300 | 0.963589 | 0.052451 |

Best ATE RMSE: `imu_9dof` = 0.047857 m

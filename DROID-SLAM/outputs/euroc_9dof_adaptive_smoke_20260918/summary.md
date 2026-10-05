# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.

| label | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_9dof | 29 | 0.018525 | 0.012441 | 0.028207 | inv_center | 0.198802 | 9.926470 |  | 2103 | 0.736345 | 0.999741 | 0.053218 |
| adaptive_9dof | 29 | 0.018511 | 0.012429 | 0.028181 | inv_center | 0.198803 | 9.926436 |  | 2103 | 0.736345 | 0.999741 | 0.068753 |

Best ATE RMSE: `adaptive_9dof` = 0.018511 m

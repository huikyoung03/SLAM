# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.

| label | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_stride2_225 | 43 | 0.071706 | 0.054425 | 0.116047 | inv_center | 0.114623 | 18.157734 | 0.116201 | 0 |  |  |  |
| new_full_imu_scale004 | 43 | 0.071710 | 0.054522 | 0.116034 | inv_center | 0.114636 | 18.157476 | 0.116201 | 646 | 0.809524 | 0.100565 | 0.000202 |
| new_full_imu_scale1 | 43 | 0.071700 | 0.054548 | 0.116067 | inv_center | 0.114639 | 18.157436 | 0.116202 | 638 | 0.799499 | 0.100812 | 0.005065 |

Best ATE RMSE: `new_full_imu_scale1` = 0.071700 m

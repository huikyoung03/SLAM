# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.

| label | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_stride2_225 | 43 | 0.071706 | 0.054425 | 0.116047 | inv_center | 0.114623 | 18.157734 | 0.116201 | 0 |  |  |  |
| old_full_imu_v1 | 43 | 0.071723 | 0.054431 | 0.116080 | inv_center | 0.114614 | 18.157673 | 0.116212 | 646 | 0.809524 | 0.370534 | 0.000746 |
| new_full_imu_center_v2 | 43 | 0.071710 | 0.054522 | 0.116034 | inv_center | 0.114636 | 18.157476 | 0.116201 | 646 | 0.809524 | 0.100565 | 0.000202 |

Best ATE RMSE: `original_stride2_225` = 0.071706 m

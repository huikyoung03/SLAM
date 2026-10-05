# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.

| label | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 38 | 0.030052 | 0.026345 | 0.045065 | inv_center | 0.158908 | 19.851484 | 0.173414 | 0 |  |  |  |
| current_9dof | 38 | 0.030081 | 0.026574 | 0.045029 | inv_center | 0.158898 | 19.851708 | 0.173394 | 933 | 0.318321 | 0.874138 | 0.049589 |
| noise_no_limits | 38 | 0.029730 | 0.026531 | 0.044010 | inv_center | 0.158864 | 19.851644 | 0.173396 | 2931 | 1.000000 | 0.883265 | 0.047668 |
| noise_rawref | 38 | 0.030045 | 0.026496 | 0.044970 | inv_center | 0.158901 | 19.851747 | 0.173396 | 2931 | 1.000000 | 0.883268 | 0.047668 |

Best ATE RMSE: `noise_no_limits` = 0.029730 m

# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.

| label | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 98 | 0.062588 | 0.049570 | 0.099906 | inv_center | 0.084496 | 17.943485 | 0.051782 | 0 |  |  |  |
| imu_checkpoint_visual_only | 98 | 0.062588 | 0.049570 | 0.099906 | inv_center | 0.084496 | 17.943485 | 0.051782 | 0 |  |  |  |
| pose_velocity_imu | 98 | 0.062625 | 0.050083 | 0.100087 | inv_center | 0.084485 | 17.943565 | 0.051709 | 9923 | 0.852491 | 0.999757 | 0.053219 |

Best ATE RMSE: `original_droid` = 0.062588 m

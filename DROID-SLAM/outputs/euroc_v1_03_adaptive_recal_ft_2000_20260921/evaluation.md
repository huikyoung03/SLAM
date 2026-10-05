# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.

| label | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vi_initialized_adaptive | 99 | 0.062271 | 0.048535 | 0.099565 | inv_center | 0.082053 | 17.935766 | 0.050957 | 8848 | 0.868217 | 0.999753 | 0.058485 |
| adaptive_recal_ft_2000 | 99 | 0.062178 | 0.048571 | 0.099910 | inv_center | 0.082060 | 17.936029 | 0.050966 | 8848 | 0.868217 | 0.911360 | 0.054094 |

Best ATE RMSE: `adaptive_recal_ft_2000` = 0.062178 m

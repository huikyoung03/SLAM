# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.

| label | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 120 | 0.048491 | 0.047925 | 0.065391 | inv_center | 0.062993 | 19.515377 | 0.058363 | 0 |  |  |  |
| current_9dof | 121 | 0.048300 | 0.047748 | 0.065051 | inv_center | 0.061997 | 19.714255 | 0.057360 | 9149 | 0.818776 | 0.904280 | 0.052883 |
| noise_no_limits | 121 | 0.048567 | 0.048705 | 0.064999 | inv_center | 0.062168 | 19.716961 | 0.057250 | 11155 | 0.998300 | 0.906525 | 0.048923 |
| noise_rawref | 121 | 0.048560 | 0.048808 | 0.065145 | inv_center | 0.062193 | 19.717014 | 0.057265 | 11155 | 0.998300 | 0.906506 | 0.048922 |

Best ATE RMSE: `current_9dof` = 0.048300 m

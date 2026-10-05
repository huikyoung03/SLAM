# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.

| label | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 43 | 0.071706 | 0.054425 | 0.116047 | inv_center | 0.114623 | 18.157734 | 0.116201 | 0 |  |  |  |
| imu_data_only | 43 | 0.071706 | 0.054425 | 0.116047 | inv_center | 0.114623 | 18.157734 | 0.116201 | 0 |  |  |  |
| rot_ba_fixed_w0002 | 43 | 0.071706 | 0.054588 | 0.116058 | inv_center | 0.114633 | 18.157539 | 0.116207 | 0 |  |  |  |
| full_ba_fixed_w0002 | 43 | 0.071703 | 0.054581 | 0.116041 | inv_center | 0.114634 | 18.157535 | 0.116207 | 209 | 0.261905 | 1.000000 | 0.002000 |
| full_ba_conf_fixed_w0002 | 43 | 0.071731 | 0.054441 | 0.116111 | inv_center | 0.114603 | 18.157579 | 0.116210 | 209 | 0.261905 | 0.336933 | 0.000674 |
| full_ba_conf_learned_scale004_info | 43 | 0.071695 | 0.054448 | 0.116060 | inv_center | 0.114624 | 18.157464 | 0.116204 | 209 | 0.261905 | 0.337007 | 0.000678 |

Best ATE RMSE: `full_ba_conf_learned_scale004_info` = 0.071695 m

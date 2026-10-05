# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
SE(3) ATE and origin ATE preserve metric scale; RPE measures local drift.
Runs are compared on the same 38 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | sim3_ate_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 38 | 38 | 0.021040 | 1.014138 | 1.424912 | 1.167515 | 0.259141 | 0.870135 | 0.341881 | 0.703608 | inv_center | 0.204620 | 19.851484 | 0.223299 | 0 |  |  |  |
| imu_8k | 38 | 38 | 0.021016 | 1.015076 | 1.426248 | 1.169860 | 0.259367 | 0.870927 | 0.341171 | 0.703062 | inv_center | 0.204631 | 19.851699 | 0.223305 | 2931 | 1.000000 | 0.820111 | 0.048672 |

Lowest metric SE(3) ATE: `original_droid` = 1.014138 m

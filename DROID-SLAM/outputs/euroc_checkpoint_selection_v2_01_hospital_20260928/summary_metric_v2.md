# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 38 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 38 | 38 | monocular_scale_unobservable |  | 0.021040 | 0.010470 | 0.027359 |  |  |  |  |  |  |  | inv_center | 0.204620 | 19.851484 | 0.223299 | 0 |  |  |  |
| imu_8k | 38 | 38 | imu_alignment_unit_scale | 0.406726 | 0.021016 | 0.010439 | 0.027336 | 0.251413 | 0.345221 | 0.117461 | 0.061982 | 0.214566 | 0.341171 | 0.703062 | inv_center | 0.204631 | 19.851699 | 0.223305 | 2931 | 1.000000 | 0.820111 | 0.048672 |

Lowest metric SE(3) ATE: `imu_8k` = 0.251413 m

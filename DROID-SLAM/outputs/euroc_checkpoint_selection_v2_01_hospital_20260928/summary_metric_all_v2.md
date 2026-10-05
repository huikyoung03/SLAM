# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 38 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 38 | 38 | monocular_scale_unobservable |  | 0.021040 | 0.010470 | 0.027359 |  |  |  |  |  |  |  | inv_center | 0.204620 | 19.851484 | 0.223299 | 0 |  |  |  |
| imu_2k | 38 | 38 | imu_alignment_unit_scale | 0.406681 | 0.021126 | 0.010442 | 0.027333 | 0.252827 | 0.346921 | 0.118045 | 0.062385 | 0.215875 | 0.341475 | 0.703245 | inv_center | 0.204541 | 19.851568 | 0.223201 | 2931 | 1.000000 | 0.753427 | 0.042879 |
| imu_4k | 38 | 38 | imu_alignment_unit_scale | 0.406629 | 0.021206 | 0.010435 | 0.027428 | 0.251913 | 0.345976 | 0.117662 | 0.062148 | 0.215023 | 0.341301 | 0.702547 | inv_center | 0.204665 | 19.851967 | 0.223325 | 2931 | 1.000000 | 0.807154 | 0.047163 |
| imu_6k | 38 | 38 | imu_alignment_unit_scale | 0.406693 | 0.021012 | 0.010449 | 0.027369 | 0.251735 | 0.345630 | 0.117596 | 0.062083 | 0.214811 | 0.341023 | 0.702632 | inv_center | 0.204599 | 19.851646 | 0.223266 | 2931 | 1.000000 | 0.832911 | 0.049335 |
| imu_8k | 38 | 38 | imu_alignment_unit_scale | 0.406726 | 0.021016 | 0.010439 | 0.027336 | 0.251413 | 0.345221 | 0.117461 | 0.061982 | 0.214566 | 0.341171 | 0.703062 | inv_center | 0.204631 | 19.851699 | 0.223305 | 2931 | 1.000000 | 0.820111 | 0.048672 |

Lowest metric SE(3) ATE: `imu_8k` = 0.251413 m

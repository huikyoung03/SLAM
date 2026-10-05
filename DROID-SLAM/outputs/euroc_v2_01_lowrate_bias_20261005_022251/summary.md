# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 73 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 101 | 73 | monocular_scale_unobservable |  | 0.017135 | 0.010999 | 0.032337 |  |  |  |  |  |  |  | inv_center | 0.082823 | 21.006754 |  | 0 |  |  |  |
| imu_9dof_fixed | 151 | 73 | monocular_scale_unobservable |  | 0.017359 | 0.010923 | 0.032160 |  |  |  |  |  |  |  | inv_center | 0.082805 | 21.016388 |  | 0 |  |  |  |
| imu_9dof_lowrate_bias | 151 | 73 | monocular_scale_unobservable |  | 0.017359 | 0.010923 | 0.032160 |  |  |  |  |  |  |  | inv_center | 0.082805 | 21.016388 |  | 0 |  |  |  |

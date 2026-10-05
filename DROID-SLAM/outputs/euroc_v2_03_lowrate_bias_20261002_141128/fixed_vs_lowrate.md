# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 318 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| imu_9dof_fixed | 324 | 318 | imu_alignment_unit_scale | 0.609494 | 0.014709 | 0.014329 | 0.030542 | 0.344864 | 0.360511 | 0.204371 | 0.146780 | 0.455125 | 0.347250 | 0.725807 | inv_center | 0.028333 | 20.484036 | 0.028415 | 0 |  |  |  |
| imu_9dof_lowrate_bias | 322 | 318 | imu_alignment_unit_scale | 0.609297 | 0.015121 | 0.014366 | 0.030606 | 0.337986 | 0.353178 | 0.199460 | 0.143894 | 0.446119 | 0.348404 | 0.724030 | inv_center | 0.028337 | 20.483847 | 0.028425 | 0 |  |  |  |

Lowest metric SE(3) ATE: `imu_9dof_lowrate_bias` = 0.337986 m

# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 324 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| imu_9dof_fixed_outer_control | 324 | 324 | imu_alignment_unit_scale | 0.782405 | 0.014448 | 0.014253 | 0.030408 | 0.730205 | 0.764373 | 0.560059 | 0.315666 | 0.954364 | 0.341479 | 0.714058 | inv_center | 0.028296 | 20.281860 | 0.028290 | 0 |  |  |  |
| imu_9dof_joint_shared_bias | 324 | 324 | imu_alignment_unit_scale | 0.783425 | 0.014674 | 0.014319 | 0.030416 | 0.729704 | 0.763846 | 0.559454 | 0.315453 | 0.953755 | 0.342523 | 0.716023 | inv_center | 0.028292 | 20.281367 | 0.028286 | 0 |  |  |  |

Lowest metric SE(3) ATE: `imu_9dof_joint_shared_bias` = 0.729704 m

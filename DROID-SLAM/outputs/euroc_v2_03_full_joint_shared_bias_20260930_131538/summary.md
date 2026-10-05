# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 261 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 271 | 261 | monocular_scale_unobservable |  | 0.015098 | 0.016147 | 0.032168 |  |  |  |  |  |  |  | inv_center | 0.028548 | 21.087925 |  | 0 |  |  |  |
| imu_9dof_fixed | 324 | 261 | imu_alignment_unit_scale | 0.609494 | 0.015008 | 0.015943 | 0.032274 | 0.371927 | 0.389111 | 0.204836 | 0.167501 | 0.482221 | 0.378097 | 0.768832 | inv_center | 0.028506 | 21.075836 |  | 0 |  |  |  |
| imu_9dof_joint_shared_bias | 324 | 261 | imu_alignment_unit_scale | 0.744228 | 0.014862 | 0.015907 | 0.032311 | 0.719031 | 0.754269 | 0.490031 | 0.322678 | 0.930894 | 0.376537 | 0.764656 | inv_center | 0.028472 | 21.087108 |  | 0 |  |  |  |

Lowest metric SE(3) ATE: `imu_9dof_fixed` = 0.371927 m

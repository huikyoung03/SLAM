# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| imu_9dof_fixed | 324 | 324 | imu_alignment_unit_scale | 0.609494 | 0.014614 | 0.014290 | 0.030379 | 0.345422 | 0.360192 | 0.204391 | 0.150006 | 0.452673 | 0.341717 | 0.720223 | inv_center | 0.028324 | 20.281693 | 0.028309 | 26118 | 0.999273 | 0.899596 | 0.053381 |
| imu_9dof_shared_bias | 322 | 322 | imu_alignment_unit_scale | 0.611200 | 0.015010 | 0.014399 | 0.030384 | 0.342848 | 0.357895 | 0.203160 | 0.147172 | 0.449694 | 0.343976 | 0.717879 | inv_center | 0.028343 | 20.299993 | 0.028326 | 26720 | 0.999215 | 0.898588 | 0.053321 |

Lowest metric SE(3) ATE: `imu_9dof_shared_bias` = 0.342848 m

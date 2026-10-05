# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 255 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 271 | 255 | monocular_scale_unobservable |  | 0.015217 | 0.016295 | 0.032387 |  |  |  |  |  |  |  | inv_center | 0.028557 | 21.247695 |  | 0 |  |  |  |
| imu_9dof_fixed | 324 | 255 | imu_alignment_unit_scale | 0.609494 | 0.015122 | 0.016077 | 0.032499 | 0.371717 | 0.390679 | 0.204840 | 0.164822 | 0.485703 | 0.386505 | 0.776031 | inv_center | 0.028515 | 21.221697 |  | 26118 | 0.999273 | 0.899596 | 0.053381 |
| imu_9dof_shared_bias | 322 | 255 | imu_alignment_unit_scale | 0.611200 | 0.015428 | 0.016131 | 0.032542 | 0.369943 | 0.388673 | 0.203654 | 0.163951 | 0.483353 | 0.385779 | 0.772446 | inv_center | 0.028527 | 21.228257 |  | 26720 | 0.999215 | 0.898588 | 0.053321 |

Lowest metric SE(3) ATE: `imu_9dof_shared_bias` = 0.369943 m

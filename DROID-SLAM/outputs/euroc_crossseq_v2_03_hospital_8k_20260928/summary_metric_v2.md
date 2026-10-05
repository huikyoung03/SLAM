# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 116 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 120 | 116 | monocular_scale_unobservable |  | 0.017100 | 0.016325 | 0.028775 |  |  |  |  |  |  |  | inv_center | 0.078637 | 19.560152 | 0.072115 | 0 |  |  |  |
| imu_8k | 121 | 116 | imu_alignment_unit_scale | 0.460616 | 0.017037 | 0.016296 | 0.028591 | 0.166526 | 0.181821 | 0.078363 | 0.081664 | 0.228172 | 0.284933 | 0.616593 | inv_center | 0.078608 | 19.553557 | 0.072110 | 11155 | 0.998300 | 0.846543 | 0.050241 |

Lowest metric SE(3) ATE: `imu_8k` = 0.166526 m

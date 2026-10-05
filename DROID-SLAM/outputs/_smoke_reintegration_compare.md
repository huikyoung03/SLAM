# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 84 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal | 84 | 84 | imu_alignment_unit_scale | 0.646602 | 0.009768 | 0.010229 | 0.012135 | 0.137789 | 0.160038 | 0.149965 | 0.088195 | 0.149409 | 0.152080 | 0.334336 | inv_center | 0.205729 | 10.353001 | 0.236950 | 0 |  |  |  |
| reintegrated | 84 | 84 | imu_alignment_unit_scale | 0.643480 | 0.010134 | 0.011296 | 0.012283 | 0.152372 | 0.176348 | 0.168570 | 0.098729 | 0.168138 | 0.148776 | 0.318644 | inv_center | 0.206644 | 10.348123 | 0.238147 | 0 |  |  |  |

Lowest metric SE(3) ATE: `nominal` = 0.137789 m

# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned_006000 | 274 | 274 | imu_alignment_unit_scale | 1.277980 | 1 | live | 44 | 0 | 0 | 0.009897 | 0.030612 | 0.075727 | 0.171602 | 0.214485 | 0.117758 | 0.105375 | 0.257620 | 0.172796 | 0.288697 | inv_center | 0.000154 | 17.174631 | 0.026473 | 12020 | 1.000000 | 0.935524 | 0.002000 |
| learned_008000 | 274 | 274 | imu_alignment_unit_scale | 1.278041 | 1 | live | 44 | 0 | 0 | 0.009897 | 0.030612 | 0.075720 | 0.171765 | 0.214668 | 0.117883 | 0.105466 | 0.257843 | 0.172743 | 0.288693 | inv_center | 0.000154 | 17.175049 | 0.026473 | 12020 | 1.000000 | 0.934281 | 0.002000 |

Lowest metric SE(3) ATE: `learned_006000` = 0.171602 m

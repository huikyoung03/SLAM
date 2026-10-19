# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 260 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned_001000 | 274 | 260 | imu_alignment_unit_scale | 1.278075 | 1 | live | 44 | 0 | 0 | 0.009331 | 0.030276 | 0.073983 | 0.167135 | 0.208761 | 0.118551 | 0.104757 | 0.255837 | 0.172311 | 0.280471 | inv_center | 0.000160 | 17.155714 | 0.026437 | 12020 | 1.000000 | 0.870532 | 0.002000 |
| learned_004000 | 274 | 260 | imu_alignment_unit_scale | 1.278134 | 1 | live | 44 | 0 | 0 | 0.009335 | 0.030277 | 0.073979 | 0.167317 | 0.208973 | 0.118695 | 0.104862 | 0.256092 | 0.172260 | 0.280497 | inv_center | 0.000160 | 17.155878 | 0.026437 | 12020 | 1.000000 | 0.932724 | 0.002000 |
| learned_005000 | 274 | 260 | imu_alignment_unit_scale | 1.278534 | 1 | live | 44 | 0 | 0 | 0.010175 | 0.030275 | 0.074032 | 0.165889 | 0.205931 | 0.117521 | 0.104221 | 0.254792 | 0.172936 | 0.287681 | inv_center | 0.000213 | 17.153714 | 0.026487 | 12048 | 1.000000 | 0.935218 | 0.002000 |
| learned_006000 | 274 | 260 | imu_alignment_unit_scale | 1.277980 | 1 | live | 44 | 0 | 0 | 0.009329 | 0.030276 | 0.073984 | 0.166938 | 0.208538 | 0.118394 | 0.104644 | 0.255559 | 0.172282 | 0.280426 | inv_center | 0.000160 | 17.155641 | 0.026437 | 12020 | 1.000000 | 0.935524 | 0.002000 |
| learned_008000 | 274 | 260 | imu_alignment_unit_scale | 1.278041 | 1 | live | 44 | 0 | 0 | 0.009331 | 0.030276 | 0.073979 | 0.167094 | 0.208716 | 0.118518 | 0.104733 | 0.255779 | 0.172249 | 0.280460 | inv_center | 0.000160 | 17.155804 | 0.026437 | 12020 | 1.000000 | 0.934281 | 0.002000 |

Lowest metric SE(3) ATE: `learned_005000` = 0.165889 m

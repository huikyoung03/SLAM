# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 274 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned_001000 | 274 | 274 | imu_alignment_unit_scale | 1.280213 | 1 | live | 44 | 0 | 0 | 0.010651 | 0.030650 | 0.075790 | 0.189939 | 0.234146 | 0.132017 | 0.115820 | 0.283455 | 0.174775 | 0.293530 | inv_center | 0.000206 | 17.173534 | 0.026480 | 12020 | 1.000000 | 0.947589 | 1.000000 |
| learned_002000 | 274 | 274 | imu_alignment_unit_scale | 1.280200 | 1 | live | 44 | 0 | 0 | 0.010670 | 0.030657 | 0.075807 | 0.189887 | 0.234064 | 0.131975 | 0.115793 | 0.283396 | 0.174917 | 0.293551 | inv_center | 0.000206 | 17.173625 | 0.026481 | 12020 | 1.000000 | 0.928884 | 1.000000 |
| learned_003000 | 274 | 274 | imu_alignment_unit_scale | 1.280168 | 1 | live | 44 | 0 | 0 | 0.010669 | 0.030655 | 0.075804 | 0.189638 | 0.233783 | 0.131778 | 0.115653 | 0.283051 | 0.174833 | 0.293536 | inv_center | 0.000206 | 17.173325 | 0.026481 | 12020 | 1.000000 | 0.924711 | 1.000000 |

Lowest metric SE(3) ATE: `learned_003000` = 0.189638 m

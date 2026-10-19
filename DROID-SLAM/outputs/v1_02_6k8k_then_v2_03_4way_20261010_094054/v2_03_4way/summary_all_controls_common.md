# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 241 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 271 | 241 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.015200 | 0.016589 | 0.033838 |  |  |  |  |  |  |  | inv_center | 0.029110 | 21.447248 |  | 0 |  |  |  |
| fixed_9d | 324 | 241 | imu_alignment_unit_scale | 0.612596 | 1 | live | 41 | 0 | 0 | 0.015173 | 0.016304 | 0.033890 | 0.388726 | 0.406217 | 0.213119 | 0.175549 | 0.521426 | 0.390046 | 0.803116 | inv_center | 0.029058 | 21.371219 |  | 26118 | 0.999273 | 1.000000 | 0.002000 |
| fixed_15d | 322 | 241 | imu_alignment_unit_scale | 0.578105 | 1 | live | 41 | 0 | 0 | 0.015626 | 0.016408 | 0.034970 | 0.380986 | 0.397746 | 0.207973 | 0.172082 | 0.511077 | 0.388876 | 0.809864 | inv_center | 0.029035 | 21.400990 |  | 13648 | 0.999122 | 1.000000 | 0.002000 |
| tuned_ramp_15d | 322 | 241 | imu_alignment_unit_scale | 0.575078 | 1 | live | 41 | 0 | 0 | 0.015682 | 0.016440 | 0.035128 | 0.367517 | 0.383446 | 0.199140 | 0.165929 | 0.493024 | 0.389328 | 0.809139 | inv_center | 0.029031 | 21.395416 |  | 13598 | 0.999118 | 0.516197 | 0.002000 |
| learned_15d | 323 | 241 | imu_alignment_unit_scale | 0.581447 | 1 | live | 41 | 0 | 0 | 0.015657 | 0.016541 | 0.035094 | 0.396898 | 0.414518 | 0.218574 | 0.179146 | 0.532370 | 0.389310 | 0.809360 | inv_center | 0.029037 | 21.412714 |  | 13470 | 0.999110 | 0.931566 | 0.002000 |

Lowest metric SE(3) ATE: `tuned_ramp_15d` = 0.367517 m

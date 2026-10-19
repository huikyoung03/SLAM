# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 242 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 271 | 242 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.015228 | 0.016590 | 0.033857 |  |  |  |  |  |  |  | inv_center | 0.029096 | 21.437407 |  | 0 |  |  |  |
| fixed_15d | 322 | 242 | imu_alignment_unit_scale | 0.578105 | 1 | live | 41 | 0 | 0 | 0.015644 | 0.016409 | 0.034995 | 0.380516 | 0.397104 | 0.207953 | 0.172082 | 0.511091 | 0.388876 | 0.809345 | inv_center | 0.029022 | 21.390827 |  | 13648 | 0.999122 | 1.000000 | 0.002000 |
| tuned_ramp_15d | 322 | 242 | imu_alignment_unit_scale | 0.575078 | 1 | live | 41 | 0 | 0 | 0.015699 | 0.016440 | 0.035151 | 0.367063 | 0.382827 | 0.199121 | 0.165929 | 0.493037 | 0.389328 | 0.808642 | inv_center | 0.029017 | 21.386521 |  | 13598 | 0.999118 | 0.516197 | 0.002000 |
| learned_15d | 323 | 242 | imu_alignment_unit_scale | 0.581447 | 1 | live | 41 | 0 | 0 | 0.015674 | 0.016541 | 0.035121 | 0.396410 | 0.413850 | 0.218554 | 0.179146 | 0.532384 | 0.389310 | 0.808848 | inv_center | 0.029023 | 21.401761 |  | 13470 | 0.999110 | 0.931566 | 0.002000 |

Lowest metric SE(3) ATE: `tuned_ramp_15d` = 0.367063 m

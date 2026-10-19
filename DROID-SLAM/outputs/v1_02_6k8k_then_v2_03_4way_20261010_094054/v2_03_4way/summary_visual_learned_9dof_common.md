# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 241 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visual | 313 | 241 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.014417 | 0.016543 | 0.032936 |  |  |  |  |  |  |  | inv_center | 0.028985 | 21.785537 |  | 0 |  |  |  |
| learned_15dof | 323 | 241 | imu_alignment_unit_scale | 0.581447 | 1 | live | 41 | 0 | 0 | 0.015642 | 0.016430 | 0.033541 | 0.395885 | 0.413142 | 0.218475 | 0.177263 | 0.512007 | 0.386458 | 0.792160 | inv_center | 0.029002 | 21.837841 |  | 13470 | 0.999110 | 0.931566 | 0.002000 |
| fixed_9dof | 324 | 241 | imu_alignment_unit_scale | 0.612596 | 1 | live | 41 | 0 | 0 | 0.015142 | 0.016200 | 0.032441 | 0.387789 | 0.404922 | 0.213060 | 0.173702 | 0.501613 | 0.387243 | 0.787248 | inv_center | 0.029024 | 21.815202 |  | 26118 | 0.999273 | 1.000000 | 0.002000 |

Lowest metric SE(3) ATE: `fixed_9dof` = 0.387789 m

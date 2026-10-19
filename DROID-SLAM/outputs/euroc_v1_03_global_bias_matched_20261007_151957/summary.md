# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 41 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 98 | 41 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.009562 | 0.043747 | 0.100463 |  |  |  |  |  |  |  | inv_center | 0.196077 | 15.510237 |  | 0 |  |  |  |
| visual_bootstrap_control | 134 | 41 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.008096 | 0.042472 | 0.096346 |  |  |  |  |  |  |  | inv_center | 0.195367 | 15.525750 |  | 0 |  |  |  |
| two_stage_15dof_pass0 | 154 | 41 | imu_alignment_unit_scale | 0.628451 | 1 | live | 40 | 0 | 0 | 0.009572 | 0.044209 | 0.102020 | 0.017311 | 0.117326 | 0.008096 | 0.045073 | 0.104646 | 0.273853 | 0.646681 | inv_center | 0.195723 | 15.514389 |  | 5202 | 0.886503 | 1.000000 | 0.002000 |
| two_stage_15dof_post_control | 154 | 41 | imu_alignment_unit_scale | 0.754321 | 1 | live | 40 | 0 | 0 | 0.009285 | 0.043547 | 0.100417 | 0.015676 | 0.116760 | 0.007096 | 0.044253 | 0.102583 | 0.269363 | 0.646833 | inv_center | 0.195749 | 15.491697 |  | 5476 | 0.886945 | 1.000000 | 0.002000 |
| two_stage_15dof_global_bias | 154 | 41 | imu_alignment_unit_scale | 0.754321 | 1 | live | 40 | 3 | 3 | 0.009286 | 0.043547 | 0.100420 | 0.015677 | 0.116762 | 0.007097 | 0.044253 | 0.102587 | 0.269363 | 0.646811 | inv_center | 0.195749 | 15.491684 |  | 5476 | 0.886945 | 1.000000 | 0.002000 |

Lowest metric SE(3) ATE: `two_stage_15dof_post_control` = 0.015676 m

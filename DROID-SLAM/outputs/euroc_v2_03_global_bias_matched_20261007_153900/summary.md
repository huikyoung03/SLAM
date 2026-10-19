# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 99 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 120 | 99 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.017900 | 0.016872 | 0.031523 |  |  |  |  |  |  |  | inv_center | 0.081026 | 19.733235 |  | 0 |  |  |  |
| visual_bootstrap_control | 161 | 99 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.019180 | 0.017110 | 0.034754 |  |  |  |  |  |  |  | inv_center | 0.080758 | 19.734090 |  | 0 |  |  |  |
| two_stage_15dof_pass0 | 171 | 99 | imu_alignment_unit_scale | 0.480830 | 1 | live | 41 | 0 | 0 | 0.017254 | 0.016747 | 0.031346 | 0.222643 | 0.236385 | 0.102304 | 0.105859 | 0.316397 | 0.279431 | 0.641192 | inv_center | 0.080669 | 19.732659 |  | 5336 | 0.795468 | 1.000000 | 0.002000 |
| two_stage_15dof_post_control | 171 | 99 | imu_alignment_unit_scale | 0.610355 | 1 | live | 41 | 0 | 0 | 0.018820 | 0.016898 | 0.031403 | 0.226775 | 0.238678 | 0.104350 | 0.107191 | 0.322780 | 0.297237 | 0.622935 | inv_center | 0.080667 | 19.731614 |  | 5622 | 0.797673 | 1.000000 | 0.002000 |
| two_stage_15dof_global_bias | 171 | 99 | imu_alignment_unit_scale | 0.610355 | 1 | live | 41 | 3 | 3 | 0.018820 | 0.016897 | 0.031403 | 0.226775 | 0.238677 | 0.104350 | 0.107190 | 0.322779 | 0.297237 | 0.622938 | inv_center | 0.080667 | 19.731603 |  | 5622 | 0.797673 | 1.000000 | 0.002000 |

Lowest metric SE(3) ATE: `two_stage_15dof_pass0` = 0.222643 m

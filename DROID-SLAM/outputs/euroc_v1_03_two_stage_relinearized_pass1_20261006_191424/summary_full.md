# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 98 | 98 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.015327 | 0.035534 | 0.096233 |  |  |  |  |  |  |  | inv_center | 0.131153 | 17.943485 | 0.081008 | 0 |  |  |  |
| visual_bootstrap_control | 134 | 134 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.013399 | 0.041539 | 0.099578 |  |  |  |  |  |  |  | inv_center | 0.125559 | 17.407338 | 0.087080 | 0 |  |  |  |
| two_stage_pass0 | 154 | 154 | imu_alignment_unit_scale | 0.628451 | 1 | live | 40 | 0 | 0 | 0.014291 | 0.028812 | 0.084518 | 0.021726 | 0.084180 | 0.010029 | 0.029427 | 0.086690 | 0.204601 | 0.498698 | inv_center | 0.128513 | 17.100357 | 0.080397 | 10404 | 0.886503 | 1.000000 | 0.002000 |
| two_stage_reintegrated | 154 | 154 | imu_alignment_unit_scale | 0.626727 | 1 | live | 40 | 0 | 0 | 0.014290 | 0.028806 | 0.084495 | 0.025318 | 0.086073 | 0.012773 | 0.029921 | 0.088067 | 0.204421 | 0.498180 | inv_center | 0.128512 | 17.100479 | 0.080396 | 5202 | 0.886503 | 1.000000 | 0.002000 |

Lowest metric SE(3) ATE: `two_stage_pass0` = 0.021726 m

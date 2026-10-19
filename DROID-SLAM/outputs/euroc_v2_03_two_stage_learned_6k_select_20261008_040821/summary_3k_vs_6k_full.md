# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 120 | 120 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.016978 | 0.016093 | 0.028120 |  |  |  |  |  |  |  | inv_center | 0.077841 | 19.515377 | 0.071384 | 0 |  |  |  |
| visual_bootstrap_control | 161 | 161 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.016211 | 0.014287 | 0.029731 |  |  |  |  |  |  |  | inv_center | 0.075917 | 18.265894 | 0.062621 | 0 |  |  |  |
| two_stage_15dof_6k | 171 | 171 | imu_alignment_unit_scale | 0.480819 | 1 | live | 41 | 0 | 0 | 0.014623 | 0.013196 | 0.025747 | 0.186300 | 0.200217 | 0.102532 | 0.082126 | 0.256064 | 0.229011 | 0.533050 | inv_center | 0.077662 | 18.173854 | 0.074956 | 6696 | 0.998211 | 0.654780 | 0.036401 |
| two_stage_15dof_3k | 171 | 171 | imu_alignment_unit_scale | 0.480975 | 1 | live | 41 | 0 | 0 | 0.014612 | 0.013198 | 0.025739 | 0.186152 | 0.200055 | 0.102442 | 0.082059 | 0.255850 | 0.229191 | 0.533158 | inv_center | 0.077667 | 18.173133 | 0.074958 | 6696 | 0.998211 | 0.651215 | 0.035503 |

Lowest metric SE(3) ATE: `two_stage_15dof_3k` = 0.186152 m

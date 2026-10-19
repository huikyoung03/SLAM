# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 1075 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 1075 | 1075 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.074870 | 1.131035 | 2.746387 |  |  |  |  |  |  |  | camera_c2w_wxyz | 0.004592 | 7.564686 | 0.025790 | 0 |  |  |  |
| learned_15dof_confidence | 1075 | 1075 | imu_alignment_unit_scale | 0.456203 | 1 | history | 24 | 0 | 0 | 0.086471 | 1.131522 | 2.745908 | 0.091500 | 3.520820 | 0.019268 | 1.120836 | 2.720046 | 55.132084 | 97.145401 | camera_c2w_wxyz | 0.004552 | 7.552528 | 0.025549 | 14694 | 1.000000 | 0.594805 | 1.000000 |

Lowest metric SE(3) ATE: `learned_15dof_confidence` = 0.091500 m

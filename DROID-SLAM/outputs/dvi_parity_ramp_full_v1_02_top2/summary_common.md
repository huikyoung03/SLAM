# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 274 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned_002000 | 274 | 274 | imu_alignment_unit_scale | 1.277979 | 1 | live | 44 | 0 | 0 | 0.009901 | 0.030612 | 0.075720 | 0.171629 | 0.214517 | 0.117779 | 0.105390 | 0.257657 | 0.172806 | 0.288710 | inv_center | 0.000154 | 17.174888 | 0.026473 | 12020 | 1.000000 | 0.858982 | 0.002000 |
| learned_008000 | 274 | 274 | imu_alignment_unit_scale | 1.278075 | 1 | live | 44 | 0 | 0 | 0.009902 | 0.030613 | 0.075721 | 0.171845 | 0.214760 | 0.117945 | 0.105512 | 0.257956 | 0.172787 | 0.288763 | inv_center | 0.000154 | 17.174861 | 0.026472 | 12020 | 1.000000 | 0.866443 | 0.002000 |

Lowest metric SE(3) ATE: `learned_002000` = 0.171629 m

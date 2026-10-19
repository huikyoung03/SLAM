# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 239 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | vi_init_requested | vi_init_source | vi_init_edges | bias_refine_attempts | bias_refine_updates | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 313 | 239 | monocular_scale_unobservable |  | 0 | visual_only | 0 | 0 | 0 | 0.014385 | 0.016665 | 0.033594 |  |  |  |  |  |  |  | inv_center | 0.029289 | 21.554050 |  | 0 |  |  |  |
| fixed_15d | 322 | 239 | imu_alignment_unit_scale | 0.578105 | 1 | live | 41 | 0 | 0 | 0.015581 | 0.016436 | 0.034088 | 0.380007 | 0.393981 | 0.207948 | 0.172874 | 0.502313 | 0.385451 | 0.812343 | inv_center | 0.029306 | 21.421314 |  | 13648 | 0.999122 | 1.000000 | 0.002000 |
| tuned_15d | 322 | 239 | imu_alignment_unit_scale | 0.575078 | 1 | live | 41 | 0 | 0 | 0.015643 | 0.016466 | 0.034262 | 0.366563 | 0.379840 | 0.199109 | 0.166692 | 0.484578 | 0.386006 | 0.811898 | inv_center | 0.029301 | 21.413207 |  | 13598 | 0.999118 | 0.516197 | 0.002000 |
| learned_old | 323 | 239 | imu_alignment_unit_scale | 0.581447 | 1 | live | 41 | 0 | 0 | 0.015614 | 0.016572 | 0.034226 | 0.395872 | 0.410587 | 0.218543 | 0.179980 | 0.523240 | 0.385869 | 0.811854 | inv_center | 0.029307 | 21.434621 |  | 13470 | 0.999110 | 0.931566 | 0.002000 |
| learned_parity_2k | 322 | 239 | imu_alignment_unit_scale | 0.575205 | 1 | live | 41 | 0 | 0 | 0.015849 | 0.016630 | 0.034364 | 0.367220 | 0.380563 | 0.199533 | 0.167080 | 0.485570 | 0.387595 | 0.814893 | inv_center | 0.029309 | 21.406893 |  | 13598 | 0.999118 | 0.854848 | 0.002000 |

Lowest metric SE(3) ATE: `tuned_15d` = 0.366563 m

# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
SE(3) ATE and origin ATE preserve metric scale; RPE measures local drift.
Runs are compared on the same 116 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | sim3_ate_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 120 | 116 | 0.017100 | 1.305119 | 1.371742 | 1.339073 | 0.616425 | 1.757459 | 0.282364 | 0.608558 | inv_center | 0.078637 | 19.560152 | 0.072115 | 0 |  |  |  |
| imu_8k | 121 | 116 | 0.017037 | 1.305975 | 1.372620 | 1.341130 | 0.616780 | 1.758520 | 0.284933 | 0.616593 | inv_center | 0.078608 | 19.553557 | 0.072110 | 11155 | 0.998300 | 0.846543 | 0.050241 |

Lowest metric SE(3) ATE: `original_droid` = 1.305119 m

# EuRoC Ablation Summary

Sim(3) ATE is a scale-corrected reconstruction-shape metric only.
Sim(3) RPE compares local drift after an oracle global scale fit.
Metric SE(3)/origin ATE and metric RPE are reported only when the run
contains a successful VI alignment and its DROID unit scale is converted to metres.
Runs are compared on the same 101 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | metricization | pose_unit_scale | sim3_ate_rmse_m | sim3_rpe_1s_trans_rmse_m | sim3_rpe_5s_trans_rmse_m | se3_ate_rmse_m | origin_ate_rmse_m | metric_scale_error | rpe_1s_trans_rmse_m | rpe_5s_trans_rmse_m | rpe_1s_rot_rmse_deg | rpe_5s_rot_rmse_deg | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 101 | 101 | monocular_scale_unobservable |  | 0.016053 | 0.008588 | 0.029872 |  |  |  |  |  |  |  | inv_center | 0.067832 | 20.882088 | 0.086395 | 0 |  |  |  |
| imu_pose_velocity_9dof_axisfix_hospital_wimu0.05_002000 | 101 | 101 | imu_alignment_unit_scale | 0.685248 | 0.016032 | 0.008570 | 0.029861 | 0.080220 | 0.096173 | 0.031899 | 0.013757 | 0.054798 | 0.181539 | 0.519363 | inv_center | 0.067829 | 20.884286 | 0.086385 | 11761 | 1.000000 | 0.831859 | 0.045180 |
| imu_pose_velocity_9dof_axisfix_hospital_wimu0.05_004000 | 101 | 101 | imu_alignment_unit_scale | 0.685220 | 0.016074 | 0.008573 | 0.029867 | 0.080560 | 0.096555 | 0.032031 | 0.013810 | 0.054988 | 0.181473 | 0.519130 | inv_center | 0.067834 | 20.884114 | 0.086391 | 11761 | 1.000000 | 0.842968 | 0.048621 |
| imu_pose_velocity_9dof_axisfix_hospital_wimu0.05_006000 | 101 | 101 | imu_alignment_unit_scale | 0.685227 | 0.016109 | 0.008590 | 0.029879 | 0.079226 | 0.095231 | 0.031493 | 0.013668 | 0.054332 | 0.180252 | 0.518481 | inv_center | 0.067844 | 20.884433 | 0.086381 | 11761 | 1.000000 | 0.833849 | 0.049480 |
| imu_pose_velocity_9dof_axisfix_hospital_wimu0.05_008000 | 101 | 101 | imu_alignment_unit_scale | 0.685210 | 0.016106 | 0.008586 | 0.029900 | 0.080511 | 0.096510 | 0.032009 | 0.013805 | 0.054964 | 0.181375 | 0.518607 | inv_center | 0.067832 | 20.884884 | 0.086397 | 11761 | 1.000000 | 0.829697 | 0.049455 |

Lowest metric SE(3) ATE: `imu_pose_velocity_9dof_axisfix_hospital_wimu0.05_006000` = 0.079226 m

# EuRoC Ablation Summary

ATE is Sim(3)-aligned and should be read as a reconstruction comparison metric.
Runs are compared on the same 38 reconstruction frame IDs before alignment.

| label | raw_n_poses | n_poses | ate_rmse_m | ate_median_m | ate_p90_m | gt_pose_mode | start_end_path | rot_p95_deg | return_gap_median_path | imu_used | imu_used_ratio | imu_conf_mean | imu_weight_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_droid | 38 | 38 | 0.030052 | 0.026345 | 0.045065 | inv_center | 0.158908 | 19.851484 | 0.173414 | 0 |  |  |  |
| imu_pose_velocity_9dof_hospital_v1_002000 | 38 | 38 | 0.030036 | 0.026387 | 0.045075 | inv_center | 0.158879 | 19.851568 | 0.173373 | 2931 | 1.000000 | 0.753427 | 0.042879 |
| imu_pose_velocity_9dof_hospital_v1_004000 | 38 | 38 | 0.030135 | 0.026673 | 0.045102 | inv_center | 0.158934 | 19.851967 | 0.173424 | 2931 | 1.000000 | 0.807154 | 0.047163 |
| imu_pose_velocity_9dof_hospital_v1_006000 | 38 | 38 | 0.030049 | 0.026418 | 0.044998 | inv_center | 0.158901 | 19.851646 | 0.173398 | 2931 | 1.000000 | 0.832911 | 0.049335 |
| imu_pose_velocity_9dof_hospital_v1_008000 | 38 | 38 | 0.030029 | 0.026328 | 0.045016 | inv_center | 0.158911 | 19.851699 | 0.173412 | 2931 | 1.000000 | 0.820111 | 0.048672 |

Best ATE RMSE: `imu_pose_velocity_9dof_hospital_v1_008000` = 0.030029 m

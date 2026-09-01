# IMU Experiment Comparison

Lower is better for error metrics. Higher is better for `1px`.

## mean

| metric | baseline | rotation_only | full_long | tuned_v1 | full_loss_v1 | best |
|---|---:|---:|---:|---:|---:|---|
| rot_error | 0.912252 | 1.126652 | 1.070575 | 1.027250 | 1.068934 | baseline |
| tr_error | 0.048148 | 0.056585 | 0.053395 | 0.052337 | 0.052064 | baseline |
| residual | 6.348338 | 6.350308 | 6.372945 | 6.337441 | 6.469003 | tuned_v1 |
| f_error | 17.739829 | 18.856796 | 18.227516 | 17.341918 | 18.481078 | tuned_v1 |
| 1px | 0.512226 | 0.503681 | 0.507270 | 0.506153 | 0.504243 | baseline |
| imu_rot_loss | 0.000000 | 0.290878 | 0.290274 | 0.288872 | 0.291623 | baseline |
| imu_full_loss | - | - | - | - | 1.011515 | full_loss_v1 |
| imu_pos_loss | - | - | - | - | 0.181958 | full_loss_v1 |
| imu_vel_loss | - | - | - | - | 8.816689 | full_loss_v1 |
| imu_bias_loss | - | - | - | - | 0.000403 | full_loss_v1 |
| imu_pos_error | - | - | - | - | 0.068195 | full_loss_v1 |
| imu_vel_error | - | - | - | - | 4.823060 | full_loss_v1 |
| imu_rot_error | 0.000000 | 10.440625 | 10.398883 | 10.363491 | 10.396425 | baseline |
| imu_bias_error | - | - | - | - | 0.003397 | full_loss_v1 |
| imu_conf_mean | - | 0.461609 | 0.611767 | 0.355897 | 0.305903 | - |
| imu_conf_min | - | 0.440756 | 0.512268 | 0.344607 | 0.269121 | - |
| imu_conf_max | - | 0.485019 | 0.714741 | 0.368912 | 0.349762 | - |
| imu_ba_weight | - | 0.020011 | 0.021069 | 0.049991 | 0.050323 | - |
| imu_bias_norm | - | 0.000787 | 0.071168 | 0.000010 | 0.000000 | - |
| imu_acc_bias_norm | - | - | 0.000000 | 0.000000 | 0.000000 | - |

## last5

| metric | baseline | rotation_only | full_long | tuned_v1 | full_loss_v1 | best |
|---|---:|---:|---:|---:|---:|---|
| rot_error | 1.188196 | 1.393054 | 1.227409 | 1.402965 | 1.193734 | baseline |
| tr_error | 0.060655 | 0.072921 | 0.063157 | 0.068331 | 0.055625 | full_loss_v1 |
| residual | 6.209673 | 6.263985 | 6.424851 | 6.297095 | 6.480906 | baseline |
| f_error | 17.775335 | 19.930740 | 18.854074 | 18.290810 | 16.118433 | full_loss_v1 |
| 1px | 0.516878 | 0.495752 | 0.510408 | 0.505155 | 0.506101 | baseline |
| imu_rot_loss | 0.000000 | 0.279994 | 0.281626 | 0.279147 | 0.280428 | baseline |
| imu_full_loss | - | - | - | - | 1.030893 | full_loss_v1 |
| imu_pos_loss | - | - | - | - | 0.201522 | full_loss_v1 |
| imu_vel_loss | - | - | - | - | 9.179291 | full_loss_v1 |
| imu_bias_loss | - | - | - | - | 0.000412 | full_loss_v1 |
| imu_pos_error | - | - | - | - | 0.070920 | full_loss_v1 |
| imu_vel_error | - | - | - | - | 5.126524 | full_loss_v1 |
| imu_rot_error | 0.000000 | 10.094532 | 10.115122 | 10.082235 | 10.090278 | baseline |
| imu_bias_error | - | - | - | - | 0.003441 | full_loss_v1 |
| imu_conf_mean | - | 0.461592 | 0.595619 | 0.355272 | 0.298322 | - |
| imu_conf_min | - | 0.441075 | 0.494236 | 0.344254 | 0.263711 | - |
| imu_conf_max | - | 0.484595 | 0.702311 | 0.367213 | 0.342608 | - |
| imu_ba_weight | - | 0.020011 | 0.021069 | 0.049991 | 0.050323 | - |
| imu_bias_norm | - | 0.000787 | 0.071168 | 0.000010 | 0.000000 | - |
| imu_acc_bias_norm | - | - | 0.000000 | 0.000000 | 0.000000 | - |

## last

| metric | baseline | rotation_only | full_long | tuned_v1 | full_loss_v1 | best |
|---|---:|---:|---:|---:|---:|---|
| rot_error | 3.641140 | 2.582173 | 2.885848 | 3.381896 | 2.514493 | full_loss_v1 |
| tr_error | 0.158750 | 0.117944 | 0.103323 | 0.162541 | 0.095398 | full_loss_v1 |
| residual | 8.672861 | 8.354167 | 8.867838 | 8.732921 | 8.835190 | rotation_only |
| f_error | 25.968039 | 23.551676 | 19.933725 | 28.733894 | 17.386654 | full_loss_v1 |
| 1px | 0.385200 | 0.415413 | 0.429716 | 0.400858 | 0.397776 | full_long |
| imu_rot_loss | 0.000000 | 0.296563 | 0.301820 | 0.294572 | 0.288718 | baseline |
| imu_full_loss | - | - | - | - | 1.121918 | full_loss_v1 |
| imu_pos_loss | - | - | - | - | 0.257175 | full_loss_v1 |
| imu_vel_loss | - | - | - | - | 10.157816 | full_loss_v1 |
| imu_bias_loss | - | - | - | - | 0.000503 | full_loss_v1 |
| imu_pos_error | - | - | - | - | 0.074958 | full_loss_v1 |
| imu_vel_error | - | - | - | - | 5.785866 | full_loss_v1 |
| imu_rot_error | 0.000000 | 10.642124 | 10.508552 | 10.372244 | 10.270179 | baseline |
| imu_bias_error | - | - | - | - | 0.004181 | full_loss_v1 |
| imu_conf_mean | - | 0.473901 | 0.667401 | 0.364810 | 0.319330 | - |
| imu_conf_min | - | 0.446284 | 0.577266 | 0.348867 | 0.282846 | - |
| imu_conf_max | - | 0.503175 | 0.757144 | 0.380098 | 0.372702 | - |
| imu_ba_weight | - | 0.020011 | 0.021069 | 0.049991 | 0.050323 | - |
| imu_bias_norm | - | 0.000787 | 0.071168 | 0.000010 | 0.000000 | - |
| imu_acc_bias_norm | - | - | 0.000000 | 0.000000 | 0.000000 | - |

## Quick Read

- `rot_error` mean: baseline=0.912252, rotation_only=1.126652, full_long=1.070575 (lower is better, full-baseline delta=-0.158323)
- `tr_error` mean: baseline=0.048148, rotation_only=0.056585, full_long=0.053395 (lower is better, full-baseline delta=-0.005247)
- `f_error` mean: baseline=17.739829, rotation_only=18.856796, full_long=18.227516 (lower is better, full-baseline delta=-0.487687)
- `1px` mean: baseline=0.512226, rotation_only=0.503681, full_long=0.507270 (higher is better, full-baseline delta=-0.004957)


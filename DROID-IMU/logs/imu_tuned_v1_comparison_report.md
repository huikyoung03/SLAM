# IMU Experiment Comparison

Lower is better for error metrics. Higher is better for `1px`.

## mean

| metric | baseline | rotation_only | full_200 | full_long | tuned_v1 | best |
|---|---:|---:|---:|---:|---:|---|
| rot_error | 0.912252 | 1.126652 | 1.005827 | 1.070575 | 1.012729 | baseline |
| tr_error | 0.048148 | 0.056585 | 0.053895 | 0.053395 | 0.055868 | baseline |
| residual | 6.348338 | 6.350308 | 6.392510 | 6.372945 | 6.410790 | baseline |
| f_error | 17.739829 | 18.856796 | 18.009088 | 18.227516 | 17.943004 | baseline |
| 1px | 0.512226 | 0.503681 | 0.503437 | 0.507270 | 0.502814 | baseline |
| imu_rot_loss | 0.000000 | 0.290878 | 0.289966 | 0.290274 | 0.290491 | baseline |
| imu_rot_error | 0.000000 | 10.440625 | 10.336385 | 10.398883 | 10.365131 | baseline |
| imu_conf_mean | - | 0.461609 | 0.472637 | 0.611767 | 0.356174 | - |
| imu_conf_min | - | 0.440756 | 0.448786 | 0.512268 | 0.344481 | - |
| imu_conf_max | - | 0.485019 | 0.498683 | 0.714741 | 0.369094 | - |
| imu_ba_weight | - | 0.020011 | 0.020016 | 0.021069 | 0.049991 | - |
| imu_bias_norm | - | 0.000787 | 0.002087 | 0.071168 | 0.000010 | - |
| imu_acc_bias_norm | - | - | 0.000000 | 0.000000 | 0.000000 | - |

## last5

| metric | baseline | rotation_only | full_200 | full_long | tuned_v1 | best |
|---|---:|---:|---:|---:|---:|---|
| rot_error | 1.188196 | 1.393054 | 1.379180 | 1.227409 | 0.975640 | tuned_v1 |
| tr_error | 0.060655 | 0.072921 | 0.074886 | 0.063157 | 0.058477 | tuned_v1 |
| residual | 6.209673 | 6.263985 | 6.290067 | 6.424851 | 6.357179 | baseline |
| f_error | 17.775335 | 19.930740 | 18.866235 | 18.854074 | 17.364314 | tuned_v1 |
| 1px | 0.516878 | 0.495752 | 0.493761 | 0.510408 | 0.511003 | baseline |
| imu_rot_loss | 0.000000 | 0.279994 | 0.279991 | 0.281626 | 0.277458 | baseline |
| imu_rot_error | 0.000000 | 10.094532 | 10.014785 | 10.115122 | 9.955622 | baseline |
| imu_conf_mean | - | 0.461592 | 0.473798 | 0.595619 | 0.355107 | - |
| imu_conf_min | - | 0.441075 | 0.449775 | 0.494236 | 0.343814 | - |
| imu_conf_max | - | 0.484595 | 0.498649 | 0.702311 | 0.367030 | - |
| imu_ba_weight | - | 0.020011 | 0.020016 | 0.021069 | 0.049991 | - |
| imu_bias_norm | - | 0.000787 | 0.002087 | 0.071168 | 0.000010 | - |
| imu_acc_bias_norm | - | - | 0.000000 | 0.000000 | 0.000000 | - |

## last

| metric | baseline | rotation_only | full_200 | full_long | tuned_v1 | best |
|---|---:|---:|---:|---:|---:|---|
| rot_error | 3.641140 | 2.582173 | 3.156629 | 2.885848 | 2.762367 | rotation_only |
| tr_error | 0.158750 | 0.117944 | 0.131641 | 0.103323 | 0.149043 | full_long |
| residual | 8.672861 | 8.354167 | 8.068176 | 8.867838 | 8.932014 | full_200 |
| f_error | 25.968039 | 23.551676 | 24.252808 | 19.933725 | 26.941647 | full_long |
| 1px | 0.385200 | 0.415413 | 0.412234 | 0.429716 | 0.402448 | full_long |
| imu_rot_loss | 0.000000 | 0.296563 | 0.300649 | 0.301820 | 0.291286 | baseline |
| imu_rot_error | 0.000000 | 10.642124 | 10.521613 | 10.508552 | 10.337562 | baseline |
| imu_conf_mean | - | 0.473901 | 0.486803 | 0.667401 | 0.365548 | - |
| imu_conf_min | - | 0.446284 | 0.453278 | 0.577266 | 0.347613 | - |
| imu_conf_max | - | 0.503175 | 0.519545 | 0.757144 | 0.383538 | - |
| imu_ba_weight | - | 0.020011 | 0.020016 | 0.021069 | 0.049991 | - |
| imu_bias_norm | - | 0.000787 | 0.002087 | 0.071168 | 0.000010 | - |
| imu_acc_bias_norm | - | - | 0.000000 | 0.000000 | 0.000000 | - |

## Quick Read

- `rot_error` mean: baseline=0.912252, rotation_only=1.126652, full_long=1.070575 (lower is better, full-baseline delta=-0.158323)
- `tr_error` mean: baseline=0.048148, rotation_only=0.056585, full_long=0.053395 (lower is better, full-baseline delta=-0.005247)
- `f_error` mean: baseline=17.739829, rotation_only=18.856796, full_long=18.227516 (lower is better, full-baseline delta=-0.487687)
- `1px` mean: baseline=0.512226, rotation_only=0.503681, full_long=0.507270 (higher is better, full-baseline delta=-0.004957)


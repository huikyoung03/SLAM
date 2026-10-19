# DROID-SLAM 기반 Visual-Inertial SLAM 수행 결과 보고

## 1. 목적 및 최종 모델

목표는 단안 DROID-SLAM의 visual tracking 및 dense reconstruction 구조를 유지하면서,
IMU를 결합하여 metric trajectory를 추정 가능한 Visual-Inertial SLAM으로 확장하는 것이다.

최종 모델은 camera image에서 FNet/CNet, correlation, shared ConvGRU를 거친다.
ConvGRU hidden state에서 visual update head와 IMU confidence head가 병렬로 갈라진다.

- Visual update head: flow/reprojection update 및 visual confidence W^r
- IMU confidence head: 1x1 Conv(128->1), spatial average pooling, sigmoid을 거쳐 edge별 scalar W^u_ij 출력
- IMU path: gyro/accelerometer -> preintegration(Delta R, Delta v, Delta p, bias Jacobian)
- Stage 1: visual pose/depth warm-up DBA
- VI initialization: scale, gravity, velocity, base bias 초기화
- Stage 2: visual reprojection factor와 IMU factor를 결합한 full 15DoF DBA

15DoF motion state는 pose 6DoF, velocity 3DoF, accelerometer bias correction 3DoF,
gyro bias correction 3DoF로 구성된다. Inverse depth는 DROID처럼 dense variable로
함께 최적화하지만 15DoF motion state에는 포함하지 않는다. Feature-metric factor는 사용하지 않았다.

IMU factor 영향은 다음과 같이 구성된다.

    E_IMU ∝ lambda_IMU * w^u_ij * ||r_IMU,ij||^2

- lambda_IMU = 0.002: 모든 IMU factor에 공통인 global weight
- fixed: w^u = 1
- tuned: 0.25 -> 0.50 -> 0.75 -> 1.00 수동 ramp
- learned: ConvGRU hidden state로부터 예측한 0~1 scalar

## 2. 데이터와 평가 원칙

- Stage-3 confidence 학습: 실제 IMU가 있는 EuRoC MH_03_medium, camera 10 Hz
- checkpoint 후보: 2k/4k/6k/8k step
- 선택 checkpoint: validation 결과를 기준으로 선택한 parity 2k
- 최종 confidence 평가: EuRoC V2_03_difficult, 10 Hz 입력 이미지 961장
- Primary final VIO table: common reconstruction frame 320개, GT match 307개
- Metric SE(3) ATE: GT로 scale을 보정하지 않고, 회전과 이동만 정렬하여 계산

서로 다른 common frame subset에서 계산한 ATE는 직접 섞지 않는다.
특히 original을 포함한 scale/PLY audit은 247 common reconstruction frame을 사용하므로
primary final VIO table과 별도로 해석한다.

## 3. 실험 진행과 결과

### 3.1 IMU 초기화-only

초기 실험에서는 IMU로 scale, gravity, velocity, bias의 초기값만 구하고 이후에는
visual update 위주로 동작시켰다.

- VI initialization 및 좌표계/scale 연결 경로는 확인했다.
- 그러나 visual tracking이 흔들리는 구간에서 IMU가 이후 pose/depth update에 계속
  관여하지 못했다.
- 따라서 IMU residual을 DBA 내부 factor로 직접 결합하는 구조로 확장했다.

이 단계는 최종 성능 baseline이 아니라 구조 검증 실험이다.

### 3.2 9DoF와 15DoF

9DoF는 pose와 velocity만 최적화하고 bias를 VI initialization 후 고정한다.
15DoF는 accel/gyro bias correction까지 keyframe state에 포함한다.

EuRoC V2_03, 동일 241 common reconstruction frame 기준:

| 조건 | Metric SE(3) ATE RMSE |
|---|---:|
| Fixed 9DoF | 0.388726 m |
| Fixed 15DoF two-stage | 0.380986 m |

15DoF는 9DoF 대비 0.007740 m, 약 1.99% 낮은 ATE를 보였다.
이는 bias를 state로 다루는 방향의 타당성을 보이지만, 개선 폭은 작다.

### 3.3 Two-stage와 global bias refinement

Two-stage는 visual warm-up 후 VI initialization을 수행하고, 이후 visual+IMU full
15DoF DBA를 적용한다.

Shared global bias를 별도로 재최적화하는 실험도 수행했다. 그러나 작은 matched V2_03
구간에서 pass-0 ATE는 0.222643 m, global-bias refine 후에는 0.226775 m로 악화됐다.
V1_03에서도 pass-0 0.021726 m, global-bias refine 0.021984 m로 개선되지 않았다.

따라서 global bias refinement는 최종 성능 향상 요소로 채택하지 않았다.
Stage-3 confidence 학습에서는 shared global bias를 고정해 confidence 효과를 분리했고,
full 15DoF DBA의 keyframe별 local bias correction은 유지했다.

### 3.4 IMU confidence: primary final result

모든 조건은 동일한 two-stage 15DoF, VI initialization, lambda_IMU=0.002, 입력 이미지,
IMU preintegration을 사용한다. 차이는 IMU confidence뿐이다.

| 조건 | IMU confidence | Metric SE(3) ATE RMSE | Fixed 대비 |
|---|---|---:|---:|
| Fixed 15DoF | w^u=1.0 | 0.349735 m | - |
| Manual tuned 15DoF | ramp, mean 0.516 | **0.337333 m** | **-0.012402 m, -3.55%** |
| Learned parity 2k 15DoF | edge별 learned, mean 0.855 | 0.337930 m | **-0.011805 m, -3.38%** |

해석:

- 모든 IMU factor를 항상 최대 confidence로 쓰는 fixed보다 tuned/learned가 낮은 ATE를 보였다.
- learned와 tuned의 차이는 0.000597 m, 약 0.60 mm다.
- learned가 tuned보다 우수하다고 말할 수는 없지만, 수동 schedule 없이 tuned에 근접했다.
- 현재 결과는 V2_03 단일 hold-out 중심이므로 다중 시퀀스 검증이 필요하다.

### 3.5 Original DROID와의 scale audit

Original DROID, tuned, learned의 공통 reconstruction frame 247개, 유효 GT match 246개를
별도로 평가했다.

| 조건 | Metric scale 사용 가능 | GT에 맞추기 위한 Sim(3) scale | Metric scale error |
|---|---|---:|---:|
| Original DROID | 불가 | 2.022707 | 정의 불가 |
| Tuned 15DoF | 가능 | 1.199069 | 19.9069% |
| Learned parity 2k 15DoF | 가능 | 1.199491 | 19.9491% |

Original은 단안이므로 absolute scale을 관측할 수 없다. 2.022707은 GT 비교를 위해
사후적으로 필요한 oracle scale이며 original의 metric scale error가 아니다.

Tuned/learned는 IMU로 metre 단위 scale을 추정하지만 GT에 맞추려면 약 1.199배 확대가
필요하다. 즉 trajectory 크기가 GT 대비 약 0.834배로 형성되는 경향이 있다.
따라서 scale은 관측/추정 가능해졌지만, 정확히 복원됐다고 말할 수는 없다.

### 3.6 Original 대비 visual shape

동일 247-frame audit의 Sim(3) ATE는 GT scale을 사후 보정한 visual shape metric이다.

| 조건 | Sim(3) ATE RMSE |
|---|---:|
| Original DROID | **0.015203 m** |
| Tuned 15DoF | 0.015673 m |
| Learned parity 2k 15DoF | 0.015868 m |

따라서 original보다 visual shape ATE가 좋아졌다고 주장할 수 없다.
제안 모델의 original 대비 핵심은 visual shape의 단순 향상이 아니라, IMU로
scale/velocity/bias를 state로 다루고 metric trajectory를 출력 가능하게 만든 구조적 확장이다.

### 3.7 PLY output

Original/tuned/learned PLY는 다음 조건으로 맞췄다.

- 세 run에 공통인 247 frame ID
- 동일 depth-consistency filter: threshold 0.005, multi-view support count >= 2
- 동일 point count: 2,414,289 points
- original/tuned는 seed 20261015로 균일 샘플링, learned는 기준 point count를 그대로 사용

PLY는 dense reconstruction과 filter 결과를 보여주는 정성 출력이다. Original은 non-metric
scale이므로 세 PLY를 하나의 좌표계에 겹쳐 절대 크기를 비교하면 안 된다.

## 4. 확인된 개선점과 한계

### 확인된 개선점

1. 단안 DROID에서 관측 불가능했던 absolute scale을 IMU로 추정 가능한 상태로 확장했다.
2. VI initialization, IMU preintegration, velocity/bias state, 15DoF DBA를 구현했다.
3. 동일 VIO 조건에서 tuned/learned confidence가 fixed보다 약 3.4~3.6% 낮은 ATE를 보였다.
4. metric trajectory, trajectory evaluation, filtered dense PLY를 생성하는 재현 가능한 경로를 구성했다.

### 한계

1. Original보다 visual shape ATE가 좋아졌다고 말할 수 없다.
2. Tuned/learned 모두 약 20% residual global scale mismatch가 남아 있다.
3. Learned confidence는 tuned와 유사하지만 tuned를 초과하지는 않았다.
4. Final confidence 결과는 V2_03 중심 단일 hold-out 결과다.
5. Full preintegration covariance whitening은 최종 성능 향상 요소로 채택하지 않았다.

## 5. 보고용 결론

기존 DROID-SLAM의 visual recurrent update 구조를 보존한 상태에서 IMU preintegration,
VI initialization, 15DoF DBA, scalar IMU confidence를 결합한 metric VIO 프로토타입을
구현했다. V2_03에서 tuned와 learned confidence는 fixed confidence 대비 각각 3.55%,
3.38%의 Metric SE(3) ATE 감소를 보였다. 다만 learned가 manual tuning을 초과하지는
않았고 약 20% residual global scale mismatch가 남아 있으므로, 다중 시퀀스 검증과
scale/bias calibration은 후속 개선 과제다.

## 6. ORB-SLAM3 결과 사용 원칙

별도 ORB-SLAM3 monocular-inertial 결과는 V1_03 ATE RMSE 0.014607 m, V2_03 ATE RMSE
0.035174 m이다. 그러나 입력 frame rate, matched pose 수, alignment/evaluation protocol이
현재 DROID-IMU 결과와 다르므로 confidence table과 직접 수치 경쟁으로 해석하지 않고,
외부 reference로 별도 제시한다.


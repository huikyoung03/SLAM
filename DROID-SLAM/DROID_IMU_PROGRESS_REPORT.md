# DROID-SLAM + IMU 결합 개발 및 실험 진행 보고서

작성 기준: 2026-09-29

대상 저장소: `/home/ubuntu/DROID-SLAM`
연구 범위: 단안 `cam0` 영상과 IMU를 사용하는 DROID-SLAM 기반 Visual-Inertial SLAM

## 1. 요약

본 과제의 목표는 원본 DROID-SLAM의 visual-only dense bundle adjustment (DBA)에
IMU 사전적분(preintegration) 제약을 실제 최적화 factor로 결합하는 것이다. 목표는
단순히 gyro로 pose를 사후 보정하는 것이 아니라, camera pose, velocity, dense inverse
depth를 visual reprojection residual 및 IMU residual과 함께 최적화하는 DROID+IMU
구조를 만드는 것이다.

현재 구현은 다음 상태에 도달했다.

1. Gyroscope뿐 아니라 accelerometer를 이용하는 rotation, velocity, position IMU
   residual이 local DBA 경로에 연결되어 있다.
2. 각 keyframe의 active motion state는 `T` (pose, 6DoF)와 `v` (velocity, 3DoF)이며,
   dense inverse depth `d`는 원본 DROID 방식대로 함께 최적화한다. 이를 편의상
   **9DoF local Visual-Inertial DBA**라고 부른다.
3. ConvGRU update module에 IMU confidence head를 추가했고, 학습된 confidence와
   학습된 IMU residual weight, preintegration covariance information을 factor weight에
   반영한다.
4. Visual-inertial initialization으로 scale, gravity, 초기 velocity, global
   accelerometer/gyroscope bias를 추정한다.
5. bias를 모든 keyframe의 15DoF state로 넣는 대신, local window에서 하나의 shared
   6D bias block `(b_a, b_g)`을 저주기로 보정하는 구조를 추가했다.

다만 현재까지의 정확도 결과는 다음처럼 해석해야 한다.

- 초기의 고정 weight IMU 실험은 성능 향상을 보이지 않았고, 일부 조건에서는 소폭
  악화됐다.
- 좌표축, velocity propagation, covariance weighting을 정리한 뒤에는 EuRoC의 두
  개발/held-out sequence에서 Sim(3) ATE가 약 `0.09%`, `0.17%` 개선된 결과가 있었다.
- 이 개선량은 작으며, 저장된 reconstruction keyframe 기준의 중간 평가 결과다.
  따라서 "IMU 결합의 동작 가능성과 작은 positive signal"은 확인했지만,
  "일반적으로 유의미한 성능 개선"을 아직 확정할 단계는 아니다.
- 현재 EuRoC V2_03 difficult 전체 입력에 대해 `original`, `9DoF fixed-bias`,
  `9DoF shared-bias`를 같은 조건에서 비교하는 실험이 진행 중이다. 이 결과가
  bias 보정의 실제 효과를 판단하는 첫 번째 핵심 결과가 된다.

이번 단계의 결론은 다음과 같다.

> DROID+IMU의 tightly coupled 구조와 학습 confidence 경로는 실제로 구현되어
> 동작하고 있다. 방향은 타당하다. 다만 최종 성능 주장은 full trajectory 기반의
> held-out EuRoC 평가와 shared-bias ablation이 끝난 뒤에 해야 한다.

## 2. 연구 목표와 범위

### 2.1 목표

단안 DROID-SLAM에서 visual information이 약해지는 구간, 예를 들어 빠른 회전,
짧은 baseline, 저텍스처, 단기 tracking 불안정 구간에서 IMU가 물리적 motion prior를
제공하도록 한다. 또한 IMU로 monocular scale ambiguity와 단기 velocity/position
drift를 완화하는 것이 목표다.

최종 활용 환경은 EuRoC만을 위한 모델이 아니라, 휴대폰에서 들어오는 단안 이미지와
IMU 입력이다. 따라서 데이터셋/실행 경로는 camera와 IMU timestamp가 들어오면
동일한 preintegration pipeline을 타도록 일반화해 두고, EuRoC는 개발 및 검증용
공식 데이터로 사용한다.

### 2.2 의도적으로 제외한 범위

본 단계는 DVI-SLAM 전체를 그대로 재현하는 작업이 아니다.

- Appearance feature branch와 feature-metric residual `E^f`는 현재 범위에서 제외했다.
- 모든 keyframe에 `T, v, b_a, b_g`를 독립적으로 두는 full 15DoF-per-keyframe VIO는
  계산량과 CUDA solver 변경 범위가 커서 채택하지 않았다.
- 현재 목적함수의 핵심은 `E^r + E^u`이며, IMU 결합 설계에서 DVI-SLAM의
  preintegration 및 learned confidence 개념을 참고한다.

## 3. 현재 모델 구조와 실행 흐름

### 3.1 원본 DROID-SLAM baseline

원본 DROID-SLAM은 선택된 keyframe의 pose `T`와 dense inverse depth `d`를
reprojection residual로 반복 최적화한다. `MotionFilter`가 모든 입력 프레임 중
motion이 충분한 프레임을 keyframe 후보로 고르고, `FactorGraph`와 ConvGRU가
correspondence/visual weight를 업데이트하며, CUDA DBA가 pose와 depth를 갱신한다.

```text
image stream
  -> MotionFilter (keyframe selection)
  -> feature/correlation + ConvGRU
  -> visual reprojection factors
  -> local DBA: pose T + dense inverse depth d
  -> global backend + trajectory filler
```

본 보고서의 visual-only baseline은 IMU 옵션을 끈 original `droid.pth`이다.

### 3.2 현재 DROID+IMU 구조

```text
cam0 image stream                         raw IMU stream
       |                                      |
       |                               timestamp synchronization
       |                                      |
       |                          preintegration per image interval
       |                         (Delta R, Delta v, Delta p, covariance,
       |                          bias Jacobians)
       v                                      v
MotionFilter -> selected keyframes -> DepthVideo IMU state buffer
       |                                      |
       v                                      |
DROID feature/correlation + ConvGRU          |
       |                                      |
       +-- visual correspondence/weight       |
       +-- learned IMU confidence ------------+
                                              v
                    VI initialization: scale, gravity, initial v, b_a, b_g
                                              |
                                              v
               local 9DoF DBA over {T_i, v_i, d_i}
                 visual reprojection residual E^r(T,d)
                 IMU rotation/velocity/position residual E^u(T,v; g,b_a,b_g)
                 learned confidence x learned factor weight x covariance information
                                              |
                           periodic shared-bias 6D refinement of {b_a, b_g}
                                              |
                            one post-bias local DBA re-linearization pass
                                              v
                         global backend -> filled trajectory / reconstruction / PLY
```

### 3.3 최적화 상태와 목적함수

각 active keyframe의 state는 다음과 같다.

\[
X_i = (T_i, v_i, d_i)
\]

- `T_i`: camera pose, 6DoF
- `v_i`: world-frame velocity, 3DoF
- `d_i`: dense inverse depth

여기서 `9DoF`는 pose와 velocity만 의미하며, dense depth는 pixel 단위 변수로 기존
DROID의 Schur complement 구조를 유지한다. Scale `s`, gravity `g`, accelerometer
bias `b_a`, gyroscope bias `b_g`는 VI initialization에서 초기화한다.

local DBA의 개념적 목적함수는 다음과 같다.

\[
E = E^r(T,d) + E^u(T,v; g,b_a,b_g) + E_{prior}
\]

\[
E^u = \sum_{(i,j)} \rho\left(
w^u_{ij}
\left[ r_R^{ij},\; \lambda_v r_v^{ij},\; \lambda_p r_p^{ij}\right]
\right)
\]

`r_R`, `r_v`, `r_p`는 각각 preintegrated IMU rotation, velocity, position 측정과
현재 상태가 만드는 prediction의 차이다. 실제 solver에는 residual 값만 더하는 것이
아니라, 각 residual의 Jacobian에서 만들어지는 `J^T W J`와 `J^T W r` block을 visual
DBA system에 추가한다.

각 IMU edge의 effective weight는 다음 요소의 조합이다.

\[
w^u_{ij} = w_{global} \cdot c_{ij}^{GRU} \cdot W_{cov,ij}
\cdot w_{p/v}
\]

- `w_global`: checkpoint에 저장된 학습 가능한 global IMU factor scale
- `c_ij^GRU`: ConvGRU IMU confidence head가 출력한 신뢰도
- `W_cov,ij`: preintegration covariance에서 얻은 information weighting
- `w_p/v`: position/velocity residual의 학습 가능한 상대 가중치

즉, runtime에서 모든 IMU factor를 같은 수동 상수로 강제하지 않고, 학습된 global
weight와 edge별 confidence로 적응적으로 반영하는 구조다.

## 4. 구현한 핵심 모듈

| 영역 | 주요 파일 | 현재 구현 내용 |
| --- | --- | --- |
| IMU 입력/사전적분 | `tools/imu_preintegrate.py`, `droid_slam/imu_prior.py`, `droid_slam/data_readers/imu_preintegration.py` | 이미지 timestamp 사이의 IMU를 적분하여 `Delta R, Delta v, Delta p`, covariance, bias Jacobian을 생성/로딩 |
| 상태 버퍼 | `droid_slam/depth_video.py` | keyframe별 IMU delta, covariance/information, bias Jacobian, velocity, calibration 정보를 함께 저장 |
| IMU factor | `droid_slam/geom/imu_factor.py`, `droid_slam/geom/ba.py` | rotation/velocity/position residual 및 analytic Jacobian을 local pose-velocity DBA system에 결합 |
| factor graph/runtime | `droid_slam/factor_graph.py`, `droid_slam/droid_frontend.py`, `droid_slam/droid_backend.py` | local IMU factor 추가, VI initialization, learned confidence 전달, shared-bias update와 post-update DBA 실행 |
| network | `droid_slam/droid_net.py` | ConvGRU hidden state로부터 IMU confidence map/temporal confidence를 예측 |
| 학습 | `train.py`, `droid_slam/geom/losses.py` | full preintegration loss, confidence NLL/regularizer, learned residual weight prior |
| 데이터 | `droid_slam/data_readers/tartan_v2.py`, `droid_slam/data_readers/euroc.py` | TartanAir V2 학습용 IMU/GT 로더와 EuRoC cam0 준비 경로 |
| 평가 | `tools/evaluate_euroc_recon.py`, `tools/run_euroc_heldout_validation.sh` | baseline/IMU ablation, Sim(3), metric SE(3), RPE, IMU usage/debug 비교 |

추가로, TartanAir에서 visual pose 좌표축은 변환했지만 gyro/accelerometer 축은 변환하지
않던 문제를 수정했다. 이 수정 후 IMU relative rotation과 GT pose relative rotation의
평균 불일치는 약 `6.98 deg`에서 `0.385 deg`로 감소했다. 학습 unroll에서 DBA가
갱신한 velocity를 다음 iteration에 다시 pose difference로 덮어쓰던 문제도 수정하여,
학습과 runtime의 motion-state propagation을 일치시켰다.

## 5. 학습 구조와 loss

TartanAir V2의 dense depth, GT pose, IMU를 사용해 pretrained `droid.pth`에서
IMU 관련 부분을 fine-tuning한다. 현재 clean fine-tuning에서는 pretrained visual
backbone을 주로 고정하고, IMU confidence head 및 IMU factor 관련 weight를 중심으로
학습한다. 이렇게 하면 기존 visual tracking 능력을 급격히 훼손하지 않으면서 IMU를
언제 얼마나 믿을지 학습하게 된다.

학습의 개념적 total loss는 다음과 같다.

\[
L = w_{geo}L_{geo} + w_{res}L_{res} + w_{flow}L_{flow}
  + w_{imu}L_{preint}
  + w_{conf}L_{conf}
  + w_{prior}L_{weight-prior}
\]

| 항 | 역할 |
| --- | --- |
| `L_geo` | GT pose와 추정 pose의 geodesic pose loss |
| `L_res` | visual reprojection residual supervision |
| `L_flow` | 기존 DROID의 dense flow/depth supervision. IMU를 넣어도 visual geometry 학습 기반으로 유지 |
| `L_preint` | `r_R`, `r_v`, `r_p`의 robust Smooth L1 기반 full preintegration loss |
| `L_conf` | IMU confidence가 무조건 1 또는 0으로 포화되는 것을 막고, residual 크기에 맞는 uncertainty를 학습시키는 항 |
| `L_weight-prior` | 학습 가능한 position/velocity residual scale이 임의로 붕괴하거나 발산하는 것을 방지하는 prior |

중요한 구분은 다음과 같다.

- `w_imu=0.05`는 **학습 중 IMU supervision loss의 multiplier**다.
- 이는 runtime에서 IMU factor를 고정 `0.05`로 적용한다는 의미가 아니다.
- runtime factor strength는 checkpoint의 learned global IMU weight, GRU confidence,
  covariance information, learned position/velocity relative weight의 조합으로 결정된다.

현재 shared-bias local refinement는 runtime block-coordinate optimizer로 추가된
상태다. 따라서 다음 fine-tuning 단계에서는 raw/synthetic bias perturbation을 학습
데이터에 주입해 confidence가 bias-contaminated interval도 구분하도록 확장하는 것이
필요하다.

## 6. 데이터셋과 검증 분리

| 용도 | 데이터 | 역할 |
| --- | --- | --- |
| 학습 | TartanAir V2 Hospital subset, `lcam_front` 단안 | dense depth, GT pose, IMU가 함께 있어 DROID visual loss와 IMU loss를 동시에 학습 |
| 개발/validation | EuRoC V2_01 easy, `cam0` | checkpoint 후보 2k/4k/6k/8k 중 선택 및 구조 점검 |
| held-out test | EuRoC V2_03 difficult, `cam0` | 선택 후 변경하지 않은 모델의 cross-sequence 평가 |
| exploratory/debug | EuRoC V1_03 difficult, `cam0` | 초기 연결, ablation, stride 2 실험 |
| 향후 domain test | 직접 촬영한 휴대폰 RGB+IMU | 최종 목표 도메인에서 timestamp, extrinsic, noise 특성을 검증 |

학습 데이터와 평가 데이터를 분리하는 이유는 checkpoint, threshold, factor weight를
같은 sequence에서 반복 조정한 뒤 그 sequence의 수치만 보고하면 과적합된 선택이 될 수
있기 때문이다. 현재는 TartanAir로 학습하고 EuRoC로 domain-shift 검증을 한다.
EuRoC 안에서도 V2_01은 validation, V2_03은 held-out으로 구분한다. 휴대폰 데이터는
마지막에 별도의 domain transfer test로 남긴다.

## 7. 수행한 실험과 결과

### 7.1 실험 A: IMU 연결 및 초기 ablation

**목적**: IMU 파일을 읽는 것과 IMU residual이 실제 DBA에 들어가는 것을 구분하고,
rotation-only / full IMU prior의 기본 동작을 확인했다.

**조건**: EuRoC V1_03, 초기 450-frame 범위, 동일한 43 reconstruction keyframe,
Sim(3)-aligned ATE.

| 모델 | ATE RMSE [m] | IMU 사용 edge | 해석 |
| --- | ---: | ---: | --- |
| Original DROID | 0.069182 | 0 | visual-only baseline |
| IMU data only | 0.069182 | 0 | IMU를 읽기만 하면 결과가 바뀌지 않음 |
| Rotation fixed prior | 0.069185 | 0 | 당시 rotation prior는 사실상 active factor가 아님 |
| Full IMU fixed | 0.069191 | 247, ratio 0.310 | factor는 사용됐으나 개선 없음 |
| Full IMU + learned confidence/info | 0.069200 | 247, ratio 0.310 | 초기 formulation에서는 소폭 악화 |

**결론**: "IMU 데이터가 들어간다"와 "IMU가 optimizer를 바꾼다"를 분리해 확인했다.
초기 실험은 성능 개선 실험이 아니라 연결/ablation 실험이었으며, 고정 weight만으로는
효과가 없다는 결과가 나왔다.

### 7.2 실험 B: 휴대폰 목표 frame rate에 가까운 stride 2 시험

**목적**: 향후 약 10 FPS 이미지 입력을 가정해, image stride 2 조건에서도 IMU를
200 Hz 수준의 고주파 interval data로 사전적분해 사용할 수 있는지 확인했다.

**조건**: EuRoC V1_03, 입력 첫 225 frame, stride 2, 43 reconstruction keyframe.

| 모델 | ATE RMSE [m] | 변화 |
| --- | ---: | ---: |
| Original DROID | 0.071706 | - |
| Full IMU fixed | 0.071703 | +0.004% |
| Full IMU + learned confidence/info | 0.071695 | +0.016% |

**결론**: stride 2에서도 camera-IMU interval integration과 runtime 실행은 가능했다.
그러나 이 수치는 매우 작고 짧은 prefix이므로 정확도 개선 근거로는 부족하다.

### 7.3 실험 C: 좌표계/velocity propagation 수정 후 9DoF IMU 평가

**목적**: TartanAir pose와 raw IMU의 축 불일치, training unroll velocity reset,
covariance/information weighting 문제를 수정한 뒤 IMU factor가 실제로 유효한지
확인했다.

**조건**: EuRoC common reconstruction frame 기준, learned confidence 및 IMU
noise/covariance weighting 사용.

| Sequence | 모델 | 공통 pose 수 | Sim(3) ATE RMSE [m] | 변화 | IMU factor 사용 |
| --- | --- | ---: | ---: | ---: | ---: |
| V2_01 easy | Original DROID | 38 | 0.030052 | - | 0 |
| V2_01 easy | 9DoF IMU | 38 | 0.030024 | +0.093% | 2931, ratio 1.000 |
| V2_03 difficult | Original DROID | 116 | 0.047936 | - | 0 |
| V2_03 difficult | 9DoF IMU | 116 | 0.047857 | +0.165% | 11155, ratio 0.998 |

평균 learned IMU confidence는 각각 `0.954`, `0.964`였고, learned global IMU
weight는 약 `0.052`였다.

**결론**: IMU factor는 거의 모든 temporal edge에서 실제로 사용됐고, 두 sequence에서
작지만 일관된 positive signal이 나타났다. 다만 0.1% 수준의 차이는 sequence variance,
evaluation subset, tracker keyframe 선택에 민감할 수 있으므로 통계적으로 강한 개선
주장으로 확대하면 안 된다.

### 7.4 실험 D: TartanAir Hospital fine-tuning과 checkpoint 선택

**목적**: original `droid.pth`의 visual 성능을 보존하면서 IMU confidence와
factor weight를 학습한다.

**학습 조건**:

- TartanAir V2 Hospital environment의 22 trajectory, 약 32,340 frame
- `lcam_front` 단안, GT pose, dense depth, gyro/acc 사용
- fixed-bias 9DoF DBA, full preintegration loss, covariance-aware weighting
- checkpoint: 2k/4k/6k/8k step
- pretrained non-IMU visual backbone은 주로 frozen

**checkpoint selection 조건**: EuRoC V2_01에서 IMU 후보끼리 공통-frame metric
SE(3) ATE RMSE로 ranking했다. baseline에는 monocular metric scale이 없으므로
baseline과의 절대 metric SE(3) 우열 비교가 아니라 IMU candidate selection용 지표다.

선택 checkpoint는 다음과 같다.

```text
checkpoints/pose_velocity_9dof_axisfix_hospital_wimu0.05_006000.pth
```

| 항목 | 값 |
| --- | ---: |
| selected IMU candidate metric SE(3) ATE | 0.079226 m |
| metric origin ATE | 0.095231 m |
| metric RPE 1 s | 0.013668 m |
| selected candidate Sim(3) ATE | 0.016109 m |
| original DROID Sim(3) ATE | 0.016053 m |

**결론**: `006000`은 IMU 후보 중 선택된 checkpoint이지만, V2_01에서 visual shape
metric 기준 baseline을 이겼다는 뜻은 아니다. 이 결과는 checkpoint ranking을 고정하고
다른 sequence에서 평가해야 한다는 이유를 명확히 보여준다.

### 7.5 실험 E: V2_03 cross-sequence preliminary test

**목적**: TartanAir Hospital 학습 및 V2_01 selection과 다른 EuRoC V2_03에서
cross-sequence 가능성을 확인했다.

| 모델 | 공통 pose 수 | Sim(3) ATE [m] | Sim(3) RPE 1 s [m] | Sim(3) RPE 5 s [m] |
| --- | ---: | ---: | ---: | ---: |
| Original DROID | 116 | 0.017100 | 0.016325 | 0.028775 |
| IMU 8k candidate | 116 | 0.017037 | 0.016296 | 0.028591 |

IMU candidate는 physical scale을 생성했고, metric SE(3) ATE는 `0.166526 m`,
scale residual은 `7.84%`였다. original monocular DROID에는 metric scale 자체가
관측되지 않으므로 해당 metric 값은 직접 비교 대상이 아니다.

**결론**: ATE와 단기/중기 RPE가 각각 약 `0.37%`, `0.18%`, `0.64%` 개선된
preliminary signal이 있었다. 하지만 이 결과는 8k candidate와 source sequence의
일부 reconstruction keyframe을 쓴 중간 결과이며, final held-out 결론으로 사용하지
않는다.

### 7.6 실험 F: Jacobian 및 synthetic bias validation

**목적**: shared-bias refinement를 실제 데이터에 적용하기 전에 analytic Jacobian과
bias recovery 방향이 수치적으로 맞는지 확인했다.

**방법**:

- rotation, velocity, position preintegration residual의 bias Jacobian에 대해
  finite-difference check 수행
- 알려진 synthetic accelerometer/gyroscope bias를 주입한 뒤 shared 6D block이
  보정 방향을 회복하는지 확인
- 충분한 rotation excitation이 없는 window에서는 bias update를 거절하는지 확인

**결과**: `tools/test_shared_bias_refinement.py`가 통과했다. 즉, shared bias block의
analytic Jacobian과 synthetic recovery/observability rejection은 동작한다.

**결론**: 이것은 numerical implementation validation이며, 실제 EuRoC에서 bias
trajectory와 ATE/RPE가 개선됐다는 증명은 아니다. 그 실증은 아래 진행 중인 실험에서
확인한다.

### 7.7 실험 G: 현재 진행 중인 V2_03 full-input shared-bias ablation

**목적**: bias를 VI initialization 값으로 고정하는 방식과, low-frequency shared bias
refinement를 추가하는 방식의 차이를 같은 held-out sequence에서 확인한다.

**데이터/조건**:

- EuRoC `V2_03_difficult`, `cam0`, 1,922 image
- `stride=2`로 약 961 tracking input
- 동일한 selected 6k checkpoint, 동일 calibration, 동일 IMU prior
- 학습이나 runtime weight tuning 없이 세 configuration을 순차 실행

| 순서 | Configuration | 의미 |
| ---: | --- | --- |
| 1 | `original_droid` | visual-only original `droid.pth` |
| 2 | `imu_9dof_fixed` | `T,v,d` local DBA, VI-init global bias 고정 |
| 3 | `imu_9dof_shared_bias` | fixed 구조에 shared `(b_a,b_g)` low-frequency refinement 추가 |

실행 출력 경로는 다음이다.

```text
outputs/euroc_v2_03_shared_bias_20260929_132338/
```

작성 시점에는 original run이 완료됐고 `imu_9dof_fixed`가 실행 중이며 shared-bias run과
최종 summary는 아직 생성되지 않았다. 따라서 이 보고서에는 이 실험의 수치를 넣지
않는다.

## 8. Bias 오차에 대한 대응 방안

### 8.1 문제 정의

gyro bias는 rotation preintegration error를, accelerometer bias는 velocity 및
position preintegration error를 시간에 따라 누적시킨다. bias를 영구 고정하면 특히
긴 sequence와 휴대폰 센서에서 IMU factor가 잘못된 motion prior가 될 수 있다.

반대로 모든 keyframe마다 독립적인 `b_a`, `b_g`를 최적화하면 pose+velocity+bias로
keyframe당 15DoF motion state가 되어 local reduced system과 runtime cost가 크게
증가한다.

### 8.2 채택한 해결책: 초기화 + shared low-frequency 6D refinement

현재 해결책은 full 15DoF per-keyframe 상태가 아니라 다음 두 단계다.

1. **VI initialization**: visual trajectory와 IMU preintegration을 사용해 initial
   scale, gravity, velocity, global `b_a`, global `b_g`를 추정한다.
2. **Runtime local refinement**: local DBA의 main active state는 계속 `T,v,d`로
   유지한다. 다만 일정 interval마다 local window 전체에 공유되는 하나의 6D bias
   block `(b_a,b_g)`을 별도로 갱신한다.

이는 다음의 block-coordinate optimization이다.

\[
\{T_i,v_i,d_i\}_{i\in window}
\quad \leftrightarrow \quad
\{b_a,b_g\}
\]

shared bias update는 pose/velocity/depth CUDA DBA와 번갈아 수행한다. bias step이
accepted되면 곧바로 한 번의 local 9DoF DBA를 다시 수행하여 `T,v,d`가 새 bias로
corrected preintegration residual에 반응하도록 한다.

### 8.3 안정성 장치

shared-bias block은 아무 window에서나 업데이트하지 않는다.

- preintegration이 저장한 `J_R_bg`, `J_v_ba`, `J_v_bg`, `J_p_ba`, `J_p_bg`를 사용한다.
- GRU learned confidence 및 covariance-derived information을 기존 local IMU factor와
  같은 방식으로 반영한다.
- Huber IRLS robust cost를 사용한다.
- bias random-walk/noise density 기반 trust prior를 사용한다.
- minimum edge 수, time duration, rotation excitation, Hessian condition number를
  확인한다.
- step 크기와 bias norm을 제한하며 robust objective가 실제로 감소할 때만 accept한다.

따라서 단순히 bias를 매 프레임마다 자유롭게 움직이는 방식이 아니라, 관측 가능한
motion 구간에서만 저주기로 보정하는 방식이다.

### 8.4 현재 상태와 남은 검증

구조 구현 및 synthetic/Jacobian test는 완료됐다. 실제 EuRoC에서의 효과는 현재
`fixed-bias vs shared-bias` full-input ablation으로 검증 중이다. 이 실험이 끝난 뒤
다음 네 가지를 함께 비교한다.

1. ATE/RPE
2. preintegration IMU residual의 전후 변화
3. shared `b_a`, `b_g` trajectory와 update acceptance rate
4. tracking failure/success 및 runtime overhead

이 결과가 좋아야 bias 대응이 실제 성능 개선으로 이어졌다고 말할 수 있다. 그렇지
않다면 bias parameterization 자체보다 camera-IMU time offset, extrinsic, raw IMU
noise model 또는 initialization observability를 먼저 점검해야 한다.

## 9. 평가 지표의 해석과 현재 한계

### 9.1 왜 ATE 하나로 결론 내리면 안 되는가

단안 visual-only DROID는 absolute metric scale이 관측되지 않는다. 따라서 IMU 모델은
metric trajectory를 생성하지만 baseline은 scale을 알 수 없다는 비대칭이 있다.

현재 다음 지표를 함께 본다.

| 지표 | 목적 | 해석상 주의점 |
| --- | --- | --- |
| Sim(3) ATE / ATE-scale | global scale/rigid alignment 후 visual reconstruction shape 비교 | monocular baseline과 IMU 모델을 공통으로 비교할 수 있으나, physical scale 정확도는 제거됨 |
| Sim(3) RPE 1 s, 5 s | local drift/relative motion 비교 | oracle global scale alignment 후 값임 |
| metric SE(3) ATE/RPE | IMU가 제공하는 actual metric scale/trajectory 정확도 | visual-only baseline에는 직접 대응하는 metric scale이 없음 |
| IMU usage/confidence/residual | IMU factor가 실제로 쓰였는지 및 잘못 과신하지 않는지 확인 | trajectory accuracy와 함께 해석해야 함 |

원본 DROID 논문도 EuRoC에서 ATE[m]를 보고하고, DVI-SLAM은 monocular VIO에서
`ATE-scale`과 fixed/tuned/learned IMU factor ablation을 사용한다. 현재 Sim(3) ATE는
그와 같은 계열의 shape metric으로 사용하고 있다. 그러나 현재 evaluator는 저장된
reconstruction keyframe을 중심으로 비교하므로, 논문과 동일한 최종 protocol은 아직
아니다. [DROID-SLAM 논문](https://papers.nips.cc/paper_files/paper/2021/file/89fcd07f20b6785b92134bd6c1d0fa42-Paper.pdf),
[DVI-SLAM 논문](https://arxiv.org/pdf/2309.13814)

### 9.2 반드시 보완할 평가 절차

`demo.py`는 종료 시 `PoseTrajectoryFiller`를 거친 full `traj_est`를 생성하지만,
현재 `save_reconstruction()`은 주로 map keyframe pose를 저장한다. 따라서 지금까지의
수치는 유용한 중간 ablation이지만, final paper-style result로 사용하기 전에 다음을
수행해야 한다.

1. `traj_est`와 정확한 source timestamp를 별도로 저장한다.
2. tracking stride가 2여도 filler에는 원래 image stream을 제공해 full trajectory를
   생성한다.
3. EuRoC GT와 timestamp association 후 `evo_ape`, `evo_rpe` 또는 동등한 evaluator로
   full trajectory ATE/RPE를 계산한다.
4. V2_01은 validation으로 고정하고, V2_03 및 추가 EuRoC sequence에서
   baseline/fixed-bias/shared-bias를 동일한 input, calibration, checkpoint로 비교한다.
5. sequence별 수치와 평균/median, success rate를 같이 보고, test sequence에서
   runtime weight/threshold를 다시 조정하지 않는다.

이 보완이 끝나면 현재의 small positive signal이 우연인지, 재현 가능한 개선인지
판단할 수 있다.

## 10. 왜 아직 DVI-SLAM 수준의 뚜렷한 개선이 보이지 않는가

DVI-SLAM은 learned IMU confidence가 fixed weight보다 유리할 수 있음을 보인 중요한
참고 사례지만, 현재 프로젝트는 의도적으로 더 좁은 IMU-only 구조다. 따라서 논문의
전체 성능 향상을 그대로 기대하면 안 된다.

가능한 원인은 다음과 같다.

1. 현재는 `E^f` feature-metric residual 없이 `E^r + E^u`만 사용한다.
2. 학습 데이터가 Hospital 한 synthetic environment에 집중돼 있으며, 실제 EuRoC 및
   향후 휴대폰 센서의 motion/noise distribution과 다르다.
3. 초기 fixed-bias 구조는 bias drift를 충분히 흡수하지 못했다. shared-bias block은
   방금 추가되어 실제 sequence 결과를 검증 중이다.
4. preintegration weight, covariance, camera-IMU coordinate/extrinsic, timestamp offset은
   작은 오차만 있어도 accelerometer 기반 position residual에 큰 영향을 준다.
5. DROID의 visual geometry 자체가 강하기 때문에 안정적인 visual 구간에서는 IMU가
   큰 개선을 만들 여지가 작고, 이득은 visual degradation 구간에서 주로 나타날 수 있다.
6. 현재 일부 결과는 map keyframe 평가/partial prefix 기준이므로 full trajectory의
   failure recovery 또는 long-term drift 효과를 충분히 포착하지 못한다.

따라서 "IMU가 쓸모없다"는 결론이 아니라, **현재 구조의 correctness와 calibration,
training diversity, 평가 protocol을 먼저 고정해야 작은 IMU 이득을 신뢰성 있게
측정할 수 있다**는 결론이 더 정확하다.

## 11. 향후 진행 계획

### 단계 1: 현재 shared-bias 실험 완료 및 분석

현재 V2_03 full-input three-way run을 끝낸다. 결과가 나오면 fixed-bias와
shared-bias의 ATE/RPE, bias update history, IMU residual, runtime을 비교한다.

### 단계 2: full trajectory 평가 정식화

`traj_est` export와 EuRoC timestamp-aware `evo` evaluation을 추가한다. 이후 결과는
map keyframe 중간 지표와 full trajectory 최종 지표를 분리해 보고한다.

### 단계 3: checkpoint/ablation 동결

V2_01 validation에서 checkpoint를 한 번 선택한 뒤 변경하지 않는다. V2_03 및 추가
held-out EuRoC sequence에서 다음 세 모델을 동일 조건으로 비교한다.

1. Original DROID baseline
2. 9DoF fixed-bias IMU
3. 9DoF shared-bias IMU

### 단계 4: bias-aware fine-tuning

shared-bias 구조가 안정적이면 TartanAir 학습에 synthetic gyro/accelerometer bias 및
random-walk perturbation을 주입한다. 이렇게 하면 GRU confidence가 단순한 ideal-IMU
신뢰도가 아니라 bias-contaminated interval의 uncertainty도 학습하게 된다.

### 단계 5: 학습 다양성 확대

현재 Hospital subset만으로 충분한 일반화를 기대하기 어렵다. 저장공간이 허용되는
범위에서 서로 다른 TartanAir V2 environment/difficulty/trajectory를 추가하고,
EuRoC validation/test split은 고정한다.

### 단계 6: 휴대폰 적용 전 calibration 검증

모델 구조가 안정화된 뒤 실제 휴대폰 데이터에서 다음을 별도로 검증한다.

- camera-IMU time offset
- camera-IMU rotation/translation extrinsic
- IMU noise density와 bias random walk
- 10 FPS image + 고주파 IMU 입력에서 preintegration interval 일관성

이 단계는 EuRoC에 과적합하지 않고 최종 휴대폰 입력에 적용하기 위한 필수 절차다.

## 12. 교수님께 보고할 핵심 결론

1. 원본 DROID-SLAM baseline은 IMU를 사용하지 않는 official visual-only `droid.pth`다.
2. 현재 모델은 gyro만의 사후 pose 보정이 아니라, gyro와 accelerometer의
   preintegrated rotation/velocity/position residual을 DBA Jacobian/Hessian 경로에
   결합한 9DoF visual-inertial 구조다.
3. IMU confidence는 ConvGRU가 예측하고, runtime에서는 learned global/relative
   weight 및 covariance information과 함께 IMU factor 강도를 결정한다.
4. 초기 고정-weight 실험은 개선이 없었지만, 좌표계/velocity/covariance를 수정한 뒤
   두 EuRoC sequence에서 작지만 일관된 positive signal이 나타났다. 이는 구조적
   feasibility를 뒷받침하지만, 아직 강한 accuracy claim의 근거는 아니다.
5. bias 문제는 full 15DoF-per-keyframe 대신 **VI initialization + shared 6D
   low-frequency bias refinement**로 해결 중이다. analytic Jacobian 및 synthetic
   validation은 통과했고, 실제 EuRoC full-input ablation이 진행 중이다.
6. 다음 우선순위는 모델 복잡도를 더 늘리는 것이 아니라, full trajectory 기반의
   held-out 평가를 정식화하고 fixed-bias/shared-bias 차이를 공정하게 검증하는 것이다.
7. 이 평가에서 Sim(3) ATE/RPE, metric scale/RPE, IMU residual, tracking success가
   동시에 유지 또는 개선되면 DROID+IMU 방식의 실질적 성능 개선을 주장할 수 있다.

## 13. 주요 산출물

| 항목 | 위치 |
| --- | --- |
| 9DoF 주차 보고서 | `WEEKLY_REPORT_9DOF_IMU.md` |
| 현재 보고서 | `DROID_IMU_PROGRESS_REPORT.md` |
| 9DoF noise fine-tuning script | `tools/run_9dof_noise_finetune.sh` |
| held-out validation script | `tools/run_euroc_heldout_validation.sh` |
| shared bias Jacobian/synthetic test | `tools/test_shared_bias_refinement.py` |
| V2_01 checkpoint selection | `outputs/euroc_checkpoint_selection_V2_01_easy_droid_cam0_20260928_223138/selection.txt` |
| V2_03 preliminary cross-sequence metric | `outputs/euroc_crossseq_v2_03_hospital_8k_20260928/summary_metric_v2.md` |
| 진행 중 shared-bias ablation | `outputs/euroc_v2_03_shared_bias_20260929_132338/` |

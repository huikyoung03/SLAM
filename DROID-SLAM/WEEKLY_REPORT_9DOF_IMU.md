# 주차보고서: DROID-SLAM 기반 9DoF Visual-Inertial DBA

작성일: 2026-09-28

## 1. 연구 목표

본 주차의 목표는 monocular DROID-SLAM의 dense bundle adjustment(DBA)에 IMU를
사후 pose 보정값이 아니라 실제 최적화 factor로 결합하는 것이다. 최종 목표는
휴대폰 RGB와 IMU 입력에도 적용 가능한 DROID+IMU 모델이며, 이번 단계에서는
공식 TartanAir V2 학습 데이터와 EuRoC 검증 데이터를 사용해 모델 구조와
성능 측정 방법을 정립했다.

검증하려는 가설은 다음과 같다.

> Pose, velocity, dense inverse depth를 공동 최적화하고, IMU preintegration
> residual을 visual reprojection residual과 함께 DBA에 넣으면 monocular DROID의
> scale 모호성과 단기 motion drift를 완화할 수 있다. GRU가 예측한 confidence는
> IMU가 신뢰할 만한 구간에서 factor의 영향을 적응적으로 조절한다.

Appearance/feature-metric branch는 이번 범위에 포함하지 않았다. Full 15DoF
per-keyframe VIO도 계산 비용이 크므로, local DBA의 active state를 9DoF로
제한했다.

## 2. 제안한 9DoF 구조

각 keyframe의 local state는 다음과 같다.

\[
X_k = (T_k, v_k, d_k)
\]

- `T_k`: camera pose, 6DoF
- `v_k`: world-frame velocity, 3DoF
- `d_k`: dense inverse depth

Gravity `g`, gyro bias `b_g`, accelerometer bias `b_a`, scale은 visual-inertial
initialization 단계에서 추정한 뒤 local DBA에서는 고정 calibration parameter로
사용한다. 이 방식은 모든 keyframe에 bias까지 넣는 15DoF 방식보다 reduced camera
system의 크기를 줄인다.

Runtime objective는 다음과 같다.

\[
E(T,v,d) = E^r(T,d) + E^u(T,v; g,b_a,b_g)
\]

IMU factor는 rotation, velocity, position residual을 사용한다.

\[
E^u = \rho(r_R) + \lambda_v\rho(r_v) + \lambda_p\rho(r_p)
\]

DBA에는 단순 residual 값이 아니라 각 residual의 Jacobian으로 구성한
`J^T W J`, `J^T W r` block이 추가된다. `W`에는 learned global IMU factor
weight, GRU confidence, covariance-derived information, learned position/velocity
relative weight가 반영된다.

## 3. 구현 내용

| 영역 | 구현 내용 |
| --- | --- |
| IMU 입력 | image timestamp 사이의 gyro/acc window를 preintegrate하여 `Delta R`, `Delta v`, `Delta p`, bias Jacobian, covariance를 생성 |
| DBA | pose+velocity state에 rotation/velocity/position IMU factor와 analytic Jacobian을 추가하고 dense depth에는 기존 Schur complement를 재사용 |
| 네트워크 | ConvGRU update module에 IMU confidence head를 추가하고 global/position/velocity residual weight를 학습 가능 parameter로 추가 |
| 학습 | TartanAir V2 dense depth 기반 기존 DROID loss를 유지하면서 full IMU preintegration loss를 추가 |
| 초기화 | visual-inertial alignment로 scale, gravity, global bias의 초기값을 준비하고 local DBA에는 fixed parameter로 전달 |
| 평가 | EuRoC body GT를 cam0 GT로 변환하고, VI 정렬 이후 DROID 내부 length unit을 metre로 환산하여 metric ATE/RPE를 계산 |

주요 코드 위치는 다음과 같다.

- `droid_slam/geom/ba.py`: 9DoF IMU factor, Jacobian, reduced system
- `droid_slam/droid_net.py`: unrolled DBA와 learned confidence/weight parameter
- `droid_slam/geom/losses.py`: full preintegration training loss
- `train.py`: fixed-bias 9DoF training mode와 loss aggregation
- `droid_slam/data_readers/tartan_v2.py`: TartanAir V2 IMU loader 및 coordinate transform
- `tools/evaluate_euroc_recon.py`: Sim(3)과 metric VI 지표 평가

## 4. 이번 주에 확인하고 수정한 문제

### 4.1 TartanAir pose-IMU 좌표축 불일치

기존 DROID TartanAir pose 변환은 simulator axis를 `(x,y,z) -> (y,z,x)`로
바꾸고 있었지만 gyro/acc vector는 raw simulator axis 그대로 적분하고 있었다.
이 경우 visual pose와 IMU residual이 서로 다른 좌표계에 있게 된다.

- 수정: gyro와 accelerometer에도 동일한 cyclic axis transform 적용
- 사전 검증: IMU relative rotation과 GT pose relative rotation의 평균 불일치가
  약 `6.98 deg`에서 `0.385 deg`로 감소

### 4.2 학습 unroll에서 velocity state를 재초기화하던 문제

기존 `DroidNet.forward()`는 각 unrolled BA iteration 뒤 pose difference로
velocity를 다시 만들었다. 따라서 DBA가 갱신한 velocity가 다음 iteration에서
소실됐다.

- 수정: BA가 갱신한 velocity state를 유지하고 pose-difference velocity는
  초기 prior 역할로만 사용
- 의미: 학습 unroll과 runtime local DBA가 동일한 motion-state 전개를 사용

### 4.3 평가 metric scale 환산 누락

VI alignment 이후 DROID backend는 내부 pose/depth 단위를 normalize한다.
기존 evaluator는 `imu_unit_scale`을 적용하지 않아 metric ATE/RPE에 잘못된
unit이 사용될 가능성이 있었다.

- 수정: alignment가 성공한 reconstruction은 저장된 `imu_unit_scale`로 metre 환산
- visual-only reconstruction은 scale을 알 수 없으므로 metric SE(3) ATE 대신
  Sim(3) shape metric만 보고

## 5. 학습 손실 구조

현재 training objective는 다음과 같다.

\[
L = 10L_{geo} + 0.01L_{res} + 0.05L_{flow}
  + w_{imu}L_{preint} + L_{conf-reg} + L_{weight-prior}
\]

- `L_geo`: synthetic GT pose에 대한 DROID geodesic pose loss
- `L_res`: reprojection residual supervision
- `L_flow`: dense depth를 사용한 DROID flow/depth loss
- `L_preint`: position, velocity, rotation full preintegration loss
- `L_conf-reg`: confidence가 전 구간에서 1로 포화되는 것을 방지
- `L_weight-prior`: position/velocity residual weight가 식별 불가능하게 붕괴하는 것을 방지

fixed-bias 9DoF 모드에서는 bias random-walk loss와 global bias regularization은
의도적으로 제외된다. Bias는 residual 계산에는 반영되지만 optimizer가 학습하지
않는다.

`w_imu`는 training supervision multiplier이고 runtime IMU factor weight와는 다르다.
Runtime strength는 learned global weight, GRU confidence, covariance information,
position/velocity relative weight의 곱으로 구성된다.

## 6. 수행한 실험

### 6.1 구조 및 gradient smoke test

- official `droid.pth`를 불러온 뒤 새 IMU head가 missing parameter로 정상 초기화되는지 확인
- `--use_pose_velocity_imu_ba`, `--use_full_imu_loss`, `--freeze_non_imu` 조건에서
  2-step training 실행
- pose, velocity, dense depth 및 IMU loss/learned confidence/weight log가 모두 생성되는지 확인
- finite-difference 방식으로 IMU factor Jacobian의 수치 일관성을 확인

결과: 9DoF factor가 단순 post-processing prior가 아니라 BA gradient/Hessian 경로에
들어가며, IMU 관련 학습 parameter에도 gradient가 전달됨을 확인했다.

### 6.2 Hospital TartanAir V2 fine-tuning

- 학습 데이터: Hospital environment 22 trajectories, 32,340 frames
- camera: `lcam_front` 단안
- IMU: `acc_nograv_body.txt`, gyro, camera/IMU timestamp 사용
- image temporal stride: `1, 2, 4, 8`
- 기존 8k run: fixed-bias 9DoF, learned IMU confidence, learned residual weights,
  covariance-relative information weighting

### 6.3 EuRoC cross-dataset validation

TartanAir에서 학습한 checkpoint를 EuRoC V2_01 easy와 V2_03 difficult의 cam0
sequence에 적용했다. 평가 시 EuRoC body GT를 cam0 extrinsic으로 변환했고, baseline과
IMU 모델은 동일 frame ID에서 비교했다.

## 7. 기존 8k checkpoint 결과

아래 수치는 metric conversion을 고친 뒤 재평가한 결과다.

| Sequence | Model | Sim(3) ATE RMSE | Metric SE(3) ATE RMSE | Metric RPE 1 s | Scale residual |
| --- | --- | ---: | ---: | ---: | ---: |
| V2_01 easy | Original DROID | 0.021040 m | scale unobservable | - | - |
| V2_01 easy | IMU 8k | 0.021016 m | 0.251413 m | 0.061982 m | 11.75% |
| V2_03 difficult | Original DROID | 0.017100 m | scale unobservable | - | - |
| V2_03 difficult | IMU 8k | 0.017037 m | 0.166526 m | 0.081664 m | 7.84% |

V2_03에서 IMU factor는 `11,155 / 11,174` temporal edge, 약 `99.83%`에서 실제로
사용됐다.

## 8. 결과 해석

### 확인된 내용

1. IMU preintegration의 rotation뿐 아니라 accelerometer 기반 velocity/position
   residual도 9DoF DBA에 물리적으로 결합됐다.
2. learned confidence와 learned factor weight가 runtime DBA에 전달되며, IMU edge가
   실제 optimization에 사용됐다.
3. visual-only monocular DROID에는 없는 metric scale constraint를 IMU 모델이 제공했다.
4. EuRoC라는 서로 다른 실제 데이터셋에서 tracking과 metric trajectory 생성을 수행했다.

### 아직 주장할 수 없는 내용

1. visual reconstruction shape가 유의미하게 좋아졌다고 말할 수 없다. Sim(3) ATE 차이는
   V2_01과 V2_03 모두 매우 작다.
2. baseline monocular DROID는 metric scale이 관측 불가능하므로, baseline의 metric
   SE(3) ATE와 IMU 모델의 metric SE(3) ATE를 직접 우열 비교할 수 없다.
3. 기존 8k checkpoint는 axis mismatch와 velocity-state reset 문제를 수정하기 전에
   학습됐으므로 최종 후보로 사용하면 안 된다.
4. Hospital 하나의 synthetic environment와 ideal IMU만으로 실제 휴대폰 robustness를
   보장할 수 없다. 현재 noise model은 covariance weighting이며 raw IMU noise/bias
   augmentation 자체는 아니다.

## 9. 현재 진행 중인 clean experiment

수정된 coordinate transform과 velocity propagation을 반영한 clean fine-tuning을
official `droid.pth`에서 다시 시작한다.

- script: `tools/run_9dof_axisfix_finetune.sh`
- training steps: 8,000
- checkpoints: 2k, 4k, 6k, 8k
- trainable parameters: IMU confidence head, learned global factor weight,
  learned position/velocity relative weights
- frozen parameters: pretrained visual DROID network, global gyro/acc bias

이 실험은 기존 8k 결과를 대체할 최종 candidate family를 만드는 과정이다.

## 10. 다음 주 계획

1. clean 2k/4k/6k/8k checkpoint를 EuRoC V2_01 전체 sequence에서 비교한다.
   선택 기준은 metric SE(3) ATE와 RPE 1 s이며, runtime threshold/weight를
   sequence별로 수동 조정하지 않는다.
2. 선택된 checkpoint 하나를 V2_03 또는 아직 보지 않은 별도 EuRoC sequence에서
   cross-sequence validation한다.
3. 다음 ablation을 같은 조건에서 비교한다.
   - Original visual DROID
   - fixed/non-learned IMU 9DoF
   - learned confidence/weight IMU 9DoF
4. clean 결과에서 IMU supervision이 여전히 약하거나 metric 성능 이득이 없을 경우,
   `w_imu = 0.05, 0.25, 1.0`의 제한된 controlled sweep을 V2_01 validation 기준으로
   수행한다.
5. 이후 실제 휴대폰 IMU의 noise density, bias stability, camera-IMU extrinsic을
   반영한 noise/bias augmentation 및 calibration 실험을 추가한다.

## 11. 주차 결론

이번 주에는 DROID-SLAM의 visual-only DBA를 fixed-bias 9DoF visual-inertial DBA로
확장하고, gyro와 accelerometer를 모두 사용하는 IMU residual을 실제 Hessian/Jacobian
optimization 경로에 결합했다. 구조적 동작과 metric trajectory 생성은 확인됐지만,
기존 8k checkpoint만으로 visual accuracy 개선을 확정할 수는 없다. 다음 clean
fine-tuning과 full-sequence held-out validation이 성능 개선 여부를 판단하는 핵심
실험이다.

## 12. 문헌 근거와 성능 전망

가장 직접적인 참고 문헌은 [DVI-SLAM](https://arxiv.org/html/2309.13814v2)이다.
이 연구는 DROID 계열의 differentiable DBA에 preintegrated IMU factor와 learned
IMU confidence를 tightly coupled 방식으로 넣었다. EuRoC ablation에서 monocular
baseline ATE-scale은 `0.179 m`였고, preintegration weight만 쓴 fixed IMU는
`0.251 m`로 오히려 악화됐다. 반면 tuned IMU는 `0.129 m`, learned IMU confidence는
`0.114 m`로 baseline 대비 `36.3%` 감소했다. Feature-metric factor까지 포함한
전체 DVI-SLAM은 `0.098 m`, 즉 `45.3%` 감소를 보고했다.

따라서 다음 두 가지 근거가 있다.

1. IMU factor를 DBA Hessian에 넣고 learned confidence로 신뢰도를 조절하는 방향은
   성능 개선 가능성이 있는 연구 방향이다.
2. 고정 또는 수동 IMU weight는 성능을 악화시킬 수 있으므로, 현재 구현의 learned
   global weight, learned position/velocity weights, GRU confidence가 필요하다.

그러나 DVI-SLAM의 `45.3%` 수치는 feature-metric branch와 full IMU state까지 포함한
결과이므로 현재 9DoF IMU-only 구조의 예상 성능으로 사용하면 안 된다. 우리 구조는
feature-metric branch가 없고, bias를 local state에서 고정하며, Hospital 한 환경의
ideal synthetic IMU로 fine-tuning한다. 따라서 문헌의 IMU-only `36.3%`도 상한에
가까운 참고값으로 봐야 한다.

또한 [VINS-Mono](https://arxiv.org/abs/1708.03852)와
[online visual-inertial calibration 연구](https://arxiv.org/abs/2004.05534)는
monocular VIO에서 scale, velocity, gravity, gyro/accelerometer bias, camera-IMU
extrinsic 및 time offset 초기화가 정확도에 중요함을 보여준다. 현재 코드에서
camera-IMU rotation은 반영하지만 translation lever arm과 online time calibration은
아직 완성 범위가 아니므로, 실제 휴대폰에서의 성능은 이 calibration 품질에 크게
좌우된다.

현재 근거를 반영한 engineering forecast는 다음과 같다. 이는 통계적으로 계산된
확률이 아니라 문헌, 현재 코드 구조, 기존 8k 결과를 함께 반영한 보수적 추정이다.

| 예상 결과 | 추정 가능성 | 해석 |
| --- | ---: | --- |
| learned 9DoF가 fixed-weight 9DoF보다 적어도 한 held-out VI metric에서 개선 | 60-70% | learned confidence의 목적에 가장 직접적인 검증 |
| 여러 held-out sequence 평균에서 RPE 또는 metric SE(3) ATE가 5-15% 개선 | 35-50% | 현재 구조에서 현실적인 1차 목표 |
| 빠른 motion/저텍스처 구간에서 10-25% 수준의 local RPE 개선 | 45-60% | IMU가 가장 도움이 되는 조건 |
| 전체 평균 ATE/RPE가 20% 이상 개선 | 15-25% | 더 많은 환경/실제 IMU augmentation 및 calibration이 필요할 가능성이 큼 |
| DVI-SLAM 전체 구조 수준의 36-45% 개선 | 10% 이하 | feature-metric branch, full-state optimization, 더 넓은 학습/검증 조건이 현재와 다름 |

기존 8k 결과의 Sim(3) shape 개선은 1% 미만이므로, clean training 뒤에도 전체
평균 visual shape가 자동으로 크게 좋아질 것이라고 기대하면 안 된다. 이번 구조가
성공했다고 판단할 가장 현실적인 기준은 다음과 같다.

1. learned 9DoF가 fixed/non-learned IMU ablation보다 metric RPE와 scale error에서
   일관되게 좋아질 것
2. visual-only DROID보다 difficult motion segment의 Sim(3) RPE 또는 tracking
   completion이 좋아질 것
3. metric scale residual이 현재 `8-12%` 수준에서 `5-8%` 이하로 낮아질 것

이 기준을 충족하면 DROID+IMU 결합이 성능 개선의 실질적인 여지를 보였다고
보고할 수 있다. 충족하지 못하면 `w_imu`를 임의로 키우기보다 timestamp, extrinsic,
bias/noise augmentation, VI initialization을 먼저 점검해야 한다.

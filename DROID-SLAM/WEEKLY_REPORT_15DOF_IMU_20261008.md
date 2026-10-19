# 주차보고서: DROID-SLAM 기반 15DoF Two-Stage Visual-Inertial DBA

작성일: 2026-10-08  
보고 범위: 2026-09-29 이후 진행 사항  
대상 저장소: `/home/ubuntu/DROID-SLAM`

## 1. 이번 주 요약

지난 보고서에서는 pose와 velocity를 최적화하는 fixed-bias 9DoF IMU-DBA를
구축하고, TartanAir 학습 및 EuRoC 검증 기준을 정립했다. 이번 주에는 이를
**keyframe별 pose, velocity, accelerometer bias, gyro bias를 함께 다루는 15DoF
two-stage Visual-Inertial DBA**로 확장했다.

핵심 진척은 다음과 같다.

1. local DBA를 visual pose/depth 단계와 full 15DoF IMU 단계로 분리했다.
2. VI 초기화가 scale, gravity, velocity, bias를 안정적으로 공급한 뒤에만 full
   IMU 최적화를 켜도록 연결했다.
3. bias 변화에 대한 Jacobian 검증, 재적분(reintegration), 전역 bias refinement 및
   matched control 실험을 추가했다.
4. 기존 scalar IMU confidence를 회전·속도·위치별 3-channel confidence와
   IMU-quality feature fusion으로 확장하고, 해당 checkpoint의 held-out 전체
   sequence 평가를 시작했다.

이번 주의 결론은 다음과 같다.

> 15DoF tightly-coupled IMU factor와 metric trajectory 생성은 정상 동작한다.
> 다만 current fixed/learned confidence 및 bias-refinement 조합만으로
> visual-only DROID 대비 일관된 성능 향상을 아직 주장할 수 없다. 다음 우선순위는
> full preintegration covariance/whitening과 confidence 효과의 엄밀한 ablation이다.

Feature-metric/appearance residual은 이번 주에도 추가하지 않았다. 목적함수는
visual reprojection residual과 IMU preintegration residual의 결합으로 유지했다.

## 2. 구조 변경: 9DoF fixed-bias에서 15DoF two-stage로

### 2.1 최적화 상태

기존 9DoF local state는 다음과 같았다.

\[
X_i=(T_i,v_i,d_i)
\]

이번 주에는 IMU joint stage에서 다음 상태를 사용하도록 확장했다.

\[
X_i=(T_i,v_i,b_{a,i},b_{g,i},d_i)
\]

- `T_i`: camera pose, 6DoF
- `v_i`: world-frame velocity, 3DoF
- `b_{a,i}, b_{g,i}`: accelerometer/gyroscope bias, 각 3DoF
- `d_i`: 기존 DROID-SLAM의 dense inverse depth

visual reprojection residual은 pose/depth block에, IMU rotation/velocity/position
residual 및 bias random-walk constraint는 full motion-state block에 결합된다.
Dense depth에 대한 기존 Schur complement 구조는 유지했다.

### 2.2 two-stage local DBA

초기 visual pose가 불안정한 상태에서 IMU와 bias까지 동시에 강하게 최적화하면
scale과 bias가 잘못된 local minimum으로 갈 수 있다. 이를 피하기 위해 local
frontend를 다음 두 단계로 분리했다.

```text
Stage 1: visual pose/depth DBA
         -> visual geometry를 먼저 안정화

VI initialization 성공
         -> scale, gravity, velocity, initial bias 제공

Stage 2: full 15DoF DBA
         -> pose + velocity + accel/gyro bias + dense depth 공동 최적화
```

runtime에서 VI alignment가 아직 성공하지 않은 경우에는 second stage를 수행하지
않는다. 이로써 IMU factor가 초기 불안정 상태를 증폭시키지 않게 했다.

### 2.3 구현 및 검증 범위

| 영역 | 이번 주 변경/검증 |
| --- | --- |
| CUDA DBA | pose/velocity/bias를 포함하는 full-state prior 및 reduced system 연결 |
| Factor graph | `pose_only`와 `full_15dof` stage의 분리, stage 전환 단위 테스트 |
| VI initialization | history snapshot을 포함한 후보 탐색, scale/gravity/bias seed 적용 |
| Bias | keyframe별 bias state, random-walk factor, 전역/shared bias refinement 실험 |
| Preintegration | bias Jacobian finite-difference 검증 및 reconstruction 기준 재적분 경로 |
| Runtime contract | camera–IMU extrinsic, timestamp/manifest, prior 일치성 검사 |
| 평가 | Sim(3), metric SE(3), RPE, scale error, VI-init/IMU factor 사용률 기록 |

## 3. 학습 confidence 확장

기존 confidence는 ConvGRU hidden state에서 하나의 scalar map을 출력하고 temporal
IMU edge별로 평균화하는 구조였다. 이번 주에는 다음의 component-wise 구조를 추가했다.

\[
c_{ij}=[c^R_{ij},c^v_{ij},c^p_{ij}]
\]

- `c^R`: rotation residual confidence
- `c^v`: velocity residual confidence
- `c^p`: position residual confidence

추가로 preintegrated interval의 `dt`, angular-rate/acceleration/position proxy,
rotation/velocity/position information을 7D quality vector로 만들고, MLP를 통해
각 confidence logit에 더하도록 했다. scalar checkpoint로부터 component head를
초기화해, fine-tuning 시작 시 기존 behavior를 보존하도록 구성했다.

이는 DVI-SLAM의 learned IMU confidence 원칙을 유지하면서, rotation·velocity·position
오차가 서로 다른 실제 IMU 구간을 구분하기 위한 확장이다. 다만 component head가
성능을 개선했다는 결론은 아직 내리지 않았다.

## 4. 수행한 실험

### 4.1 9DoF checkpoint 선택 및 cross-sequence 확인

V2_01 easy에서 Hospital TartanAir로 학습한 2k/4k/6k/8k checkpoint를 비교했다.
가장 낮은 metric SE(3) ATE는 6k checkpoint의 `0.079226 m`였다. 동일 조건의
original DROID Sim(3) ATE는 `0.016053 m`, 2k checkpoint는 `0.016032 m`로
visual shape 차이는 매우 작았다.

**해석:** 9DoF의 metric trajectory 및 IMU factor 사용은 확인됐지만, Sim(3)
visual shape의 개선량만으로 일반화 성능 향상을 주장할 정도는 아니었다.

### 4.2 V2_03 full-sequence 9DoF bias 실험

V2_03 difficult 전체 sequence에서 fixed-bias와 low-rate/shared-bias refinement를
비교했다. low-rate bias 결과는 fixed-bias보다 metric SE(3) ATE가
`0.371717 m -> 0.364341 m`로 소폭 감소했으나, Sim(3) ATE는
`0.015122 m -> 0.015500 m`로 악화됐다.

**해석:** bias update가 metric scale/trajectory에 줄 수 있는 영향은 보였지만,
visual trajectory와 metric trajectory 모두에서 일관되게 이득을 만들지는 못했다.
따라서 이 구조만으로 bias refinement의 유효성을 확정하지 않았다.

### 4.3 15DoF two-stage 및 재적분 검증: V1_03

V1_03 difficult 41 common reconstruction pose에서 다음 결과를 얻었다.

| 조건 | Sim(3) ATE RMSE | Metric SE(3) ATE RMSE | IMU 사용 비율 |
| --- | ---: | ---: | ---: |
| Original DROID | 0.009562 m | - | - |
| Visual bootstrap control | 0.008096 m | - | - |
| Two-stage 15DoF pass-0 | 0.009572 m | 0.017311 m | 0.887 |
| Reintegrated pass | 0.009562 m | 0.021616 m | 0.887 |
| Two-stage + global bias | 0.009301 m | 0.015676 m | 0.887 |

global-bias run의 결과는 좋아 보이지만, matched ablation에서 동일한 post-control이
거의 같은 값을 냈다. 즉 이 수치만으로 global bias update 자체의 이득이라고
귀속할 수 없다.

### 4.4 Held-out V2_03 two-stage/global-bias ablation

V2_03 difficult 99 common reconstruction pose에서는 다음과 같았다.

| 조건 | Sim(3) ATE RMSE | Metric SE(3) ATE RMSE | Scale error | IMU 사용 비율 |
| --- | ---: | ---: | ---: | ---: |
| Original DROID | 0.017900 m | - | - | - |
| Visual bootstrap control | 0.019180 m | - | - | - |
| Two-stage 15DoF pass-0 | 0.017254 m | 0.222643 m | 10.23% | 0.795 |
| Post-control | 0.018820 m | 0.226775 m | 10.44% | 0.798 |
| Global bias | 0.018820 m | 0.226775 m | 10.44% | 0.798 |

Two-stage pass-0는 matched visual bootstrap control보다 Sim(3) ATE가 낮았지만,
후속 post/global-bias 단계는 이득을 보이지 않았다. 또한 original DROID와의
비교는 keyframe 수와 bootstrap path가 달라, 이를 단독 최종 성능 주장으로 사용하지
않는다.

### 4.5 Learned scalar confidence checkpoint

6k learned checkpoint를 동일 V2_03 held-out sequence에서 평가했다.

| 조건 | Sim(3) ATE RMSE | Metric SE(3) ATE RMSE | Mean confidence | Mean IMU weight |
| --- | ---: | ---: | ---: | ---: |
| Two-stage fixed/pass-0 | 0.017254 m | 0.222643 m | 1.000000 | 0.002000 |
| Two-stage learned 6k | 0.017267 m | 0.222935 m | 0.654780 | 0.036401 |

learned confidence가 더 이상 `1.0`에 고정되지 않고 실제로 runtime factor에
전달되는 것은 확인했다. 그러나 이 checkpoint는 fixed/pass-0 대비 metric ATE나
Sim(3) ATE에서 유의미한 이득을 보이지 않았다.

## 5. 결과 해석 및 현재 판단

### 확인된 내용

1. IMU는 post-processing이 아니라 15DoF DBA의 Hessian/gradient 경로에 결합된다.
2. VI initialization 후 metric scale, gravity, velocity, bias seed를 이용하는
   full-sequence runtime이 동작한다.
3. IMU factor는 V1_03/V2_03에서 약 80~89%의 유효 temporal edge에 실제 적용됐다.
4. learned scalar confidence는 neutral value에 고정되지 않고 runtime weight를
   변화시킬 수 있다.
5. bias Jacobian, stage separation, reintegration, global-bias 경로에 대한
   단위/통합 검증 도구를 마련했다.

### 아직 주장할 수 없는 내용

1. IMU-only 구조가 original DROID보다 일반적으로 정확하다고 결론 낼 수 없다.
2. component-wise confidence가 scalar confidence보다 좋다고 말할 결과가 아직 없다.
3. global/shared bias refinement의 독립적인 성능 이득은 matched control에서
   확인되지 않았다.
4. 현재 사전적분 uncertainty는 rotation/velocity/position별 diagonal information
   근사이다. 회전–속도–위치 및 bias 간 상관관계를 포함하는 full covariance가
   아직 아니다.

## 6. DVI-SLAM 관점의 gap 분석

DVI-SLAM은 feature-metric branch가 없는 IMU-only ablation에서도 learned IMU
confidence를 통해 fixed IMU보다 개선을 보였다. 현재 구현은 full 15DoF state,
two-stage solve, learned confidence라는 핵심 방향을 갖췄다. 다만 다음 차이가 남아
있다.

| 항목 | 현재 상태 | 다음 개선 |
| --- | --- | --- |
| Preintegration uncertainty | 3개 scalar information 근사 | Forster 방식 full covariance 전파 및 whitening |
| Confidence 검증 | mean/weight 기록 중심 | NIS/whitened residual과 confidence의 calibration 분석 |
| 학습-런타임 일치 | unrolled PyTorch BA와 CUDA DBA가 모두 존재 | 같은 입력의 1-step Hessian/gradient/update parity test |
| Stage 전환 | VI alignment 성공 여부 중심 | initialization residual/condition 및 innovation 기반 adaptive ramp |
| Bias 처리 | first-order Jacobian 및 별도 재적분 경로 | bias 변화 시 자동 재적분 기준 확립 |

따라서 다음 단계의 핵심은 IMU branch를 더 크게 만드는 것이 아니라,
**물리적으로 올바른 covariance weight \(w^{pu}\)** 와 **검증 가능한 learned
confidence \(w^u\)** 를 분리해 개선하는 것이다.

## 7. Component-wise confidence 전체 sequence 결과

V2_03 difficult 전체 sequence에서 component-wise confidence checkpoint 평가를
완료했다.

- checkpoint: `two_stage_15dof_component_confidence_quality_3000_final_003000.pth`
- runtime: two-stage 15DoF, IMU noise model, learned IMU confidence,
  component confidence, learned IMU BA/residual weights
- 입력: stride 2, 전체 961 image frame
- reconstruction pose: 323개
- 출력 경로: `outputs/v2_03_component_fullseq_20261008_145718/`

| 지표 | 결과 |
| --- | ---: |
| Sim(3) ATE RMSE | 0.015628 m |
| Metric SE(3) ATE RMSE | 0.353340 m |
| Metric scale error | 20.995% |
| Metric RPE 1 s translation | 0.151963 m |
| IMU factor 사용률 | 0.999110 |
| IMU factor 사용 수 | 13,466 / 13,478 debug record |
| Mean component confidence | 0.662543 |
| Mean runtime IMU weight | 0.058376 |

runtime debug record의 confidence 평균은 rotation `0.656`, velocity `0.654`,
position `0.678`이었다. median은 각각 `0.479`, `0.464`, `0.488`로, 각 channel이
중립값 1에 고정되지 않고 interval별로 다른 weight를 출력했음을 확인했다.
따라서 component head와 IMU-quality fusion의 **실행 경로 및 factor 반영 자체는
확인**됐다.

단, 이 결과는 기존 scalar/fixed 15DoF 결과가 사용한 450-frame prefix가 아니라
전체 961-frame sequence에서 나온 결과다. 입력 길이, keyframe 수, 누적 drift가
다르므로 `0.015628 m`를 기존 prefix의 `0.017254 m` 또는 `0.017267 m`와 직접
비교해 component confidence의 성능 개선이라고 해석하면 안 된다. 같은 전체 sequence,
같은 stride, 같은 calibration, 같은 keyframe policy에서 visual-only/fixed/scalar를
다시 실행한 paired 4-way ablation이 필요하다.

## 8. 다음 주 계획

1. 같은 전체 V2_03 sequence, stride, calibration, keyframe policy에서 다음 4-way
   paired ablation을 수행한다.
   - visual-only control
   - fixed IMU (`w^u=1`)
   - learned scalar confidence
   - learned component-wise confidence
2. 각 run에서 confidence와 covariance-normalized IMU residual(NIS)의 상관관계를
   기록해 confidence가 실제로 불량 IMU interval을 낮추는지 검증한다.
3. ablation으로 learned confidence의 순수 기여를 확인한 뒤,
   `tools/imu_preintegrate.py`부터 DBA까지 full preintegration covariance와
   whitening을 구현한다.
4. PyTorch training DBA와 CUDA runtime DBA의 one-step parity test를 추가하고,
   visual DROID parameter를 동결한 stage-3 confidence 학습을 수행한다.

## 9. 주차 결론

이번 주에는 기존 9DoF fixed-bias IMU 결합을 15DoF two-stage tightly-coupled
Visual-Inertial DBA로 확장했고, VI 초기화·bias·재적분·metric evaluation의
운영 경로를 정리했다. 또한 component-wise confidence가 전체 sequence에서
실제로 dynamic factor weight를 생성하고 IMU factor에 반영되는 것을 확인했다.
그러나 현재의 diagonal covariance 및 confidence 학습만으로 held-out sequence에서
일관된 visual accuracy 향상을 확정하지 못했다.

다음 주에는 동일 전체 sequence에서 엄밀한 matched 4-way ablation으로 component
confidence의 순수 효과를 먼저 검증한다. 그 뒤 full covariance/whitening을 우선
구현해 DVI-SLAM의 `w^{pu} \times w^u` 결합을 더 정확히 재현할 예정이다.

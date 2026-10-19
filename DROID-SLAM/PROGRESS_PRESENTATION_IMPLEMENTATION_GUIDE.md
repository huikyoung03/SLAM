# DROID-SLAM 기반 단안 Visual–Inertial SLAM: 진행 내용 발표 자료

> 이 문서는 **Chapter 3. 프로젝트 진행 내용**을 위한 슬라이드 초안과 발표 대본이다.  
> 수치·구조는 최종 실행한 `two-stage 15DoF + DVI-style scalar learned confidence` 조건과 실제 로그를 기준으로 작성했다.

---

## 0. 발표에서 반드시 구분할 용어

| 용어 | 이 프로젝트에서의 의미 | 혼동하면 안 되는 것 |
| --- | --- | --- |
| **two-stage runtime** | 실행 중 visual warm-up 후 full 15DoF VIO DBA로 전환하는 추정 절차 | 학습의 stage-3가 아님 |
| **stage-3 training** | 시각 DROID backbone을 고정하고 IMU confidence head만 재학습하는 학습 절차 | runtime의 두 번째 단계가 아님 |
| **15DoF state** | 키프레임별 pose 6 + velocity 3 + accelerometer bias 3 + gyroscope bias 3 | depth는 기존 DROID의 별도 최적화 변수 |
| **confidence \(w^u\)** | IMU를 넣을지 말지를 결정하는 binary gate가 아니라, 각 시간 구간의 IMU residual 비중을 0~1 사이에서 조절하는 값 | IMU 측정값 자체나 sensor noise 값이 아님 |
| **\(\lambda_{imu}=0.002\)** | visual reprojection 항과 IMU factor의 전체 상대 세기를 정하는 전역 계수 | learned confidence와 다른 역할 |

발표 중에는 `stage`라는 단어가 두 번 쓰인다는 점을 먼저 밝혀 두면 질문을 줄일 수 있다.

---

## 1. 이 장의 권장 흐름

아래는 10분 제약을 무시하고 모든 구현 근거를 담은 구성이다. 실제 발표에서는 ★ 표시 7~8장을 중심으로 줄이면 된다.

1. ★ 1학기 수행 내용 리마인드 및 문제 재정의
2. ★ 데이터셋 선정 근거와 split
3. ★ 기존 DROID-SLAM baseline 구조 분석
4. ★ IMU를 단순히 더하지 않은 이유: 단안 초기화 문제
5. ★ 전체 제안 구조: two-stage 15DoF DROID+IMU
6. IMU 전처리와 preintegration
7. ★ Stage 1 / VI alignment / Stage 2의 구체적인 처리 순서
8. ★ 15DoF DBA에서 실제로 최적화하는 변수와 residual
9. ★ DVI-inspired learned confidence branch의 역할
10. ★ 학습(stage-3) 설계와 데이터 구성
11. 학습 실패 진단 및 parity 재학습
12. ★ 실험 설계와 공정한 비교 기준
13. ★ 최종 결과와 해석
14. 한계와 다음 개선 방향

모델 구조 그림 파일: [`presentation_assets/final_vi_architecture.svg`](presentation_assets/final_vi_architecture.svg)

---

# 슬라이드별 자료와 대본

## 슬라이드 1. 1학기 리마인드와 이번 학기의 질문 ★

### 화면에 넣을 내용

**1학기**

- 모바일 카메라·IMU 수집/전송 흐름 조사 및 구현
- DROID-SLAM baseline 실행, trajectory 및 dense reconstruction 확인
- IMU preintegration과 visual–inertial initialization 가능성 검토

**이번 학기 질문**

> 단안 DROID-SLAM의 dense visual tracking을 유지하면서, IMU로 metric scale과 motion constraint를 안정적으로 결합할 수 있는가?

**구현 목표**

`camera + IMU → synchronized stream → DROID+IMU optimization → metric trajectory + dense reconstruction`

### 발표 대본

> 1학기에는 모바일 카메라와 IMU를 수집하고 전달하는 기반, 그리고 DROID-SLAM의 실행 구조를 확인했습니다. 이번 학기에는 단순히 센서를 추가하는 것이 아니라, 단안 DROID-SLAM이 가지는 절대 scale 부재와 빠른 움직임에서의 불안정성을 IMU 제약으로 보완하는 것을 목표로 했습니다. 최종 목표는 metric 단위 trajectory와 dense reconstruction을 함께 만드는 것입니다.

### 발표자 이해용 메모

- **단안 scale 부재**: 영상만으로는 ‘1만큼 움직였다’가 1 m인지 0.1 m인지 절대적으로 알 수 없다.
- IMU의 가속도는 SI 단위를 가지므로, 시각 trajectory와 올바르게 정렬하면 scale의 단서를 줄 수 있다.
- 모바일 수집 파트는 실제 구현한 범위만 보여 준다. 최종 정량 평가는 동기화된 GT/IMU가 있는 EuRoC에서 했다고 분리해서 말한다.

---

## 슬라이드 2. 데이터셋 선정 근거와 split ★

### 화면에 넣을 내용

**왜 EuRoC MAV인가?**

| 선정 기준 | EuRoC가 제공하는 근거 | 프로젝트에 필요한 이유 |
| --- | --- | --- |
| 동기화된 visual–inertial 입력 | global-shutter camera 20 Hz + gyro/accel IMU 200 Hz, shutter-centric temporal alignment | image–IMU interval을 정확히 preintegrate해야 함 |
| metric 평가 기준 | GT pose/position, camera–IMU calibration, time-aligned GT 제공 | scale, ATE, RPE와 IMU initialization을 검증 가능 |
| 환경/난이도 변화 | Machine Hall과 Vicon Room, easy/medium/difficult sequence | 특정 장면·쉬운 motion에만 맞춘 confidence인지 확인 가능 |
| 구현 조건 적합성 | monocular camera만 선택 가능, IMU는 원래 주파수 유지 가능 | 최종 목표인 10 Hz monocular + IMU 조건을 재현 가능 |

**현재 split**

```text
MH_03_medium        → stage-3 confidence training
V1_02_medium
V2_02_medium        → validation / checkpoint selection
V2_03_difficult     → final re-evaluation (common 239 frames)
```

### 발표 대본

> EuRoC MAV를 선택한 이유는 카메라와 IMU가 시간 동기화되어 있고, calibration과 metric ground truth까지 제공하기 때문입니다. 단순 visual dataset으로는 IMU preintegration의 시간 구간이나 scale recovery가 맞는지 검증하기 어렵습니다. 원본 카메라는 20 Hz, IMU는 200 Hz인데, 본 프로젝트에서는 실제 목표 시스템에 맞춰 이미지만 10 Hz로 낮추고 그 사이의 모든 IMU sample은 보존했습니다. 학습은 Machine Hall의 medium sequence인 MH_03으로 수행했고, checkpoint 선택은 분리된 Vicon Room medium sequence 두 개에서 했습니다. 최종 수치는 V2_03 difficult에서 동일한 239개 frame으로 다시 비교했습니다.

### 발표자 이해용 메모

- EuRoC 공식 페이지는 20 FPS stereo image, 200 Hz MEMS IMU, temporal alignment, camera–IMU extrinsics, aligned ground truth를 명시한다. 본 프로젝트는 **left camera만 사용한 monocular-inertial 조건**이다.
- `MH_03_medium`을 선택한 이유: easy보다 motion/조명 변화가 있어 confidence 학습이 trivial하지 않고, difficult보다 짧은 stage-3 fine-tuning에서 초기화/추적 실패가 학습 신호를 지배할 위험이 작다. 즉, **난이도와 학습 안정성의 절충**이다.
- `V1_02_medium`, `V2_02_medium`을 둘 다 쓴 이유: checkpoint를 한 trajectory에 맞춰 고르는 것을 줄이고, medium 난이도에서 안정적으로 model selection하기 위해서다.
- `V2_03_difficult`을 최종 비교에 쓴 이유: easy가 아닌 어려운 motion/visual 조건에서 fixed·tuned·learned IMU weighting의 차이를 보기 위해서다. ORB-SLAM3 비교를 수행한 sequence와도 맞출 수 있다.
- **정직한 한계**: V2_03은 학습/validation raw data와는 겹치지 않지만, 이전 exploratory ablation에서 이미 결과를 확인했다. 따라서 발표에서는 엄밀한 “blind hold-out” 대신 **final re-evaluation**이라고 표현한다.

### 한 줄 결론

> EuRoC는 동기화·calibration·metric GT를 모두 제공하므로, “IMU를 실제로 올바르게 결합했는가”를 정량 검증하기 위한 최소 조건을 충족한다.

---

## 슬라이드 3. Baseline: DROID-SLAM에서 유지한 부분 ★

### 화면에 넣을 내용

```text
Image sequence
  → feature encoder (FNet / CNet)
  → correlation volume
  → ConvGRU update operator
  → correspondence correction Δ + visual confidence Wᵛ
  → Dense Bundle Adjustment
  → camera pose + inverse depth + dense reconstruction
```

**유지한 핵심**

- learned feature matching
- correlation 기반 motion estimation
- ConvGRU의 반복 update
- visual reprojection 기반 dense BA

**확장한 부분**

- IMU preintegration factor
- VI initialization 및 metric scale alignment
- velocity·bias를 포함한 15DoF state
- DVI-inspired scalar IMU confidence

### 발표 대본

> 제안 모델은 DROID를 새로 만든 모델이 아닙니다. DROID의 feature encoder, correlation, ConvGRU, dense bundle adjustment라는 시각 backbone은 유지했습니다. 그 위에 IMU preintegration factor와 visual–inertial initialization, 그리고 IMU 신뢰도 예측 head를 추가했습니다. 따라서 성능 차이가 visual backbone 교체 때문이 아니라 IMU 결합 방식 때문인지 분리해 볼 수 있습니다.

### 발표자 이해용 메모

- FNet은 feature map, CNet은 ConvGRU의 context/initial hidden state를 만든다.
- DROID는 한 번만 pose를 예측하지 않는다. `ConvGRU → BA`를 반복하면서 correspondence와 pose/depth를 계속 고친다.
- `Wᵛ`는 DROID가 원래 갖던 **픽셀/visual correspondence** confidence다. 새 `wᵘ`는 **시간 구간의 IMU factor** confidence다. 둘은 다르다.

---

## 슬라이드 4. 왜 IMU를 처음부터 바로 결합하지 않았는가? ★

### 화면에 넣을 내용

| 단안 초기 상태에서 모르는 값 | 바로 full VIO에 넣었을 때 문제 |
| --- | --- |
| 절대 scale | IMU position residual과 visual translation의 단위가 맞지 않음 |
| gravity 방향 | 가속도에서 중력과 실제 가속도를 분리하기 어려움 |
| velocity | position/velocity residual의 초기 오차가 큼 |
| gyro/accelerometer bias | 작은 bias가 누적되어 잘못된 motion constraint가 됨 |

**결론: “IMU를 더 많이 넣는 것”보다 “IMU를 넣기 전 상태를 초기화하는 것”이 우선**

### 발표 대본

> IMU를 raw 값으로 GRU에 바로 넣는 방식은 선택하지 않았습니다. 단안 visual trajectory가 아직 scale과 gravity, velocity, bias를 모르는 상태에서 IMU residual을 강하게 적용하면, IMU가 잘못된 초기 pose를 오히려 고정할 수 있기 때문입니다. 그래서 visual 결과를 먼저 안정화하고, 그 결과를 이용해 visual–IMU alignment를 수행한 뒤 full VIO 최적화로 전환하는 two-stage 구조를 사용했습니다.

### 발표자 이해용 메모

- IMU가 나빠서가 아니라, **초기값이 없는 비선형 최적화**가 불안정한 것이다.
- 이 문제는 VINS-Mono 등 전통 VIO도 visual–inertial initialization을 별도로 두는 이유다.
- 질문에 대한 짧은 답: “단안의 gauge freedom(scale·gravity·velocity·bias)을 해소한 다음 IMU factor를 활성화하기 위해 two-stage를 사용했습니다.”

---

## 슬라이드 5. 최종 모델 구조 ★

### 화면에 넣을 내용

삽입 그림: [`final_vi_architecture.svg`](presentation_assets/final_vi_architecture.svg)

그림 아래 한 줄:

> Raw IMU를 ConvGRU에 입력하지 않고, preintegrated IMU factor와 learned confidence를 **DBA의 최적화 항**으로 결합하였다.

### 발표 대본

> 왼쪽의 이미지 경로는 기존 DROID-SLAM입니다. 이미지는 feature, correlation, ConvGRU를 거쳐 visual correspondence correction과 visual confidence를 만들고 dense BA로 들어갑니다. 아래 IMU 경로에서는 gyro와 accelerometer를 카메라 프레임 사이에서 preintegrate해 상대 회전, 속도 변화, 위치 변화를 만듭니다. Visual warm-up 뒤 alignment가 scale, gravity, velocity, bias를 초기화하고, 마지막 15DoF DBA에서 visual reprojection과 IMU residual을 함께 최소화합니다. ConvGRU hidden feature에서 나온 scalar confidence는 IMU factor의 가중치만 조절합니다.

### 발표자 이해용 메모

- `ΔR, Δv, Δp`는 각각 카메라 두 프레임 사이의 회전 변화, 속도 변화, 위치 변화를 IMU로 누적한 값이다.
- `P, J_b`는 각각 preintegration 불확실도/정보와 bias 변화에 대한 1차 보정 Jacobian이다.
- confidence branch가 **IMU 데이터를 생성하는 것**이 아니라, “현재 visual 상태에서 이 IMU constraint를 얼마나 믿을까”를 조절한다.

---

## 슬라이드 6. IMU 전처리: Preintegration이 하는 일

### 화면에 넣을 내용

**센서 모델**

\[
\omega_m = \omega + b_g + n_g, \qquad
a_m = a + b_a + n_a
\]

**프레임 \(i\rightarrow j\) 사이에 저장하는 값**

\[
(\Delta R_{ij},\; \Delta v_{ij},\; \Delta p_{ij},\; \Sigma_{ij},\; J_{b,ij})
\]

- \(\Delta R, \Delta v, \Delta p\): rotation / velocity / position increment
- \(\Sigma\) 또는 information: 해당 구간 IMU 노이즈에 따른 신뢰도
- \(J_b\): bias가 바뀌었을 때 preintegration을 1차 보정하기 위한 Jacobian

**10 Hz image 조건**

- EuRoC 원본 20 Hz camera에서 매 2번째 이미지 사용 → 10 Hz
- 이미지 사이의 **모든 고주파 IMU sample은 유지**하여 적분
- 학습 clip 간격: 3/4/5 frames → 0.3/0.4/0.5 s motion을 함께 학습

### 발표 대본

> IMU는 한 샘플을 image feature처럼 사용하지 않았습니다. 카메라 두 프레임 사이에 있는 여러 gyro·accelerometer sample을 적분해 상대 motion measurement를 만들었습니다. 이때 센서 노이즈와 bias의 영향을 함께 전파해 covariance와 bias Jacobian도 보존했습니다. 이미지는 실제 목표 시스템과 맞추기 위해 10 Hz로 사용했지만, 그 사이의 IMU sample을 버리지는 않았습니다. 학습에서는 0.3초, 0.4초, 0.5초 구간을 섞어 여러 motion scale에서 confidence가 동작하도록 했습니다.

### 발표자 이해용 메모

- `3/4/5`는 10 Hz 입력 기준 **frame gap**이다. 5~10fps라는 뜻이 아니다.
- preintegration의 장점은 BA가 pose/bias를 반복 갱신해도 원시 IMU를 매번 전부 적분하지 않고 Jacobian으로 보정할 수 있다는 점이다.
- covariance weighting은 이번 최종 모델에서 사용한다. 단순히 IMU residual 크기만 쓰지 않고, 긴 interval/노이즈가 큰 interval의 정보량을 낮춘다.

---

## 슬라이드 7. Two-stage runtime: 실제 추정 순서 ★

### 화면에 넣을 내용

```text
Stage 1. Visual warm-up
  Images → DROID visual DBA → relative pose + inverse depth
                                      │
                                      ▼
VI alignment
  visual trajectory + IMU preintegration
  → metric scale, gravity, v, b_a, b_g initialization
                                      │
                                      ▼
Stage 2. Full VIO DBA
  visual reprojection + IMU preintegration factor
  → pose, depth, velocity, local bias jointly refined
```

**Stage 1에서는 IMU factor를 의도적으로 비활성화**

### 발표 대본

> 첫 단계에서는 visual pose와 inverse depth만 DROID 방식으로 업데이트합니다. 그 다음 visual trajectory와 preintegrated IMU를 정렬해 metric scale, 중력, velocity, accelerometer bias, gyroscope bias의 초기값을 구합니다. 두 번째 단계에서야 pose뿐 아니라 velocity와 bias를 포함한 full 15DoF state를 DBA가 함께 보정합니다. 코드에서도 pose-only stage는 full IMU factor와 rotation-only prior 모두 끄고, full 15DoF stage에서만 IMU factor를 활성화하도록 분리했습니다.

### 발표자 이해용 메모

- two-stage의 핵심은 **시작부터 IMU를 약하게 쓴다**가 아니라, stage 1에서 **IMU factor 자체를 끈다**는 점이다.
- VI alignment가 충분한 motion/parallax를 얻기 전에는 연기될 수 있다. 이는 “실패 후 visual-only로 몰래 처리”가 아니라, 관측 가능성이 낮은 구간을 분리하는 안전장치다.
- `full 15DoF` 이후 final global backend refinement가 추가로 한 번 수행될 수 있다. 발표에서는 “local–global DBA refinement” 한 줄 정도로만 언급해도 된다.

---

## 슬라이드 8. 15DoF DBA: 무엇을 함께 최적화하는가? ★

### 화면에 넣을 내용

키프레임 \(k\)의 IMU motion state:

\[
x_k = (T_k,\; v_k,\; b_{a,k},\; b_{g,k})
\]

| 항 | 자유도 | 역할 |
| --- | ---: | --- |
| \(T_k\) | 6 | camera pose: translation + rotation |
| \(v_k\) | 3 | velocity |
| \(b_{a,k}\) | 3 | accelerometer bias |
| \(b_{g,k}\) | 3 | gyroscope bias |
| 합계 | **15** | full IMU motion state |

최적화 목적의 개념:

\[
E = E_{visual} + \lambda_{imu}\sum_{(i,j)} w_{ij}^{u}\, \lVert r_{imu}^{ij}\rVert^2_{\tilde{\Sigma}_{ij}^{-1}}
\]

\[
r_{imu} = [r_p, r_v, r_R, r_{b_a}, r_{b_g}]
\]

### 발표 대본

> 기존 DROID의 DBA는 주로 pose와 inverse depth를 visual reprojection에 맞게 조정합니다. 우리는 키프레임마다 velocity, accelerometer bias, gyroscope bias를 추가해 15DoF state를 만들었습니다. IMU factor는 position, velocity, rotation, 그리고 bias random-walk residual을 포함합니다. 최적화는 visual reprojection error와 IMU residual을 함께 줄이며, IMU 항에는 전역 세기와 confidence, preintegration information이 결합됩니다.

### 발표자 이해용 메모

- 위 식은 **개념식**이다. 구현에서는 residual/Jacobian에 각 항의 square-root weight를 곱해 normal equation을 만든다. 발표에서는 이 상세까지 말할 필요 없다.
- `\tilde{\Sigma}^{-1}`은 covariance-derived information을 interval 기준으로 정규화한 값이다.
- `\lambda_{imu}=0.002`는 confidence가 아니라, visual과 IMU의 전체 균형을 정하는 전역 scale이다.
- DROID의 inverse depth는 15DoF에 포함시키지 않았다. 기존 dense BA의 depth block에서 계속 업데이트된다.

---

## 슬라이드 9. DVI-inspired learned IMU confidence: 무엇을 학습했는가? ★

### 화면에 넣을 내용

```text
ConvGRU hidden feature hᵢⱼ
      │
      ├─ original DROID heads → Δ (correspondence), Wᵛ (visual confidence)
      │
      └─ added DVI-style head
          1×1 Conv (128 → 1)
          → spatial average pooling
          → sigmoid
          → scalar wᵘᵢⱼ ∈ [0, 1]
```

IMU factor의 유효 세기:

\[
\text{IMU strength} \propto
\underbrace{0.002}_{\text{global prior weight}}
\times
\underbrace{w^u}_{\text{learned confidence}}
\times
\underbrace{\text{information}}_{\text{preintegration uncertainty}}
\]

| 방식 | \(w^u\) |
| --- | --- |
| Fixed | 항상 1.0 |
| Tuned | 사람이 정한 값/스케줄 |
| Learned | ConvGRU feature에서 매 구간 예측 |

### 발표 대본

> DVI-SLAM의 아이디어를 따라 ConvGRU의 hidden feature에서 scalar IMU confidence를 추가로 예측했습니다. 구조는 128채널 feature에 1×1 convolution을 적용하고 spatial average pooling과 sigmoid를 거치는 매우 작은 head입니다. 이 값은 raw IMU나 bias를 대신하는 값이 아닙니다. preintegration uncertainty와 전역 IMU weight가 정한 기본 신뢰도 위에서, 현재 visual tracking 상태에 따라 IMU residual을 상대적으로 얼마나 반영할지를 조절합니다. Fixed는 항상 1을 쓰고, tuned는 사람이 정하며, learned만 이 값을 네트워크가 예측합니다.

### 발표자 이해용 메모

- 최종 학습 대상은 이 head의 **129개 parameter**뿐이다: 1×1 Conv의 `128 weight + 1 bias`.
- 0.002가 “IMU의 0.2%만 사용한다”는 뜻은 아니다. visual normal equation과 IMU normal equation의 수치 scale이 달라서 쓰는 **상대 최적화 계수**다.
- scalar `w^u=0.85`이면 해당 IMU factor를 85% 신뢰한다는 개념적 해석은 가능하지만, 최종 영향은 covariance/info와 residual weight도 함께 결정한다.
- 최신 결과에서 mean confidence는 `0.855`다. 이전 learned의 `0.932`보다 덜 포화됐지만 여전히 높은 편이므로, 미래에는 더 다양한 motion/noise 조건 학습이 필요하다.

---

## 슬라이드 10. 학습(stage-3): 무엇을 고정하고 무엇을 업데이트했는가? ★

### 화면에 넣을 내용

**데이터**

- training: EuRoC `MH_03_medium`의 실제 camera + IMU + GT pose
- image rate: 10 Hz (원본 20 Hz에서 2배 decimation, IMU는 모두 보존)
- clip: 7 frames, temporal gap 3/4/5 frames 혼합
- training: 8,000 steps, learning rate `1e-4`

**고정(freeze)**

- DROID visual backbone: encoder / correlation / ConvGRU / visual heads
- global IMU bias
- global IMU prior weight: `0.002`

**학습(update)**

- DVI-style scalar confidence head 129 parameters
- 시작 출력: sigmoid 0.5가 되도록 마지막 projection reset

### 발표 대본

> 학습은 DROID 전체를 다시 학습하지 않았습니다. 사전학습된 visual backbone과 global bias, 전역 IMU weight는 고정하고 scalar confidence head 129개 parameter만 fine-tuning했습니다. 데이터는 실제 synchronized camera/IMU와 GT pose가 있는 EuRoC MH_03_medium을 사용했습니다. 10 Hz 이미지 입력을 유지하되, preintegration에는 모든 IMU sample을 사용했습니다. 마지막 1×1 confidence projection은 출력이 0.5가 되도록 초기화해, 기존 checkpoint의 과도한 IMU 신뢰도를 그대로 이어받지 않도록 했습니다.

### 발표자 이해용 메모

- 이 방식의 장점: “visual network가 바뀌어서 결과가 변했다”는 변수를 줄이고, confidence 효과만 확인할 수 있다.
- 단점: visual feature 자체가 IMU confidence를 구분하기에 충분하지 않으면 129 parameter만으로 개선 폭이 제한될 수 있다.
- EuRoC는 dense optical flow/depth GT 형식이 없어서, 여기서는 **GT pose + visual reprojection residual + full IMU preintegration residual**을 주 감독 신호로 쓴다. GT dense flow를 사용한 것이 아니다.

---

## 슬라이드 11. 학습 손실과 two-stage unroll

### 화면에 넣을 내용

학습 중 1개 clip에서 DROID update/DBA를 7회 unroll:

```text
1–3회: visual pose/depth-only update
4–7회: full 15DoF IMU DBA update
       weak confidence targets: 0.25 → 0.50 → 0.75 → 1.00
```

\[
\mathcal{L} =
10\mathcal{L}_{geo}
+0.01\mathcal{L}_{res}
+0.05\mathcal{L}_{imu}
+0.02\mathcal{L}_{conf\text{-}NLL}
+0.01\mathcal{L}_{sat}
+0.05\mathcal{L}_{schedule}
\]

| 손실 | 역할 |
| --- | --- |
| \(\mathcal{L}_{geo}\) | GT pose와 추정 pose의 기하학적 일치 |
| \(\mathcal{L}_{res}\) | DROID visual reprojection residual 안정화 |
| \(\mathcal{L}_{imu}\) | position/velocity/rotation/bias preintegration consistency |
| \(\mathcal{L}_{conf\text{-}NLL}\) | IMU residual 크기와 confidence calibration 연결 |
| \(\mathcal{L}_{sat}\) | confidence가 0.90 이상으로 포화되는 현상 억제 |
| \(\mathcal{L}_{schedule}\) | 약한 curriculum: 네 번의 full-IMU update 평균 confidence를 0.25→1.0 방향으로 유도 |

### 발표 대본

> 학습에서도 runtime two-stage와 같은 순서를 사용했습니다. 한 clip을 일곱 번 unroll하며 첫 세 번은 visual-only, 이후 네 번은 full 15DoF IMU DBA를 수행합니다. 손실은 pose, visual residual, full preintegration consistency를 기본으로 하고, confidence가 무조건 1에 붙지 않도록 saturation penalty와 confidence NLL을 추가했습니다. 0.25에서 1.0으로 증가하는 schedule은 정답 confidence를 강제로 지정하는 것이 아니라, confidence가 모든 구간에서 무조건 포화되는 것을 막기 위해 사용한 약한 curriculum입니다.

### 발표자 이해용 메모

- “0.25, 0.5, 0.75, 1.0을 네 번의 추론에서 순서대로 쓴다”가 아니다. **학습 중 네 개 full-IMU unroll의 평균 confidence에 걸린 약한 보조 손실**이다.
- 최종 runtime learned mode는 수동 schedule을 재생하지 않는다. ConvGRU feature에서 `w^u`를 예측한다.
- 식의 수치는 현재 실행 스크립트의 weights를 그대로 옮겼다. `L_flow`는 EuRoC stage-3에서 dense depth GT가 없어 비활성이다.
- 발표가 길어지면 이 슬라이드는 appendix로 보내도 된다.

---

## 슬라이드 12. 실패 진단: 왜 이전 learned가 fixed보다 나빴는가? ★

### 화면에 넣을 내용

| 이전 learned에서 관찰한 현상 | 원인 가설 | 수정 |
| --- | --- | --- |
| mean \(w^u\) = 0.932, 거의 항상 IMU를 과신 | confidence head 포화 | 마지막 projection을 0.5로 reset, saturation regularizer |
| fixed보다 높은 ATE | 학습/추론의 IMU 상대 세기 불일치 가능성 | train/inference 모두 \(\lambda_{imu}=0.002\) 고정 |
| visual network까지 변하면 원인 분리 어려움 | 학습 범위 과다 | visual backbone·global bias·global weight freeze |

**parity training의 의미**

> 학습 조건과 배포 조건의 IMU weighting을 맞춘 상태에서, confidence head의 효과만 다시 검증하였다.

### 발표 대본

> 초기 learned 모델은 mean confidence가 약 0.93으로 매우 높았고, 결과적으로 fixed control보다도 metric ATE가 나빴습니다. 이를 단순히 “학습이 실패했다”고 끝내지 않고, confidence 포화와 학습–추론 간 IMU weight 불일치를 원인 후보로 분리했습니다. 이후 train과 inference 모두 전역 IMU weight를 0.002로 맞추고, confidence head를 0.5에서 다시 시작하며 visual backbone은 고정했습니다. 이를 parity training이라고 부르고 다시 검증했습니다.

### 발표자 이해용 메모

- 이 슬라이드는 평가 기준의 ‘예기치 못한 상황 대응/문제 해결 능력’을 보여 주는 가장 좋은 근거다.
- “parity”는 두 모델이 같다는 뜻이 아니라, **학습과 추론 조건을 일치시켜 비교가 공정해졌다는 뜻**이다.
- 두 번째 revised learned 모델은 tuned보다 아주 근소하게 높지만 fixed보다 개선됐으므로, “완전한 최적 모델”보다 “실패 원인을 검증 가능한 방식으로 개선한 결과”로 표현하는 게 정확하다.

---

## 슬라이드 13. 실험 설계와 공정성 ★

### 화면에 넣을 내용

**학습/선정/평가 역할**

| 역할 | 시퀀스 | 사용 목적 |
| --- | --- | --- |
| stage-3 training | `MH_03_medium` | confidence head fine-tuning |
| validation / checkpoint selection | `V1_02_medium`, `V2_02_medium` | 2k/8k 중 checkpoint 선택 |
| final re-evaluation | `V2_03_difficult` | 동일 조건 final comparison |

**비교 조건**

| 방법 | two-stage 15DoF | IMU confidence |
| --- | --- | --- |
| Original | 아니오 | IMU 없음 |
| Fixed 15DoF | 예 | \(w^u=1\) |
| Tuned 15DoF | 예 | 수동 지정 |
| Old learned | 예 | 이전 scalar head |
| Parity learned 2k | 예 | 재학습 scalar head |

**공정성 원칙**: V2_03의 동일한 reconstruction frame ID **239개**에서 모든 수치를 계산

### 발표 대본

> confidence 학습은 Machine Hall 계열의 실제 IMU sequence에서 수행하고, checkpoint 선택은 Vicon Room 계열 두 sequence에서 분리했습니다. 최종 비교는 V2_03 difficult의 동일한 239개 frame ID로 맞췄습니다. Fixed, tuned, learned는 모두 같은 two-stage 15DoF 구조와 동일 IMU prior weight를 쓰고 confidence 적용 방식만 다릅니다. 따라서 fixed와 learned의 차이는 confidence 전략의 차이로 해석할 수 있습니다.

### 발표자 이해용 메모

- 엄밀하게는 V2_03에서 앞선 탐색 실험을 이미 보았으므로, 최종 논문에서는 완전히 blind hold-out이라 부르지 말고 **final re-evaluation**이라고 부르는 편이 정직하다.
- Original은 metric scale이 없어서 metric SE(3) ATE를 직접 계산할 수 없다. original은 Sim(3) shape 지표에서만 비교한다.

---

## 슬라이드 14. 최종 정량 결과: IMU confidence 효과 ★

### 화면에 넣을 내용

**EuRoC V2_03_difficult, common 239 frames, metric SE(3)**

| 방법 | ATE RMSE ↓ | RPE 1 s ↓ | RPE 5 s ↓ | scale error ↓ | mean \(w^u\) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 15DoF | 0.380007 m | 0.172874 m | 0.502313 m | 0.207948 | 1.000 |
| Tuned 15DoF | **0.366563 m** | **0.166692 m** | **0.484578 m** | **0.199109** | 0.516 |
| Old learned | 0.395872 m | 0.179980 m | 0.523240 m | 0.218543 | 0.932 |
| **Parity learned 2k** | **0.367220 m** | **0.167080 m** | **0.485570 m** | **0.199533** | 0.855 |

강조할 숫자:

- Parity learned vs fixed: ATE **3.36% 감소**
- Parity learned vs old learned: ATE **7.24% 감소**, scale error **8.70% 감소**
- Parity learned vs tuned: ATE 차이 **0.000658 m (0.18%)**

### 발표 대본

> 최종 비교에서 parity learned 2k는 fixed 15DoF보다 ATE를 3.36% 낮췄고, 이전 learned 대비로는 7.24% 낮췄습니다. 특히 수동 tuned와의 ATE 차이는 0.000658 m, 0.18%에 불과했습니다. 즉, 사람이 정한 tuned confidence의 성능에 learned confidence가 거의 도달했다는 결과입니다. 또한 confidence 평균이 0.932에서 0.855로 내려가, 이전처럼 IMU를 거의 항상 최대 신뢰도로 사용하던 포화 문제는 완화됐습니다.

### 발표자 이해용 메모

- **가장 좋은 ATE는 tuned**다. 이 사실을 숨기지 말고, learned가 tuned와 근접하면서 fixed를 이긴 점을 주장한다.
- ATE는 전체 trajectory 위치 오차, RPE는 일정 시간 간격에서의 local drift다.
- scale error는 VI alignment로 얻은 metric scale의 상대 오차다. 작을수록 좋다.
- 결과 근거 파일: `outputs/dvi_parity_2k_v2_03_learned_only/summary_common.md`

---

## 슬라이드 15. Original DROID와의 비교: 무엇을 주장할 수 있고, 없는가

### 화면에 넣을 내용

| 비교 관점 | Original DROID | Parity learned VIO | 해석 |
| --- | ---: | ---: | --- |
| metric SE(3) ATE | 측정 불가 | 0.367220 m | Original은 monocular scale이 관측되지 않음 |
| Sim(3) ATE (shape) | **0.014385 m** | 0.015849 m | visual shape만 보면 original이 약간 우세 |
| metric scale | 없음 | VI alignment로 확보 | VIO extension의 핵심 산출물 |

**정확한 결론**

> Original DROID의 시각 trajectory shape를 넘어섰다고 주장하지 않는다.  
> 대신 IMU를 결합해 metric trajectory를 생성했고, learned confidence로 fixed IMU 결합 대비 성능을 개선했다.

### 발표 대본

> Original DROID와 VIO를 하나의 ATE 숫자로 직접 비교하는 것은 공정하지 않습니다. Original monocular DROID는 scale이 관측되지 않기 때문에 Sim(3) scale alignment 후 shape만 비교할 수 있습니다. 이 지표에서는 original이 조금 더 좋습니다. 따라서 우리는 original visual 성능 자체를 이겼다고 주장하지 않습니다. 이 프로젝트의 확인된 성과는 DROID에 IMU를 결합해 metric trajectory를 만들고, 그 IMU 결합의 confidence를 learned 방식으로 조절해 fixed control보다 개선한 것입니다.

### 발표자 이해용 메모

- 이 슬라이드는 오히려 신뢰도를 높인다. 심사자가 “왜 original이 더 낮은가?”라고 물어도 먼저 답한 구조가 된다.
- Original, ORB-SLAM3, 제안 모델의 결과를 같은 표에 억지로 섞지 않는다. 정렬 방식과 sensor mode가 다르므로 별도 표/각주를 쓴다.

---

## 슬라이드 16. 현재 한계와 다음 개선 방향 ★

### 화면에 넣을 내용

**현재 확인한 성과**

- DROID visual backbone 유지 + full IMU preintegration factor 결합
- two-stage initialization으로 full 15DoF DBA 안정화
- parity learned confidence가 fixed 대비 ATE 3.36% 개선
- learned가 manual tuned 성능에 근접

**남은 한계**

- 최종 learned가 manual tuned를 아직 넘지는 못함
- learned confidence 평균이 아직 높은 편(0.855)
- visual shape Sim(3)은 original보다 소폭 낮음
- 단일 training sequence(`MH_03`)에 의존한 stage-3 fine-tuning
- 완전히 보지 않은 blind test 분리를 더 엄격하게 구성할 필요

**다음 단계**

1. MH 다중 sequence / mobile IMU를 추가한 confidence 학습
2. scale·bias·confidence의 sequence별 진단과 domain-shift 평가
3. scalar가 아니라 rotation/velocity/position component-wise confidence 비교
4. 모바일 end-to-end input에서 timestamp synchronization 및 real-time budget 검증

### 발표 대본

> 이번 결과는 learned confidence가 fixed control보다 개선되고 manual tuned에 근접할 수 있음을 보였지만, tuned를 완전히 넘었다고 말할 단계는 아닙니다. confidence도 평균 0.855로 여전히 높은 편이고, visual shape 지표에서는 original보다 약간 낮습니다. 다음 단계에서는 여러 실제 IMU sequence와 모바일 데이터로 학습 범위를 넓히고, rotation·velocity·position을 각각 다르게 다루는 confidence를 비교하며, 실제 모바일 end-to-end 조건에서 동기화와 실시간성을 검증할 계획입니다.

---

# 부록: 질문에 바로 답하기 위한 Q&A 카드

## Q1. “왜 raw IMU를 ConvGRU에 넣지 않았나?”

**짧은 답**: IMU는 프레임 사이의 고주파 motion constraint이므로 preintegration residual로 DBA에 넣는 것이 bias와 covariance를 다루기 적합하다. ConvGRU에는 raw sensor를 넣지 않고 visual recurrent feature에서 IMU factor의 신뢰도만 예측했다.

## Q2. “0.002는 IMU를 0.2%만 쓴다는 뜻인가?”

**짧은 답**: 아니다. visual residual과 IMU residual의 단위·차원·Hessian scale이 다르므로 둘의 상대적 영향을 맞추는 전역 최적화 계수다. 실제 factor 영향은 `0.002 × confidence × covariance-derived information`으로 함께 결정된다.

## Q3. “learned confidence는 무엇을 보고 무엇을 출력하나?”

**짧은 답**: ConvGRU의 128-channel hidden feature를 1×1 convolution과 global average pooling에 통과시켜, 인접 frame IMU factor 하나당 0~1 scalar를 출력한다. IMU 값 자체를 바꾸지 않고 그 residual의 가중치만 조절한다.

## Q4. “왜 confidence가 1이면 안 좋은가?”

**짧은 답**: 1 자체가 항상 나쁜 것은 아니다. 그러나 모든 구간에서 거의 1이면, motion·tracking 상태에 따라 IMU를 구분해서 신뢰한다는 head의 목적이 사라지고 sensor/model mismatch를 완화할 여지가 없다. 이전 learned 평균 0.932가 이 경우였다.

## Q5. “two-stage는 학습 단계인가?”

**짧은 답**: 아니다. two-stage는 runtime에서 visual-only warm-up 후 15DoF VIO DBA로 전환하는 추정 절차다. stage-3는 confidence head를 fine-tuning한 학습 절차다.

## Q6. “왜 15DoF인가?”

**짧은 답**: pose만으로는 acceleration bias와 velocity가 일으키는 position drift를 설명할 수 없기 때문이다. 15DoF는 pose 6, velocity 3, accelerometer bias 3, gyroscope bias 3을 함께 최적화한다.

## Q7. “Original보다 좋아졌다고 할 수 있나?”

**짧은 답**: visual shape 기준으로 original을 넘었다고는 말하지 않는다. Original은 metric scale이 없기 때문에 VIO metric ATE와 직접 비교할 수 없다. 본 결과는 IMU 결합 system 안에서 learned confidence가 fixed control보다 개선됐음을 보인 것이다.

---

# 발표 시 표현 가이드

## 사용해도 되는 표현

- “DROID visual backbone을 유지한 Visual–Inertial extension을 구현했다.”
- “IMU preintegration factor를 full 15DoF DBA에 결합했다.”
- “학습–추론 조건을 일치시킨 parity training 후, learned confidence는 fixed 대비 metric ATE를 3.36% 개선했다.”
- “Manual tuned confidence에 0.18% 차이까지 근접했다.”
- “현재 결과는 개선 가능성을 보인 중간/최종 프로토타입이며, 일반화를 위한 추가 검증이 남아 있다.”

## 피해야 하는 표현

- “Original DROID보다 성능이 좋다.”
- “0.002만큼의 IMU를 사용했다.”
- “confidence가 IMU noise를 직접 예측한다.”
- “학습에서 0.25→1.0으로 runtime confidence를 순차적으로 올린다.”
- “V2_03이 완전히 보지 않은 blind test다.”

# Chapter 3 최종안 — “결과가 다음 실험을 만든 과정”

이 문서만 Chapter 3 발표 원고로 사용한다.  
이전의 상세 구조/용어 문서는 **부록 자료**로 두고, 본문은 아래의 인과관계만 전달한다.

```text
Baseline의 한계 확인
  → [실험 1] IMU를 초기화에만 사용
  → 결과: scale은 만들지만 trajectory를 고치지 못함
  → [실험 2] IMU factor를 9DoF DBA 내부로 이동
  → 결과: factor는 동작하지만 bias가 고정되어 개선이 작음
  → [실험 3] bias를 포함하는 two-stage 15DoF로 확장
  → 결과: full VIO 구조는 동작, 추가 bias refinement는 이득 불확실
  → [실험 4] IMU confidence를 학습
  → 결과: 초기 learned confidence가 포화되어 fixed보다 악화
  → [실험 5] train–inference parity 재설계 및 재학습
  → 결과: fixed 대비 ATE 3.36% 개선, tuned에 근접
```

---

## 슬라이드 1. 출발점: DROID baseline에서 무엇이 부족했는가?

### 슬라이드 화면

`droid_baseline_vs_vi_extension_ko.svg` 그림을 넣되, 이 슬라이드에서는 **파란색 영역만 먼저 강조**한다.

| Original DROID-SLAM이 하는 일 | 남는 문제 |
| --- | --- |
| image feature → correlation → ConvGRU → dense visual BA | 단안이므로 metric scale을 직접 관측하지 못함 |
| pose와 inverse depth를 visual reprojection으로 반복 보정 | 빠른 움직임·blur·저텍스처에서는 visual correspondence가 약해질 수 있음 |

**연구 질문**

> IMU를 “시작할 때 scale을 맞추는 보조 정보”로만 쓸 것인가, 아니면 visual optimization 내부에서 지속적으로 결합할 것인가?

### 발표 대본

> 출발점은 원본 DROID-SLAM입니다. DROID는 이미지 feature와 correlation, ConvGRU, dense bundle adjustment를 통해 pose와 depth를 매우 잘 추정하지만, 단안 입력만으로는 절대 scale을 알 수 없습니다. 그래서 IMU를 어떻게 결합할지 단계적으로 탐색했습니다. 핵심 질문은 IMU가 초기화만 도우면 충분한지, 아니면 DBA 내부의 최적화 항으로 들어가야 하는지였습니다.

### 이 슬라이드가 다음 실험으로 연결하는 문장

> “먼저 가장 보수적인 방식인 **IMU 초기화-only**를 시험했습니다.”

---

## 슬라이드 2. 실험 1: IMU를 초기화에만 사용한 방식

### 실험 질문

> visual DROID trajectory는 그대로 두고, IMU로 scale·gravity·velocity·bias만 초기화하면 metric VIO가 되는가?

### 구현 방식

```text
image → DROID online local visual DBA → 잠정 relative trajectory
                                          ↓
IMU preintegration ─────────────────→ VI alignment
                                      scale / gravity / v / bₐ / b_g 추정
                                          ↓
                              metric scale로 변환된 visual state
                                          ↓
                            이후에도 visual-only DBA 계속
```

구조도: [`experiment1_vi_initialization_only_ko.svg`](presentation_assets/experiment1_vi_initialization_only_ko.svg)

여기서 visual BA는 **최종적으로 한 번 끝난 뒤** 실행되는 단계가 아니다. 매 keyframe의 online local update에서 visual DBA가 먼저 잠정 pose/depth를 갱신하고, 충분한 motion excitation이 생긴 시점에 그 현재 상태를 IMU preintegration과 정렬한다. 초기화-only 대조군에서는 이 뒤의 DBA에도 IMU residual을 넣지 않는다.

### 결과

| 확인한 것 | 결과 |
| --- | --- |
| IMU–visual alignment | scale, gravity, initial velocity, bias seed 계산 가능 |
| metric trajectory | visual trajectory를 metre 단위로 해석할 기반 확보 |
| visual drift 보정 | **불가** — IMU가 DROID DBA update에는 들어가지 않음 |

### 해석 → 다음 실험의 이유

> 초기화-only 방식은 “scale을 얻는 문제”에는 답하지만, visual tracking이 흔들리는 구간의 pose/depth를 수정할 경로가 없다. 따라서 IMU residual을 **DBA 내부 factor**로 넣는 9DoF 방식을 탐구했다.

### 발표 대본

> 첫 번째 방식은 IMU를 초기화에만 사용하는 방법입니다. visual trajectory와 IMU preintegration을 정렬해 scale, 중력, velocity, bias의 시작값을 얻었습니다. 이 방식으로 metric scale을 만들 수는 있었지만, 이후 DROID의 pose와 depth 업데이트는 여전히 visual residual만 보고 수행됩니다. 즉, visual drift를 IMU가 직접 고칠 수 없었습니다. 그래서 다음 실험에서는 IMU를 사후 정렬이 아니라 DBA 내부 factor로 옮겼습니다.

### 발표자 주의

- 이 단계는 “성능이 몇 % 좋아졌다”가 아니라 **IMU의 역할을 분리하기 위한 구조 실험**이다.
- Original monocular DROID에는 metric scale이 없으므로, metric ATE를 original과 직접 비교해 우열을 말하지 않는다.

---

## 슬라이드 3. 실험 2: IMU factor를 9DoF DBA 내부에 결합

### 실험 질문

> IMU preintegration을 DBA 안에 넣으면, visual pose와 IMU motion constraint가 함께 optimization될 수 있는가?

### 설계

| 항목 | 9DoF 방식 |
| --- | --- |
| 최적화 state | pose 6 + velocity 3 |
| IMU bias | VI initialization에서 얻은 값으로 고정 |
| IMU residual | rotation + velocity + position preintegration residual |
| 결합 위치 | visual BA normal equation의 Hessian/gradient에 IMU factor 추가 |

### 구현 검증과 결과

| 검증 항목 | 결과 |
| --- | --- |
| IMU factor가 solver에 반영되는가? | temporal IMU edge 사용률 약 99.9% 확인 |
| 좌표계가 맞는가? | visual–IMU relative rotation mismatch **6.98° → 0.385°** 수정 |
| Sim(3) visual shape가 의미 있게 좋아졌는가? | V2_01/V2_03에서 약 **0.09% / 0.17%** 수준 개선 — 유의미한 개선으로 주장하지 않음 |

### 해석 → 다음 실험의 이유

> IMU factor가 실제 optimization에 들어가는 것과 좌표계 정합은 확인했다. 하지만 bias를 고정한 9DoF만으로는 metric drift를 충분히 줄이지 못했다. 따라서 **velocity뿐 아니라 accel/gyro bias도 함께 최적화하는 15DoF**를 설계했다.

### 발표 대본

> 두 번째 실험에서는 IMU를 DBA 내부에 넣었습니다. 여기서는 pose와 velocity를 함께 최적화하는 9DoF state를 사용했고, IMU의 rotation·velocity·position residual이 visual Hessian에 더해지도록 구현했습니다. factor 사용률은 거의 100%였고, 좌표계 오류도 6.98도에서 0.385도로 줄였습니다. 다만 visual shape ATE 개선은 0.1% 수준으로 작았습니다. 그래서 단순히 factor가 들어간다는 것만으로는 부족하고, IMU bias를 고정한 것이 한계일 수 있다고 판단했습니다.

---

## 슬라이드 4. 실험 3: 15DoF two-stage 구조로 확장

### 왜 구조를 바꿨는가?

9DoF에서는 bias를 고정한다. 하지만 accelerometer/gyroscope bias는 누적되어 position·velocity residual을 틀리게 만들 수 있다. 또한 단안 시작 구간에서는 scale·gravity·velocity·bias의 초기값이 불안정하다.

### 새 설계

이 슬라이드에서 [`droid_baseline_vs_vi_extension_ko.svg`](presentation_assets/droid_baseline_vs_vi_extension_ko.svg)를 전체 색상으로 보여 준다.

```text
Stage 1: 기존 DROID visual warm-up
             ↓
VI initialization: scale, gravity, velocity, accel/gyro bias 초기화
             ↓
Stage 2: full 15DoF DBA
         pose(6) + velocity(3) + accel bias(3) + gyro bias(3)
```

### 결과

**V2_03 common-frame exploratory comparison**

| 조건 | Metric SE(3) ATE |
| --- | ---: |
| Fixed 9DoF | 0.388726 m |
| Fixed 15DoF two-stage | 0.380986 m |

→ 15DoF two-stage가 이 조건에서 약 **1.99% 낮은 ATE**를 보였다.

**추가 global-bias refinement 실험**

| 조건 | Metric SE(3) ATE |
| --- | ---: |
| Two-stage pass-0 | 0.222643 m |
| global-bias refinement 후 | 0.226814 m |

→ bias state를 더 조정하는 것만으로는 이득이 확인되지 않았다.

### 해석 → 다음 실험의 이유

> 15DoF two-stage는 9DoF보다 더 나은 metric 결과를 보였고, full VIO state를 다루는 최종 구조로 채택했다. 반면 bias를 더 세게 최적화하는 방식은 효과가 없었다. 따라서 다음 질문은 “IMU를 얼마나 강하게 믿어야 하는가?”가 되었고, DVI 방식의 learned confidence를 학습했다.

### 발표 대본

> 9DoF의 한계를 해결하기 위해 acceleration과 gyro bias까지 state에 포함한 15DoF two-stage 구조를 만들었습니다. 먼저 visual warm-up으로 geometry를 안정화하고, alignment로 scale과 IMU state를 초기화한 뒤 full VIO DBA를 수행합니다. fixed 9DoF와 비교하면 15DoF가 ATE를 약 2% 낮췄습니다. 하지만 global bias refinement를 추가했을 때는 오히려 ATE가 낮아지지 않았습니다. 따라서 복잡한 bias update를 계속 늘리기보다, 매 구간에서 IMU factor를 얼마나 신뢰할지 조절하는 confidence 방향으로 넘어갔습니다.

### 발표자 주의

- 이 슬라이드의 두 표는 **서로 다른 실험 묶음**이다. 한 표에 합치지 않는다.
- “global bias refinement는 실패했다”가 아니라, “독립적 이득을 확인하지 못해서 최종 핵심 기여로 채택하지 않았다”고 표현한다.

---

## 슬라이드 5. 실험 4: DVI-style learned IMU confidence 학습

### 실험 질문

> preintegration covariance가 주는 기본 신뢰도 위에서, visual tracking 상태에 맞춰 IMU factor의 상대 가중치를 네트워크가 스스로 조절할 수 있는가?

### 알고리즘 설계

```text
ConvGRU hidden feature
  → 1×1 convolution
  → spatial average pooling + sigmoid
  → scalar confidence wᵘ ∈ [0, 1]
  → Stage 2 IMU residual의 weight 조절
```

`wᵘ`는 IMU raw value, scale, bias를 예측하지 않는다. IMU preintegration factor를 **얼마나 반영할지**만 조절한다.

### 실제 training 데이터와 방법

| 항목 | 수행 내용 |
| --- | --- |
| 학습 데이터 | EuRoC `MH_03_medium`의 실제 camera + IMU + GT |
| 입력 구성 | 20 Hz camera를 10 Hz로 사용, 그 사이 모든 IMU sample preintegrate |
| motion interval | 7-frame clip에서 gap 3/4/5 → 0.3/0.4/0.5초 구간 혼합 |
| 학습 범위 | visual DROID backbone freeze, scalar confidence head **129 parameter**만 update |
| 손실/기록 | pose, visual residual, full preintegration, confidence NLL, saturation, ATE/RPE/scale/confidence 분석 |

### 첫 learned 결과

| 방식 | Metric SE(3) ATE | mean confidence |
| --- | ---: | ---: |
| Fixed 15DoF | 0.380007 m | 1.000 |
| Old learned | 0.395872 m | 0.932 |

### 해석 → 다음 실험의 이유

> 첫 learned 모델은 fixed보다 나빴고 confidence가 0.932로 포화됐다. 즉 “학습을 했다”는 사실만으로는 confidence가 올바르게 calibration됐다고 할 수 없다. 원인을 학습–추론 조건 불일치와 confidence 초기화 문제로 보고 parity training을 설계했다.

### 발표 대본

> 여기서부터는 실제로 학습을 수행한 단계입니다. MH_03의 실제 camera–IMU 데이터를 10 Hz 조건으로 구성하고, DROID 전체가 아니라 ConvGRU 기반 scalar confidence head 129개 parameter만 학습했습니다. 학습 결과 자체도 분석했습니다. 첫 learned 모델은 ATE가 fixed보다 나빴고 confidence 평균이 0.932로 거의 포화됐습니다. 그래서 이 결과를 성공으로 포장하지 않고, 학습과 추론의 조건을 맞추는 parity training으로 다시 설계했습니다.

---

## 슬라이드 6. 실험 5: 데이터 분석 기반 parity training 재설계

### 첫 학습 결과에서 분석한 문제

| 관찰 데이터 | 해석 | 수정 |
| --- | --- | --- |
| mean \(w^u=0.932\) | 대부분의 IMU interval을 거의 동일하게 과신 | final confidence projection을 0.5에서 reset, saturation loss 추가 |
| learned가 fixed보다 높은 ATE | training과 runtime의 visual–IMU trade-off가 달랐을 가능성 | train/inference 모두 global IMU weight \(\lambda_{imu}=0.002\)로 통일 |
| confidence 효과의 원인 분리 어려움 | visual weights까지 바뀌면 무엇이 개선됐는지 모호 | visual backbone·global bias·global IMU weight freeze |

### 재학습 후 checkpoint를 고른 방법

| validation sequence | 2k checkpoint | 8k checkpoint | 선택 |
| --- | ---: | ---: | --- |
| V1_02_medium | **0.171629 m** | 0.171845 m | 2k |
| V2_02_medium | **0.178762 m** | 0.184947 m | 2k |

**선택 규칙**: V2_03 최종 결과를 보기 전에, 두 validation sequence에서 모두 더 낮은 ATE를 낸 `2k`를 고정했다.

### 대본

> 재학습에서는 output confidence가 0.5에서 출발하도록 head를 reset하고, 포화 penalty를 넣었습니다. 더 중요한 수정은 학습 때와 실행 때 전역 IMU weight를 모두 0.002로 맞춘 것입니다. 즉 confidence가 서로 다른 IMU 세기 조건에서 학습되지 않게 했습니다. 재학습 checkpoint는 V1_02와 V2_02의 validation 결과로 고정했고, V2_03은 선택에 사용하지 않았습니다.

### 이 슬라이드의 핵심

> “데이터를 넣고 train했다”가 아니라, **학습 로그와 validation metric을 분석해 실패 원인을 정하고 재학습 조건을 바꿨다**는 것을 보여 준다.

---

## 슬라이드 7. 최종 결과: 어떤 방식이 실제로 유효했는가?

### 최종 비교 조건

- EuRoC `V2_03_difficult`, 동일 reconstruction frame **239개**
- Fixed / Tuned / Old learned / Parity learned 모두 **two-stage 15DoF**, 동일 `λ_imu=0.002`
- confidence만 다르게 적용한 paired ablation

### 결과

| 방식 | confidence 방식 | ATE ↓ | RPE 1 s ↓ | RPE 5 s ↓ | scale error ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| Fixed | \(w^u=1\) | 0.380007 m | 0.172874 m | 0.502313 m | 0.207948 |
| Tuned | 수동 조절 | **0.366563 m** | **0.166692 m** | **0.484578 m** | **0.199109** |
| Old learned | 초기 학습 | 0.395872 m | 0.179980 m | 0.523240 m | 0.218543 |
| **Parity learned 2k** | 재학습 | **0.367220 m** | **0.167080 m** | **0.485570 m** | **0.199533** |

### 결과 해석

1. **Fixed → parity learned**: ATE `0.380007 → 0.367220 m`, **3.36% 개선**
2. **Old learned → parity learned**: ATE `0.395872 → 0.367220 m`, **7.24% 개선**
3. **Parity learned ↔ tuned**: ATE 차이 `0.000658 m`, **0.18% 차이**

### 발표 대본

> 최종 비교에서는 구조와 IMU prior weight를 동일하게 유지하고 confidence만 바꿨습니다. parity learned는 fixed 대비 ATE를 3.36% 낮췄고, 첫 learned 대비로는 7.24% 낮췄습니다. 가장 낮은 값은 tuned이지만 parity learned와 차이는 0.18%입니다. 따라서 결론은 learned confidence가 fixed IMU보다 유리하고, 사람이 수동으로 조절한 수준에 근접했다는 것입니다. tuned를 넘었다고 말하지는 않겠습니다.

---

## 슬라이드 8. 진행 내용 결론: 우리가 수행한 것은 무엇인가?

### 화면 문장

> 1) IMU 초기화-only → 2) 9DoF factor 결합 → 3) 15DoF two-stage → 4) learned confidence 학습 → 5) parity 재학습의 순서로, 각 실험의 한계를 다음 설계 조건으로 반영했다.

| 단계 | 실험으로 확인한 것 | 다음 설계에 반영한 것 |
| --- | --- | --- |
| initialization-only | metric scale seed는 가능하지만 pose를 고치지 못함 | IMU factor를 DBA 안으로 이동 |
| 9DoF | factor 연결·좌표계 정합 | bias를 state로 포함 |
| 15DoF | full VIO state 동작 | confidence calibration에 집중 |
| first learned | confidence 포화와 성능 악화 | parity training·reset·validation selection |
| parity learned | fixed보다 개선, tuned에 근접 | 다중 sequence/generalization이 다음 과제 |

### 마지막 대본

> 이번 진행 내용의 핵심은 단순히 여러 버전을 실행한 것이 아닙니다. 각 실험 결과가 다음 방식의 설계 조건이 되었습니다. 초기화-only의 한계가 DBA factor 결합으로, 9DoF의 bias 한계가 15DoF two-stage로, 첫 learned의 포화 문제가 parity training으로 이어졌습니다. 최종적으로는 fixed 대비 개선을 확인했지만 tuned를 넘지는 못했기 때문에, 다음에는 학습 sequence 다양화와 confidence calibration을 더 검증할 계획입니다.

---

# 발표자가 꼭 지킬 것

1. **모든 중간 수치를 하나의 성능 순위표로 섞지 않는다.** 각 단계는 구조·frame set·평가 목적이 다르다.
2. 최종 성능 주장은 슬라이드 7의 동일 239-frame paired ablation 수치만 사용한다.
3. “원본 DROID보다 좋아졌다”가 아니라, “동일한 15DoF IMU control에서 learned confidence가 fixed보다 개선됐다”고 말한다.
4. global-bias/reintegration처럼 이득이 불확실했던 실험도 숨기지 말고, 다음 설계를 결정한 **negative result**로 설명한다.

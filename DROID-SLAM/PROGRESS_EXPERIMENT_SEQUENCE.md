# Chapter 3 발표용: 실험 순서 중심 진행 내용

> 이 문서는 기존의 긴 구조 설명을 대체하는 **진행 사항용 핵심 서사**다.  
> 중심 메시지는 “IMU를 넣어 보았다”가 아니라, **초기 실패 → 원인 분리 → 구조 수정 → 학습 재설계 → 공정한 ablation으로 개선 확인**이다.

---

## 1. 이 파트의 한 문장

> DROID-SLAM에 IMU를 연결한 초기 실험은 성능을 개선하지 못했다. 그래서 IMU가 실제 DBA를 바꾸는지, 좌표계·velocity·scale 평가가 맞는지를 순서대로 검증했고, 이후 two-stage 15DoF 구조와 parity-trained confidence 학습으로 fixed IMU 대비 metric ATE 3.36% 개선을 확인했다.

이 문장을 Chapter 3의 시작과 끝에 변형해서 반복하면 발표의 중심이 흔들리지 않는다.

---

## 2. 먼저 보여 줄 전체 실험 흐름 — 슬라이드 1

### 화면

삽입 그림: [`presentation_assets/experiment_sequence.svg`](presentation_assets/experiment_sequence.svg)

또는 아래 6단계를 가로 화살표로 작성한다.

```text
① Baseline/IMU 연결 확인
          ↓
② 좌표계·state propagation·metric 평가 진단
          ↓
③ 9DoF IMU DBA: factor가 optimization에 반영되는지 검증
          ↓
④ 15DoF two-stage: scale·velocity·bias를 함께 다루도록 구조 확장
          ↓
⑤ Stage-3 confidence 학습: 실패 원인(confidence 포화) 분석
          ↓
⑥ Parity 재학습 + fixed/tuned/learned paired ablation
```

### 대본

> 진행은 한 번의 학습으로 끝나지 않았습니다. 먼저 IMU 파일을 읽는 것과 IMU factor가 실제 DBA 최적화를 바꾸는 것을 구분했습니다. 초기 성능 개선이 없자 좌표계, velocity state propagation, metric 평가 단위를 수정했습니다. 이후 9DoF에서 IMU factor 동작을 검증하고, bias까지 다루기 위해 15DoF two-stage 구조로 확장했습니다. 마지막에는 learned confidence가 포화되는 실패를 확인해 학습과 추론 조건을 일치시킨 parity training으로 다시 학습하고 fixed·tuned·learned ablation을 수행했습니다.

### 이 슬라이드의 역할

- 이후 슬라이드가 무엇을 해결하기 위한 실험인지 미리 알려 준다.
- 단순 기능 나열이 아니라 **가설 검증 순서**였음을 보여 준다.

---

## 3. 실험 1 — “IMU를 읽는 것”과 “IMU가 DBA를 바꾸는 것”의 구분

### 문제 / 가설

> IMU stream을 전달하기만 해도 성능이 바뀌는가? 아니면 residual이 실제 solver의 Hessian/gradient에 들어가야 하는가?

### 구현

- EuRoC의 gyro/accelerometer를 frame 사이에서 preintegrate
- \(\Delta R, \Delta v, \Delta p\)를 visual DBA의 IMU factor로 전달
- visual-only, IMU data only, full IMU fixed, initial learned를 비교

### 결과 — 짧은 prefix에서의 연결 검증

**EuRoC V1_03, 동일 43 reconstruction keyframes, Sim(3) ATE**

| 조건 | ATE RMSE | IMU factor edge | 해석 |
| --- | ---: | ---: | --- |
| Original DROID | 0.069182 m | 0 | visual-only 기준 |
| IMU data only | 0.069182 m | 0 | IMU를 읽기만 하면 결과 변화 없음 |
| Full IMU fixed | 0.069191 m | 247 | factor는 사용됐지만 개선 없음 |
| Initial learned | 0.069200 m | 247 | 초기 learned도 개선 없음 |

### 여기서 내린 판단

> “IMU가 들어왔다”는 것과 “IMU가 유효하게 결합됐다”는 것은 다르다. 다음 단계는 가중치를 임의로 키우는 것이 아니라, IMU와 visual state의 **정합성**을 검사하는 것이었다.

### 대본

> 첫 실험에서는 IMU 파일을 읽는 것만으로는 DROID 결과가 바뀌지 않는다는 control을 확인했습니다. 반면 full IMU factor는 247개 temporal edge에서 실제로 solver에 적용됐지만 성능은 좋아지지 않았습니다. 이 결과는 실패라기보다, 단순 고정 weight 결합만으로는 충분하지 않으며 input convention과 state update를 점검해야 한다는 근거가 됐습니다.

### 주의

- 이 표는 **최종 성능 표가 아니다.** 구현 연결과 control의 필요성을 보이는 과정 자료다.
- prefix/keyframe 수가 작으므로 이 결과로 “성능 저하”를 강하게 주장하지 않는다.

---

## 4. 실험 2 — 원인 분석: 좌표계·velocity·metric 평가를 고쳤다

### 문제 분석

초기 IMU factor가 도움이 되지 않은 원인을 세 항목으로 분리했다.

| 진단 항목 | 발견한 문제 | 수정 | 검증 결과 |
| --- | --- | --- | --- |
| Coordinate convention | visual pose axis와 gyro/acc axis가 달랐음 | gyro/acc에도 pose와 동일한 axis transform 적용 | relative rotation mismatch **6.98° → 0.385°** |
| State propagation | 학습 unroll마다 DBA가 갱신한 velocity를 pose difference 값으로 덮어씀 | pose difference는 초기 prior로만 사용, 갱신 velocity 유지 | training/runtime state evolution 정합 |
| Metric evaluation | VI alignment 후 내부 DROID unit을 m로 환산하지 않음 | `imu_unit_scale`을 반영해 metric ATE/RPE 계산 | metric 결과의 단위 해석 정정 |

### 9DoF 재검증

- 상태: pose 6 + velocity 3, bias는 VI initialization 값으로 고정
- 목적: bias를 복잡하게 최적화하기 전에 full preintegration factor가 안정적으로 동작하는지 확인

| Sequence | Original Sim(3) ATE | 9DoF IMU Sim(3) ATE | 변화 | IMU factor 사용률 |
| --- | ---: | ---: | ---: | ---: |
| V2_01 easy | 0.030052 m | 0.030024 m | 0.093% 감소 | 1.000 |
| V2_03 difficult | 0.047936 m | 0.047857 m | 0.165% 감소 | 0.998 |

### 여기서 내린 판단

> IMU factor가 거의 모든 temporal edge에서 실제로 동작하는 것은 확인했다. 그러나 0.1% 수준의 Sim(3) 차이는 일반적인 성능 향상이라고 주장하기에 작다. 따라서 다음 문제는 bias와 initialization이라고 판단했다.

### 대본

> 초기 실험을 분석해 보니 IMU 값과 visual pose가 같은 좌표계에 있지 않았고, training에서는 업데이트된 velocity가 다음 반복에 사라지고 있었습니다. 이 문제를 수정하자 relative rotation mismatch가 6.98도에서 0.385도로 줄었습니다. 수정한 9DoF 모델은 IMU factor가 거의 모든 구간에서 적용되는 것을 확인했고 두 sequence에서 작은 positive signal을 보였지만, 0.1% 수준이므로 이를 최종 개선이라고 주장하지 않았습니다. 이 단계는 다음 구조 확장의 근거를 만든 단계입니다.

---

## 5. 실험 3 — 15DoF two-stage 구조로 확장

### 이 슬라이드에 넣을 구조도

[`presentation_assets/droid_baseline_vs_vi_extension_ko.svg`](presentation_assets/droid_baseline_vs_vi_extension_ko.svg)를 삽입한다.

- **파란색**: 원본 DROID-SLAM에서 유지한 image encoder, correlation, ConvGRU, visual update, dense visual BA
- **주황색**: 새로 추가한 gyro/accelerometer preintegration
- **초록색**: visual warm-up → VI alignment → full 15DoF DBA
- **보라색**: ConvGRU feature에서 IMU factor confidence를 예측하는 DVI-style scalar head

그림 아래에는 다음 한 줄만 둔다.

> DROID의 visual backbone은 유지하고, IMU는 raw feature가 아니라 preintegrated factor로 DBA에 결합했으며, learned confidence는 그 IMU factor의 상대 가중치만 조절한다.

### 왜 9DoF에서 멈추지 않았는가?

9DoF는 velocity를 최적화하지만 accelerometer/gyroscope bias를 고정한다. IMU bias는 시간이 지나며 position·velocity residual에 누적 오차를 만들기 때문에, metric VIO에서는 pose와 velocity만으로 충분하지 않다.

### 설계

```text
Stage 1: visual DROID pose/depth warm-up
             ↓
VI alignment: scale, gravity, v, bₐ, b_g 초기화
             ↓
Stage 2: full 15DoF DBA
         pose(6) + velocity(3) + accel bias(3) + gyro bias(3)
         + existing dense inverse depth
```

### 검증한 내용

- IMU factor를 사후 trajectory correction이 아니라 DBA normal equation에 결합
- full preintegration residual: position, velocity, rotation, bias random-walk
- stage 1에는 IMU factor를 끄고 stage 2에서만 full 15DoF factor 활성화
- IMU initialization 성공 후, metric scale·gravity·velocity·bias state를 solver에 전달

### 여기서 얻은 결과와 한계

- 15DoF two-stage에서 metric trajectory 생성과 IMU factor 사용은 확인했다.
- 초기 15DoF/initial confidence 조합만으로 항상 ATE가 좋아지지는 않았다.
- 따라서 “더 많은 state” 자체가 답이 아니라, IMU factor의 **상대 신뢰도 calibration**이 필요하다고 판단했다.

### 대본

> 9DoF에서는 bias를 고정했기 때문에 장기적으로 IMU bias가 쌓이는 문제를 직접 다루지 못했습니다. 그래서 visual warm-up 뒤 scale, gravity, velocity, bias를 초기화하고, 두 번째 단계에서 15DoF state를 joint optimization하는 구조로 확장했습니다. 이 단계에서 metric trajectory와 full preintegration factor가 동작하는 것은 확인했지만, 단순히 state를 늘린다고 성능이 자동으로 좋아지지는 않았습니다. 그래서 이후에는 IMU를 어느 구간에서 얼마나 믿을지 학습하는 confidence 문제로 초점을 옮겼습니다.

---

## 6. 실험 4 — Stage-3 confidence 학습: 무엇을 데이터로 학습했는가? ★

### 학습 가설

> IMU noise model만으로는 모든 구간의 신뢰도를 설명하기 어렵다. ConvGRU의 visual tracking 상태를 이용해 IMU factor의 상대 가중치 \(w^u\)를 학습하면 fixed \(w^u=1\)보다 좋아질 수 있다.

### 학습 데이터와 생성 방식

| 항목 | 실제 수행 조건 |
| --- | --- |
| training sequence | EuRoC `MH_03_medium` |
| camera input | left monocular camera, 20 Hz 원본에서 **10 Hz**로 decimation |
| IMU | camera frame 사이의 **모든** high-rate sample을 preintegrate |
| clip | 7 frames, gap 3/4/5를 무작위 선택 → 0.3/0.4/0.5 s interval |
| optimizer run | 8,000 training steps, learning rate `1e-4` |
| checkpoint candidates | 2k / 4k / 6k / 8k |

### 무엇을 freeze / update했는가?

| 고정한 항목 | 이유 |
| --- | --- |
| DROID encoder, correlation, ConvGRU, visual update heads | visual 성능 변화와 confidence 효과를 분리 |
| global IMU bias | confidence 학습과 bias 변화의 원인을 분리 |
| global IMU prior weight \(\lambda_{imu}=0.002\) | train/inference에서 동일한 visual–IMU trade-off 유지 |

| 학습한 항목 | 구조 |
| --- | --- |
| DVI-style scalar confidence head, **129 parameter** | ConvGRU feature → 1×1 Conv → spatial mean → sigmoid → \(w^u\in[0,1]\) |

### 학습 중 분석한 값

- pose/geodesic loss와 visual reprojection residual
- full IMU preintegration loss: position·velocity·rotation·bias consistency
- confidence NLL: residual 크기와 confidence를 연결
- saturation loss: confidence가 모든 구간에서 1에 붙는 것을 억제
- runtime/validation diagnostics: ATE, RPE, scale error, mean confidence, IMU factor 사용률

### 대본

> 여기서 학습한 것은 DROID 전체가 아닙니다. 실제 IMU가 있는 MH_03의 10 Hz monocular clip과 그 사이의 모든 IMU sample을 사용해, ConvGRU feature에서 IMU confidence 하나를 예측하는 129개 parameter만 학습했습니다. visual backbone과 global IMU weight는 고정해, 결과 차이가 visual network 변화가 아니라 confidence calibration에서 왔는지 확인할 수 있게 했습니다. 학습 중에는 pose와 reprojection loss뿐 아니라 position, velocity, rotation, bias를 포함한 preintegration residual과 confidence 분포를 함께 기록했습니다.

### 핵심 설명

- \(w^u\)는 IMU 값을 대체하거나 bias를 직접 예측하지 않는다.
- runtime IMU 영향은 개념적으로 `0.002 × w^u × covariance-derived information`으로 결정된다.
- `0.25 → 0.50 → 0.75 → 1.00`은 deployment 중 confidence를 순서대로 넣는 값이 아니라, 4회의 full-IMU training unroll에 넣은 **약한 auxiliary curriculum**이다.

---

## 7. 실험 5 — 학습 실패를 분석하고 parity training으로 재설계

### 초기 learned 결과에서 발견한 문제

| 비교 | Metric SE(3) ATE, V2_03 common frames | mean \(w^u\) | 해석 |
| --- | ---: | ---: | --- |
| Fixed 15DoF | 0.380007 m | 1.000 | 고정 IMU control |
| Old learned | 0.395872 m | 0.932 | confidence가 거의 포화, fixed보다 악화 |

### 원인 가설과 수정

| 관찰 | 수정한 설계 |
| --- | --- |
| confidence가 대부분 1에 가까움 | confidence projection을 sigmoid 0.5에서 다시 시작, saturation loss 적용 |
| 학습/추론에서 IMU의 상대 세기가 다르면 confidence가 잘못 학습될 수 있음 | train/inference 모두 \(\lambda_{imu}=0.002\)로 고정 |
| visual network 변화가 confidence 효과를 가릴 수 있음 | DROID visual backbone, global bias, global IMU weight freeze |

### checkpoint selection도 데이터로 수행

| validation sequence | 2k ATE | 8k ATE | 선택 |
| --- | ---: | ---: | --- |
| V1_02_medium | 0.171629 m | 0.171845 m | 2k |
| V2_02_medium | 0.178762 m | 0.184947 m | 2k |

> 두 validation sequence에서 모두 낮은 ATE를 보인 **2k checkpoint**를 선택했다. V2_03의 결과를 보고 checkpoint를 고르지 않았다.

### 대본

> 첫 learned 모델은 오히려 fixed보다 나빴습니다. mean confidence가 0.932로 거의 포화되어 IMU를 구간별로 구분하지 못한 것이 핵심 신호였습니다. 그래서 단순히 더 오래 학습하지 않고, 학습과 추론의 IMU weight를 0.002로 일치시킨 parity training을 구성했습니다. confidence head는 0.5에서 다시 시작했고, visual backbone은 고정했습니다. 이후 V1_02와 V2_02에서 checkpoint 2k와 8k를 비교해 둘 다에서 낮은 ATE를 보인 2k를 선택했습니다.

---

## 8. 최종 paired ablation — 무엇이 실제로 개선됐는가? ★

### 공정한 비교 조건

- EuRoC `V2_03_difficult`
- 동일 reconstruction frame ID **239개**
- Fixed / Tuned / Old learned / Parity learned는 동일한 **two-stage 15DoF** 구조
- 동일한 global IMU prior weight `0.002`
- 차이는 IMU confidence 적용 방식뿐

### 결과

| 방법 | Metric SE(3) ATE ↓ | RPE 1 s ↓ | RPE 5 s ↓ | scale error ↓ | mean \(w^u\) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 15DoF | 0.380007 m | 0.172874 m | 0.502313 m | 0.207948 | 1.000 |
| Tuned 15DoF | **0.366563 m** | **0.166692 m** | **0.484578 m** | **0.199109** | 0.516 |
| Old learned | 0.395872 m | 0.179980 m | 0.523240 m | 0.218543 | 0.932 |
| **Parity learned 2k** | **0.367220 m** | **0.167080 m** | **0.485570 m** | **0.199533** | 0.855 |

### 성과를 유형별로 해석

| 개선 유형 | 비교 | 확인된 변화 | 주장 가능한 결론 |
| --- | --- | --- | --- |
| 구현 정합성 | coordinate fix 전/후 | rotation mismatch 6.98° → 0.385° | visual–IMU convention 오류 제거 |
| 학습 calibration | old learned → parity learned | ATE 0.395872 → 0.367220 m, **7.24% 감소** | confidence 포화/weight mismatch 수정이 유효 |
| 알고리즘 성능 | fixed → parity learned | ATE **3.36% 감소**, RPE-1 **3.35% 감소**, RPE-5 **3.33% 감소** | learned confidence가 고정 confidence보다 유리 |
| 수동 tuning 대체 가능성 | tuned → parity learned | ATE 차이 0.000658 m, **0.18%** | learned가 manual tuned 수준에 근접 |

### 대본

> 최종 비교는 동일한 239개 frame에서 구조와 전역 IMU weight를 모두 맞추고 confidence 방식만 바꿨습니다. parity learned는 fixed보다 ATE를 3.36% 줄였고, 이전 learned보다 7.24% 줄였습니다. 수동 tuned가 가장 낮은 수치이지만 learned와 차이는 0.18%입니다. 따라서 이번 결과는 learned confidence가 fixed IMU보다 유리하고, manual tuning에 근접했다는 근거입니다. 반면 manual tuned를 넘었다고 말할 수는 없으므로, 그 부분은 다음 개선 과제로 남깁니다.

---

## 9. Chapter 3 마지막 결론 — 결과를 과장하지 않는 문장

### 화면에 넣을 문장

> IMU를 단순 입력으로 추가하는 수준을 넘어, preintegration residual·two-stage initialization·15DoF DBA·learned confidence를 순차적으로 구현하고 검증했다. 최종 parity learned 모델은 fixed IMU 대비 metric ATE를 3.36% 개선했지만, manual tuned보다 0.18% 높다. 따라서 현재 결과는 **완성된 최적 모델 선언**이 아니라, 학습–추론 정합과 confidence calibration이 성능에 실제로 영향을 준다는 검증된 개선 단계**이다.**

### 꼭 말할 한계

- Original monocular DROID는 metric scale이 없으므로 VIO의 metric SE(3) ATE와 직접 우열 비교할 수 없다.
- Visual shape Sim(3) ATE는 original이 더 낮다. 따라서 “original DROID visual performance를 넘어섰다”는 주장은 하지 않는다.
- training은 `MH_03_medium` 단일 sequence라 generalization 범위가 좁다.
- V2_03은 raw training/validation sequence와 겹치지는 않지만 앞선 exploratory ablation을 보았으므로, 엄밀한 blind hold-out 대신 final re-evaluation이라고 부른다.

### 마무리 대본

> 저희가 확인한 것은 모든 조건에서 original을 이긴 결과가 아닙니다. 대신 DROID의 visual backbone을 유지한 채 IMU를 tightly coupled DBA에 넣고, 초기 실패를 좌표계·state propagation·confidence calibration 문제로 분해해 수정했습니다. 그 결과 fixed IMU보다 learned confidence가 개선되는 paired ablation을 만들었습니다. 다음 단계는 다중 sequence와 모바일 IMU로 학습 범위를 넓혀 tuned를 넘어서는지 검증하는 것입니다.

---

## 10. 실제 슬라이드 배치 권장

진행 내용(Chapter 3)을 아래 **7장**으로 구성하면 “수행 과정”이 선명해진다.

| 순서 | 슬라이드 제목 | 보여 줄 것 | 한 줄 메시지 |
| ---: | --- | --- | --- |
| 1 | 실험 전체 흐름 | 6단계 timeline | 실패를 다음 설계로 연결함 |
| 2 | 초기 IMU 결합 실험 | data-only/fixed/initial learned control | 입력 연결 ≠ 성능 개선 |
| 3 | 원인 분석과 9DoF 재검증 | 6.98°→0.385°, 9DoF table | 오류를 수치로 검증하고 수정 |
| 4 | 15DoF two-stage 설계 | 구조 그림 | bias/initialization 문제를 다루도록 확장 |
| 5 | Stage-3 training | MH03 clips, freeze/update, loss/diagnostics | 데이터를 이용해 confidence만 학습·분석 |
| 6 | parity training과 checkpoint selection | old failure, V1/V2 selection table | 학습 조건 불일치를 바로잡음 |
| 7 | final paired ablation | final 4-way table + limitation | fixed 대비 3.36% 개선, tuned 근접 |

모델 전체 구조도는 4번에만 넣고, detailed loss 식은 appendix로 보낸다. 이 순서에서는 청중이 매 슬라이드마다 “그래서 다음 실험을 왜 했는데?”를 자연스럽게 이해할 수 있다.

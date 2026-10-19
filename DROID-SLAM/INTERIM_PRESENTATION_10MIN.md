# 중간발표 구성안 — DROID-SLAM 기반 단안 Visual-Inertial SLAM

발표 시간: 10분 발표 + 5분 Q&A  
발표 핵심 문장: **단안 DROID-SLAM의 scale·급격한 움직임 한계를 휴대폰 IMU로 보완하기 위해, IMU를 사후 보정이 아니라 최적화 내부 factor로 결합하고 있다. 초기 실패를 원인 분석과 공정한 검증으로 전환해 현재 15DoF two-stage 구조까지 확장했다.**

## 슬라이드 1. 제목 — 0:10

**DROID-SLAM 기반 단안 Visual-Inertial SLAM 개발**  
카메라와 IMU를 결합한 실시간 공간 인식 프로토타입

- 팀명 / 팀원 / 과목명
- 발표자

**발표 멘트**

> 저희는 스마트폰처럼 카메라와 IMU를 함께 사용할 수 있는 환경에서, 단안 SLAM의 한계를 보완하는 Visual-Inertial SLAM을 개발하고 있습니다. 오늘은 구현 결과를 나열하기보다, 초기 실험의 한계를 어떻게 다음 설계로 연결했는지 중심으로 말씀드리겠습니다.

---

## [1] 프로젝트 개요

## 슬라이드 2. 문제 정의 — 0:45

**문제: 단안 영상만 사용하는 SLAM은 절대 scale과 급격한 움직임 구간에 취약하다.**

| 환경 | 문제 | 결과 |
| --- | --- | --- |
| 단안 카메라 | 실제 거리(scale)를 직접 관측하기 어려움 | metric trajectory 생성이 어려움 |
| 빠른 회전·짧은 baseline·저텍스처 | 영상 대응점이 불안정해짐 | 자세/위치 drift 및 tracking 불안정 |
| 스마트폰 | 카메라 외에 gyro·accelerometer가 존재 | 기존 visual-only SLAM은 이 정보를 충분히 활용하지 못함 |

**해결 질문**: 이미지가 약한 구간에서 IMU의 물리적 motion prior를 어떻게 안전하게 활용할 것인가?

**발표 멘트**

> 단안 SLAM은 영상만으로 위치를 추정하므로, 실제 길이 단위의 scale을 바로 알기 어렵고 영상 특징이 부족한 구간에서 오차가 커질 수 있습니다. 반면 스마트폰에는 회전과 가속도 정보를 주는 IMU가 이미 있습니다. 저희의 문제는 이 정보를 단순히 덧붙이는 것이 아니라 SLAM 최적화에 올바르게 결합하는 것입니다.

## 슬라이드 3. 목표와 접근 — 0:45

### 최종 목표

휴대폰 RGB + IMU 입력으로부터 **metric scale을 가진 카메라 궤적과 dense reconstruction**을 생성하는 연구 프로토타입을 구현한다.

### 기술 목표

1. IMU 사전적분(preintegration) 제약을 DROID-SLAM의 dense bundle adjustment(DBA) 내부에 결합
2. pose·velocity·dense depth를 영상 reprojection residual과 함께 최적화
3. 신뢰도가 낮은 IMU 구간은 learned confidence와 covariance로 약하게 반영
4. 학습/validation/held-out test를 분리해 성능을 검증

```text
Camera frames ──> DROID visual factors ─┐
                                        ├─> Visual-Inertial DBA ─> trajectory / reconstruction
IMU stream ──> synchronization ─> preintegration ─┘
```

**한 줄 차별점**: gyro로 pose를 사후 보정하는 방식이 아니라, IMU residual의 Jacobian/Hessian을 DBA에 추가하는 tightly-coupled 방식이다.

---

## [2] 프로젝트 계획

## 슬라이드 4. 1학기 진행 내용 리마인드 — 0:35

> **아래 항목은 실제 1학기 산출물과 맞게 최종 확인 후 사용한다.**

| 1학기 기반 작업 | 다음 단계에 남긴 기반 |
| --- | --- |
| DROID-SLAM 구조·실행 환경 분석 및 visual-only baseline 재현 | IMU 결합 전 기준 성능 확보 |
| 단안 SLAM과 Visual-Inertial SLAM 문헌 조사 | IMU preintegration과 DBA 결합 방향 선정 |
| 웹 기반 카메라·IMU 수집 화면 및 서버 세션 구조 구현 | RGB frame, IMU timestamp, calibration을 같은 session으로 저장 |
| 데이터셋/평가 흐름 탐색 | TartanAir 학습, EuRoC 검증이라는 분리된 실험 기반 마련 |

**발표 멘트**

> 1학기에는 바로 모델을 바꾸기보다, visual-only baseline을 재현하고 입력·평가 기반을 먼저 만들었습니다. 이 기반이 있었기 때문에 2학기에는 IMU를 넣었을 때 모델 구조의 문제인지, 입력 동기화의 문제인지 분리해 분석할 수 있었습니다.

## 슬라이드 5. 전체 계획과 조정 과정 — 0:45

| 기간 | 원래/수행 목표 | 상태 | 조정 근거 |
| --- | --- | --- | --- |
| 1학기 | baseline, 웹 수집, 연구 방향 설정 | 완료 | 2학기 모델 개선의 기준선 확보 |
| 7–8월 | IMU 입력·사전적분·초기 DBA 결합 | 완료 | IMU factor가 실제 solver에 들어가는지 확인 |
| 9월 | 9DoF 학습 및 EuRoC 검증 | 완료 | 초기 성능 개선 부재 → 좌표계·velocity·평가 단위 문제를 먼저 수정 |
| 10월 현재 | 15DoF two-stage, bias 및 confidence 확장 | 진행 중 | bias와 불확실도가 남은 주요 오차원인 |
| 최종 단계 | paired held-out ablation, full trajectory 평가, 휴대폰 입력 검증 | 예정 | 재현 가능한 실용성 검증 |

**일정 관리 포인트**

- “성능이 안 좋으니 가중치를 키운다”가 아니라, 원인 검증 → 수정 → 분리된 재평가 순으로 계획을 조정했다.
- validation(V2_01)에서 checkpoint를 고정하고 held-out(V2_03)에서 확인해 test-set tuning을 피했다.

---

## [3] 프로젝트 진행 내용

## 슬라이드 6. 7월~현재 개발 흐름 — 0:40

```text
Baseline / input pipeline
          ↓
Initial IMU factor ablation
          ↓  (개선 없음)
Coordinate·velocity·covariance diagnosis
          ↓
9DoF visual-inertial DBA + learned confidence training
          ↓
Validation-based checkpoint selection
          ↓
15DoF two-stage bias optimization + held-out ablation (current)
```

**발표 멘트**

> 진행 과정에서 가장 중요한 의사결정은 초기 수치가 좋지 않았을 때 구조를 무작정 복잡하게 하지 않은 것입니다. 먼저 IMU가 실제 optimizer를 바꾸는지와 좌표계·상태 전파가 맞는지를 검증하고, 그 뒤에 학습과 bias 모델을 확장했습니다.

## 슬라이드 7. 현재 구현 구조 — 0:50

| 단계 | 구현 내용 | 검증 의미 |
| --- | --- | --- |
| 입력 | frame–IMU timestamp 동기화, IMU preintegration | 카메라 간 구간의 ΔR, Δv, Δp 생성 |
| 초기화 | scale, gravity, initial velocity, bias 추정 | 단안 결과를 metric trajectory로 변환할 기반 |
| Stage 1 | visual pose/depth DBA | 불안정한 초기 영상 geometry를 먼저 안정화 |
| Stage 2 | pose + velocity + accel/gyro bias + depth 15DoF DBA | IMU residual과 bias를 jointly refine |
| confidence | rotation·velocity·position별 IMU confidence | 신뢰도 낮은 IMU를 과신하지 않도록 가중 |

**설명 그림 권장**: `영상 → visual factor`, `IMU → preintegration factor` 두 화살표가 하나의 DBA block으로 합쳐지는 그림. 수식은 본문에 넣지 않는다.

---

## 슬라이드 8. 대표 실험 1 — “IMU를 읽는 것”과 “IMU가 최적화를 바꾸는 것”의 구분 — 0:45

**조건**: EuRoC V1_03, 동일 43 reconstruction keyframe, Sim(3) ATE

| 모델 | ATE RMSE | IMU factor 사용 | 해석 |
| --- | ---: | ---: | --- |
| Original DROID | 0.069182 m | 0 | visual-only baseline |
| IMU data only | 0.069182 m | 0 | 파일을 읽기만 하면 결과는 바뀌지 않음 |
| Full IMU fixed | 0.069191 m | 247 edges | factor는 실제 적용됐지만 개선 없음 |

**도출된 결론**: 단순 고정 가중치 결합은 부족하다. IMU를 “썼다”는 것과 “유효하게 썼다”는 것은 다르다.

**발표 멘트**

> 이 실험은 실패한 성능 실험이 아니라 구현 검증 실험입니다. IMU factor가 실제로 사용되는 것은 확인했지만 성능이 좋아지지 않았습니다. 따라서 이후에는 factor의 양을 늘리는 대신, 좌표계·상태 전파·불확실도 모델을 점검하는 방향으로 전환했습니다.

## 슬라이드 9. 대표 실험 2 — 원인 분석 후 9DoF 재검증 — 0:55

### 수정한 핵심 원인

- visual pose와 raw IMU의 **좌표축 불일치** 수정
  - relative rotation 불일치: **6.98° → 0.385°**
- 학습 unroll에서 DBA가 갱신한 velocity를 다음 iteration에 덮어쓰던 문제 수정
- preintegration covariance/information과 metric scale 평가 단위 정리

### 수정 후 결과: 9DoF IMU factor가 거의 모든 temporal edge에서 사용됨

| Sequence | Original DROID Sim(3) ATE | 9DoF IMU Sim(3) ATE | 변화 |
| --- | ---: | ---: | ---: |
| EuRoC V2_01 easy | 0.030052 m | 0.030024 m | +0.093% |
| EuRoC V2_03 difficult | 0.047936 m | 0.047857 m | +0.165% |

**도출된 결론**: 작은 positive signal은 확인했지만, 0.1% 수준이므로 일반적 성능 개선이라고 주장하지 않는다.

## 슬라이드 10. 대표 실험 3 — 학습 모델 선택과 공정한 검증 — 0:45

**학습**: TartanAir V2 Hospital 22 trajectories, 약 32,340 frames  
**선택**: EuRoC V2_01 validation에서 2k/4k/6k/8k 후보를 비교 → 6k checkpoint 선택  
**평가**: 선택 후 V2_03 difficult held-out sequence에서 확인

| 원칙 | 적용 방법 |
| --- | --- |
| 학습/검증/test 분리 | TartanAir 학습 → V2_01 validation → V2_03 held-out |
| 임의 tuning 방지 | validation에서 checkpoint를 한 번 고정 |
| metric의 공정한 해석 | visual-only와 IMU 모델은 Sim(3) shape metric으로 비교; IMU metric scale은 별도 보고 |

**핵심 메시지**: “가장 좋은 수치 하나”보다, 다른 sequence에서도 재현되는지 검증하는 실험 설계를 우선했다.

## 슬라이드 11. 대표 실험 4 — 15DoF two-stage와 현재 판단 — 0:55

### 왜 15DoF로 확장했는가?

9DoF는 bias를 초기값으로 고정한다. 긴 구간에서는 gyro/accelerometer bias가 누적돼 잘못된 IMU prior가 될 수 있다.

```text
Stage 1: visual pose/depth를 먼저 안정화
          ↓ VI initialization
Stage 2: pose + velocity + accel bias + gyro bias + depth 공동 최적화
```

### held-out V2_03 결과 (99 common reconstruction pose)

| 조건 | Sim(3) ATE | Metric SE(3) ATE | IMU 사용률 |
| --- | ---: | ---: | ---: |
| Visual bootstrap control | 0.019180 m | - | - |
| Two-stage 15DoF pass-0 | 0.017254 m | 0.222643 m | 79.5% |
| Post/global-bias step | 0.018820 m | 0.226775 m | 79.8% |

**현재 판단**

- 15DoF tightly-coupled factor, metric trajectory, bias refinement 경로는 동작한다.
- 그러나 global bias/learned confidence의 **독립적 정확도 향상은 아직 증명되지 않았다.**
- 다음 우선순위는 full covariance whitening과 fixed/scalar/component confidence의 paired ablation이다.

> 이 슬라이드의 정직한 결론이 기술적 문제 해결 능력을 보여준다. 다른 keyframe 경로인 original DROID와 직접 우열 비교하지 않고 matched control을 기준으로 해석한다.

## 슬라이드 12. 문제 해결 역량 요약 — 0:30

| 예상 밖 문제 | 대응 | 결과 |
| --- | --- | --- |
| 초기 IMU 결합이 성능을 개선하지 못함 | factor 사용 여부와 입력/optimizer 경로를 분리해 점검 | “연결됨”과 “유효함”을 구분 |
| pose–IMU 좌표 불일치 | gyro/acc에도 pose와 동일한 axis transform 적용 | 회전 불일치 6.98° → 0.385° |
| velocity state 소실 | training unroll의 state propagation 수정 | 학습과 runtime motion model 정합 |
| bias·불확실도 문제 | 15DoF two-stage, reintegration, confidence 확장 | 다음 공정 ablation의 기반 확보 |

---

## [4] Role and Responsibilities

## 슬라이드 13. 역할 분담과 협업 — 0:40

> **팀원 이름과 실제 담당 작업으로 대체한다. 과장 없이 commit, 실험 log, 화면 구현처럼 확인 가능한 산출물 중심으로 작성한다.**

| 팀원 | 주 역할 | 구체 산출물 | 협업 접점 |
| --- | --- | --- | --- |
| [이름 A] | SLAM/최적화 | 9DoF·15DoF IMU factor, DBA, VI initialization | 실험 조건과 runtime interface 공유 |
| [이름 B] | 데이터/학습/평가 | TartanAir·EuRoC loader, training, held-out evaluator | 모델 선택 기준과 결과표 검토 |
| [이름 C] | 웹·입력 파이프라인 | 카메라/IMU WebSocket 수집, timestamp sync, session 저장 | 실제 입력을 preintegration 경로에 전달 |
| [이름 D 또는 공동] | PM·문서·통합 | 일정, 실험 기록, 발표·데모, QA | weekly review와 재현 실행 점검 |

**협업 방식 한 줄**: 매 실험은 `입력 조건–checkpoint–실행 옵션–결과 요약`을 함께 기록해, 구현 담당자와 평가 담당자가 독립적으로 결과를 확인한다.

---

## [5] 진행 결과물 및 시연

## 슬라이드 14. 현재 결과물 — 0:45

**실행 가능한 연구 프로토타입**

```text
웹 브라우저
  └ 카메라 frame (현재 640×480, 목표 10 FPS) + IMU stream 수집
        ↓ WebSocket
세션 저장소
  └ images / frames.csv / imu.csv / synced.json
        ↓ timestamp synchronization + preintegration
DROID+IMU runtime
  └ trajectory, dense reconstruction, IMU usage/confidence/debug metrics
        ↓
EuRoC/TartanAir evaluator
  └ Sim(3) ATE/RPE, metric scale·ATE/RPE, factor usage
```

### 시연 순서 (40초 이내)

1. 웹 화면에서 카메라와 IMU streaming 시작
2. frame/IMU count와 생성된 session 파일 확인
3. `synced.json` 및 `imu_prior` 생성 확인
4. 미리 준비한 SLAM 결과의 trajectory/reconstruction 또는 evaluation summary 제시

**시연 안전장치**: live inference가 오래 걸리거나 GPU 상태가 불확실하면 수집 화면은 live로, SLAM 결과는 미리 실행한 결과 화면/짧은 영상으로 보여준다.

## 슬라이드 15. 결론 및 최종 계획 — 0:25

### 현재까지

- 카메라–IMU 수집부터 preintegration, 9DoF/15DoF tightly-coupled DBA, 학습·평가까지 구현
- 초기 성능 부진을 통해 calibration, state propagation, uncertainty가 핵심임을 확인하고 수정
- IMU factor의 실제 최적화 반영과 metric trajectory 생성을 검증

### 최종 단계

1. 동일 조건의 visual-only / fixed / scalar-confidence / component-confidence paired ablation
2. full trajectory 기반 held-out ATE/RPE와 scale 평가
3. 휴대폰 입력의 time offset, extrinsic, noise·bias를 검증

**마무리 멘트**

> 현재 단계의 성과는 IMU를 넣었다는 선언이 아니라, 실제 최적화에 결합된 visual-inertial pipeline을 만들고, 성능이 불확실한 결과도 공정한 control 실험으로 다음 개선 과제로 연결했다는 점입니다.

---

## Q&A 대비 핵심 답변

### Q1. 현재 original DROID보다 확실히 좋은가?

아직 일반적 우위를 주장하지 않는다. 좌표계와 state propagation 수정 후 작은 positive signal은 있었지만, held-out paired ablation에서 일관된 개선을 확인하는 단계다. 그래서 다음 결과는 full trajectory 기반의 동일 조건 비교로 고정했다.

### Q2. 왜 IMU를 넣으면 무조건 좋아지지 않는가?

IMU에는 bias, noise, camera–IMU 시간/좌표 calibration 오차가 있으며, 잘못 가중하면 영상 기반 추정을 오히려 끌어갈 수 있다. 그래서 covariance와 learned confidence, bias 추정이 필요하다.

### Q3. 15DoF가 무엇인가?

각 keyframe의 motion state에서 pose 6DoF, velocity 3DoF, accelerometer bias 3DoF, gyro bias 3DoF를 함께 다루는 것이다. Dense depth는 기존 DROID 방식대로 별도로 최적화한다.

### Q4. 최종 사용자는 무엇을 얻는가?

카메라·IMU가 있는 휴대폰에서 촬영한 데이터로부터, 단안 영상만으로는 얻기 어려운 metric scale 정보를 포함한 카메라 궤적과 3D reconstruction을 얻는 것을 목표로 한다.

### Q5. 왜 웹으로 만들었는가?

별도 앱 설치 없이 브라우저에서 camera·motion sensor 권한을 받아 빠르게 입력을 수집하고, timestamp가 있는 frame/IMU session을 연구 runtime으로 연결하기 위해서다. 최종 상용 앱을 주장하는 것이 아니라 현 단계의 입력·검증 도구다.

## 슬라이드 제작 원칙

- 본문 수식은 제외하고, 입력→최적화→결과 그림을 사용한다.
- 결과 표는 슬라이드 8, 9, 11만 사용하고 각 표는 세 행 이내로 제한한다.
- “개선됐다” 대신 “실제 factor 반영 확인”, “작은 positive signal”, “held-out에서 검증 중”처럼 증거 강도에 맞는 표현을 쓴다.
- `0.1% 개선`은 크게 강조하지 않는다. 대신 `6.98° → 0.385°` 원인 교정과 공정한 검증 설계를 강조한다.

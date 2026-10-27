# 최종 발표용 기술 자료 — Direct R/V/P Learned IMU Attenuation

## 0. 최종 모델 고정본

이 문서는 **포스터/최종 발표에 사용할 모델 하나만** 설명한다. 이후에 수행한 `discrepancy`, `innovation`, `edge fusion` 실험은 최종 모델에 포함하지 않는다.

| 항목 | 최종 선택 |
|---|---|
| 모델 이름 | Direct R/V/P learned IMU attenuation on two-stage 15DoF DROID-SLAM |
| checkpoint | `checkpoints/learned_attenuation_rvp_quality_pretrain_joint2000_001000.pth` |
| 학습 데이터 | EuRoC `MH_03_medium`, `MH_04_difficult` |
| held-out 발표 평가 | EuRoC `V1_03_difficult` |
| 기준 모델 | original visual DROID, `droid.pth` |
| primary metric | 동일한 dense trajectory 2,149 timestamps의 whole-trajectory Sim(3) ATE RMSE |
| final learned ATE | **0.018467 m** |
| original DROID ATE | `0.018560 m` |
| 차이 | **0.000093 m = 0.093 mm = 0.501% 감소** |

발표용 표현은 다음이 정확하다.

> 원본 DROID visual backbone을 동결한 채, visual recurrent feature와 IMU preintegration 품질로 R/V/P별 IMU factor attenuation을 학습했다. V1_03 strict dense Sim(3) 조건에서 original visual DROID보다 수치상 낮은 ATE를 관찰했다 (`0.018560 → 0.018467 m`, 0.501%).

단, V1_03 한 sequence와 단일 실행에서 checkpoint를 선택했으며 차이가 0.093 mm로 작다. 따라서 **통계적으로 일반화된 성능 향상**이라고 과장하지 않는다.

## 1. 발표에 넣을 도식

아래 두 SVG가 최종 모델 기준 도식이다. 기존 `presentation_assets/final_vi_architecture.svg`는 이전 scalar confidence / `0.002` 가중치 설명을 담고 있어 최종 발표 그림으로 사용하지 않는다.

- [최종 runtime 모델 구조도](presentation_assets/final_direct_rvp_architecture.svg)
- [최종 학습 구조도](presentation_assets/final_direct_rvp_training.svg)

색상 범례는 다음과 같다.

| 색상 | 의미 |
|---|---|
| 파란색 | 동결된 original DROID visual modules |
| 주황색 | raw IMU, preintegration, sensor covariance/bias model |
| 보라색 | 새로 학습한 direct R/V/P attenuation head |
| 초록색 | visual–inertial initialization 및 15DoF BA |

## 2. 문제 정의와 핵심 기여

단안 DROID-SLAM은 visual correspondence로 pose와 inverse depth를 반복 최적화한다. IMU는 회전·속도·위치 변화에 대한 빠른 측정 정보를 제공하지만, bias, 노이즈, calibration/time mismatch, 특정 구간의 품질 저하가 있다. 모든 IMU factor를 같은 강도로 쓰면 BA가 오히려 나빠질 수 있다.

제안의 역할은 IMU 값을 새로 예측하거나 visual DROID를 재학습하는 것이 아니다. 이미 물리적으로 정해진 IMU covariance를 기준으로, temporal IMU factor마다 다음의 **성분별 attenuation**을 출력한다.

\[
(c_R,c_v,c_p)\in(0,1]^3.
\]

- \(c_R=1\), \(c_v=1\), \(c_p=1\): calibrated sensor covariance에 따른 full IMU factor.
- \(c<1\): 해당 interval·성분의 IMU 정보만 약화.
- \(c>1\)은 허용하지 않는다. 네트워크가 sensor covariance보다 더 정확한 IMU를 임의로 만들지 못하게 한다.
- 별도의 global \(\lambda_{\rm IMU}\)는 사용하지 않는다. factor의 기준 강도는 `1.0`이며 적응 역할은 \(c_R,c_v,c_p\)만 담당한다.

## 3. 입력 계약과 데이터 흐름

### 3.1 Visual 입력

1. EuRoC의 rectified monocular image를 입력한다.
2. `demo.py`의 strict original protocol은 raw 20 Hz stream 중 `--stride 2`로 tracking 후보를 처리한다.
3. 원본 DROID `MotionFilter`가 실제 keyframe 삽입 여부를 결정한다. V1_03 final run의 기본 threshold는 2.4이다.
4. sequence 종료 후 `trajectory_fill_stride=1`이 non-tracking frame pose를 채워 20 Hz dense trajectory를 출력한다.

### 3.2 IMU 입력

camera timestamp 사이의 고주파 gyro/accelerometer sample을 preintegration하여 temporal prior를 만든다.

\[
u_{ij}=\left[\Delta t,\Delta R_{ij},\Delta v_{ij},\Delta p_{ij},\Sigma_{ij},J_{b,ij}\right].
\]

- \(\Delta R,\Delta v,\Delta p\): relative rotation, velocity increment, position increment
- \(\Sigma\): propagated sensor covariance/information
- \(J_b\): accel/gyro bias correction Jacobian
- camera–IMU extrinsic은 rectified camera calibration과 짝이 맞는 YAML을 사용한다.

`droid_slam/geom/imu_factor.py:620-678`는 head가 읽는 7D temporal quality feature를 생성한다.

\[
q=\left[
\log(1+\Delta t/0.05),
\log(1+\|\Delta R\|/\Delta t),
\log(1+\|\Delta v\|/\Delta t),
\log(1+\|\Delta p\|/\Delta t^2),
\log(I_R/I_R^{\rm ref}),
\log(I_v/I_R^{\rm ref}),
\log(I_p/I_R^{\rm ref})
\right].
\]

여기서 마지막 세 항은 interval information을 clip 내 reference로 정규화한 값이다. 즉 raw IMU 신호 자체가 아니라 **preintegrated motion 규모와 센서 정보량**을 작은 MLP에 제공한다.

## 4. 원본 DROID와 새 모듈의 경계

### 4.1 유지·동결한 original DROID

`droid_slam/droid_net.py`의 원래 visual optimizer는 다음 경로다.

```text
image → FNet/CNet → all-pairs correlation
      → UpdateOperator ConvGRU hidden state h
      ├→ delta head: correspondence correction Δ
      └→ weight head: visual reprojection weight wᵛ
      → dense visual BA
```

- ConvGRU는 IMU를 직접 입력으로 받지 않는다. correspondence, motion feature, context를 누적하는 **visual recurrent memory**다.
- ConvGRU와 head는 다르다. ConvGRU는 `h`를 만들고, visual head는 `h`에서 \(\Delta,w^v\)를, 새 IMU head는 `h`와 \(q\)에서 \(c_R,c_v,c_p\)를 읽는다.
- 최종 학습에서는 `--freeze_non_imu`로 FNet, CNet, ConvGRU, visual correction/weight head를 동결했다. Visual correspondence가 IMU loss를 맞추기 위해 변하지 않는다.

### 4.2 새 direct R/V/P precision head

최종 checkpoint 기준으로 trainable parameter는 6,291개다.

```text
ConvGRU state h ∈ R^(128×H×W)
  └─ 1×1 Conv(128→32) + ReLU + spatial average  → v ∈ R^32

IMU quality q ∈ R^7
  └─ MLP(7→16→16, ReLU)                         → u ∈ R^16

[v;u] ∈ R^48
  └─ LayerNorm → Linear(48→32) + ReLU → Linear(32→3)
                                                       → z=(z_R,z_v,z_p)
                                                       → c=exp(-softplus(z))
```

구현 근거:

- head 선언: `droid_slam/droid_net.py:555-576`
- h와 q를 fuse해 attenuation을 만드는 함수: `droid_slam/droid_net.py:922-967`
- ConvGRU update와 visual head / IMU head의 동시 호출: `droid_slam/droid_net.py:1080-1113`

현재 작업트리에는 이후 ablation을 위해 discrepancy 입력과 innovation/edge-fusion adapter가 선언돼 있다. **최종 checkpoint에는 이들을 켜지 않는다.** 최종 runtime flag는 `--use_learned_imu_precision`만 사용하며 아래를 모두 사용하지 않는다.

```text
--use_imu_precision_discrepancy          # off
--use_imu_precision_innovation           # off
--use_imu_precision_visual_innovation    # off
--use_imu_precision_edge_fusion          # off
```

현재 코드가 historical 7D checkpoint를 새 input layout으로 읽을 때 추가 열을 0으로 migration한다. 해당 extensions가 off이면 입력도 0이고 추가 계수도 0이므로, **최종 7D direct model의 출력은 보존**된다. 발표 도식에는 이 compatibility code를 넣지 않는다.

## 5. 15DoF BA에서 confidence가 적용되는 정확한 위치

각 keyframe의 local optimization state는 다음이다.

\[
x_i=[T_i\;(6),\;v_i\;(3),\;b_{a,i}\;(3),\;b_{g,i}\;(3)] \in \mathbb{R}^{15}.
\]

inverse depth는 dense visual BA의 별도 상태로 함께 최적화된다. global `imu_acc_bias`, `imu_gyro_bias` **network parameter**는 학습에서 동결됐지만, 이것이 BA가 visual-only라는 뜻은 아니다. Runtime 15DoF BA는 각 keyframe의 local velocity와 local accel/gyro bias를 계속 최적화한다.

최적화는 개념적으로 다음과 같다.

\[
E(X)=E_{\rm visual}(X)+\sum_{(i,j)}E_{\rm imu}^{ij}(X;c_R,c_v,c_p).
\]

IMU residual order는 \([p,v,R,b_a,b_g]\)이고, head 출력 order는 \([R,v,p]\)다. 구현은 이 순서를 명시적으로 재배열한다.

\[
\tilde r_p=\sqrt{c_p}\,W_p r_p,\quad
\tilde r_v=\sqrt{c_v}\,W_v r_v,\quad
\tilde r_R=\sqrt{c_R}\,W_R r_R.
\]

- \(W\)는 propagated covariance에서 온 whiten/relative information scale이다.
- residual뿐 아니라 Jacobian에도 동일한 \(\sqrt c\)를 곱한다. 따라서 normal equation에서 해당 residual energy의 계수는 정확히 \(c\)가 된다.
- bias transition rows는 \(c\)로 감쇠하지 않고 sensor random-walk covariance를 사용한다.

코드 근거는 `droid_slam/geom/ba.py:1029-1061`이다. 이 구간은 `[p,v,R,b_a,b_g]` residual row에 `[c_p,c_v,c_R,1,1]`을 매핑한다.

## 6. 최종 runtime 실행 순서

### 6.1 Frame-level streaming

```text
demo.py
  → Droid.track(t, image, intrinsics)
  → MotionFilter: current frame을 keyframe으로 넣을지 결정
  → DepthVideo에 feature / timestamp / IMU prior 저장
  → DroidFrontend: active factor graph 생성·갱신
  → FactorGraph.update()
       1) current pose/depth로 reproject
       2) correlation + motion feature 추출
       3) UpdateOperator ConvGRU 실행
       4) visual Δ, wᵛ 및 learned (c_R,c_v,c_p) 출력
       5) visual + attenuated IMU factor로 local DBA
```

핵심 호출 지점은 `droid_slam/factor_graph.py`의 `update()`와 `droid_slam/droid_net.py`의 `UpdateOperator.forward()`다.

### 6.2 Two-stage 15DoF schedule

최종 V1_03 run에서는 다음 순서가 실제로 실행됐다.

```text
online local frontend
  visual-only DBA × 3
  → VI initialization (충분한 motion/history 후 scale, gravity, velocity, bias initial estimate)
  → joint full 15DoF DBA × 2

termination/global backend
  DROID-compatible visual-only global DBA × 7
  → joint global full 15DoF BA × 12
  → PoseTrajectoryFiller로 non-tracking frame 보간/복원
  → dense 20 Hz trajectory
```

- local split: `droid_slam/droid_frontend.py:145-163`
- termination-time visual pre-pass와 global full BA: `droid_slam/droid.py:730-779`
- `--no_vi_init_bootstrap`은 startup keyframe을 특수하게 오래 보존하는 bootstrap control을 끈 것이며, VI initialization 자체를 끈 것이 아니다.
- 이 구조는 visual 단계를 생략한 IMU-only backend가 아니라, visual optimum에서 full 15DoF refinement를 시작하는 **two-stage VIO**다.

## 7. 학습 구조

### 7.1 데이터와 unroll

- Training: `MH_03_medium`, `MH_04_difficult`
- Held-out selection/evaluation: `V1_03_difficult`
- Camera: original DROID protocol의 stride 2 및 MotionFilter keyframe manifest
- IMU: raw samples를 frame interval prior로 preintegrate
- Clip: 7 frames
- 최종 checkpoint 계보: 500-step IMU quality/augmentation calibration pretraining 후 joint fine-tuning의 1k checkpoint 선택

학습 중 작은 clip은 `visual ×3 → full 15DoF ×4` unroll을 사용한다. 이는 runtime의 긴 global `3→2→7→12` schedule과 목적이 다르다. 학습은 gradient를 안정적으로 전달할 짧은 truncated unroll이고, runtime은 한 sequence 전체에 대한 global refine다.

### 7.2 무엇을 학습하고 무엇을 고정했는가

| 항목 | 최종 모델에서 상태 |
|---|---|
| FNet/CNet, correlation, ConvGRU, visual Δ/w head | 동결 |
| direct precision visual projection + quality MLP + fusion MLP | 학습 |
| global IMU lambda | 사용하지 않음 |
| historical p/v fixed compatibility weight | learned precision mode에서 사용하지 않음 |
| global accel/gyro bias parameter | 학습에서 동결 |
| local velocity/local accel bias/local gyro bias | 매 15DoF BA에서 state로 최적화 |
| bias random-walk factor | sensor model에 따라 유지 |

### 7.3 손실

visual flow/geometry loss는 `w1=w2=w3=0`으로 두었다. 새 head에는 다음 두 loss가 전달된다.

\[
\mathcal L = 1.0\,\mathcal L_{\rm postBA\ pose}+1.0\,\mathcal L_{\rm aug}.
\]

1. **Post-BA pose loss**: confidence를 적용한 full 15DoF BA 뒤의 pose와 training GT pose를 비교한다. confidence가 실제로 제어하는 BA output을 직접 감독한다.
2. **Augmentation calibration loss**: training에서 gyro/accel noise, bias drift, 짧은 dropout을 synthetic하게 주입하고, 그때의 known reliability와 predicted attenuation을 log-space Smooth-L1로 맞춘다.

GT와 synthetic corruption은 **training loss에만** 들어간다. Final inference는 image, raw IMU, calibration, preintegration 결과만 사용한다.

구현 근거:

- trainable/inactive branch freeze: `train.py:1233-1245`, `train.py:1362-1468`
- post-BA pose loss와 augmentation calibration loss: `train.py:2271-2349`
- synthetic IMU degradation 및 known reliability target: `train.py:540-925`

중요한 정직성: 최종 선택 checkpoint는 이후의 component-specific augmentation-target 실험보다 앞선 direct model이다. 이후 gyro→R, accel→V/P target 분리와 disagreement feature를 시도했지만 이 checkpoint보다 낮은 V1_03 ATE를 만들지 못했다. 따라서 최종 모델을 “source-specific target으로 검증된 모델”이라고 주장하면 안 된다.

## 8. 평가 프로토콜과 결과

### 8.1 공정성

| 설정 | original DROID | final learned model |
|---|---:|---:|
| 입력 sequence | V1_03_difficult | V1_03_difficult |
| tracking stride / fill | 2 / 1 | 2 / 1 |
| rectified calibration | 동일 | 동일 |
| dense poses | 2,149 | 2,149 |
| GT frame convention | camera c2w | camera c2w |
| primary metric | whole-trajectory Sim(3) ATE | whole-trajectory Sim(3) ATE |

`tools/evaluate_euroc_recon.py`는 dense reconstruction pose와 GT timestamp를 맞춘 뒤 trajectory 전체에 하나의 Sim(3)을 fit한다. 이 평가는 scale-corrected trajectory shape/drift metric이다. Visual monocular DROID에는 metric scale이 없으므로 learned VIO의 metric SE(3) ATE와 original visual DROID의 metric SE(3) ATE를 headline으로 직접 비교하지 않는다.

### 8.2 수치

| Model | Sim(3) ATE RMSE ↓ | Sim(3) RPE 1 s ↓ | Sim(3) RPE 5 s ↓ | Dense poses |
|---|---:|---:|---:|---:|
| Original visual DROID | 0.018560 m | 1.129584 m | 2.746213 m | 2,149 |
| Final direct learned R/V/P | **0.018467 m** | **1.129534 m** | **2.746179 m** | 2,149 |

공개 release에는 재현 결과를 [results/V1_03_FINAL_RESULT.md](results/V1_03_FINAL_RESULT.md)에 고정해 둔다. Learned run의 metric SE(3) ATE는 `0.198508 m`이나, baseline visual DROID에는 scale observable metric trajectory가 없으므로 비교 headline으로 사용하지 않는다.

### 8.3 최종 모델을 선택한 이유

| 후보 | V1_03 dense Sim(3) ATE | 판정 |
|---|---:|---|
| Final poster direct R/V/P 1k | **0.018467 m** | 선택 |
| direct-discrepancy 1k | 0.018516 m | 더 나쁨 |
| direct-discrepancy total 3k | 0.018508 m | 더 나쁨 |
| direct-discrepancy total 4k | 0.018594 m | baseline보다 나쁨 |
| innovation adapter 1k | 0.018386 m | 수치상 낮지만, 별도 adapter를 포함해 최종 단순 구조에서는 제외 |

## 9. PPT 권장 구성과 발표 멘트

1. **문제/목표** — “IMU를 항상 강하게 넣지 않고, 어떤 interval의 어떤 residual을 얼마나 믿을지 학습한다.”
2. **원본 DROID 대비 확장 위치** — runtime SVG의 blue/purple/orange 영역을 사용한다.
3. **Direct R/V/P head** — 128→32와 7→16→16, 48→32→3 구조와 \(c=\exp(-\mathrm{softplus}(z))\)를 제시한다.
4. **15DoF BA integration** — state 15DoF와 \(\sqrt c\) residual/Jacobian scaling을 보여준다.
5. **학습 방식** — training SVG와 \(L_{\rm postBA\ pose}+L_{\rm aug}\)를 제시한다.
6. **Runtime schedule** — `local 3→2`, `global visual 7→full 12`, dense fill을 타임라인으로 제시한다.
7. **평가/결과** — 동일 2,149 dense poses, Sim(3) ATE table을 제시한다.
8. **정직한 결론** — “구조 구현과 수치상 개선은 확인했지만, 단일 sequence의 0.501% 차이이므로 반복·다중 sequence 검증이 필요하다.”

## 10. 재현 설정

최종 evaluation의 핵심 설정은 다음과 같다.

```text
WEIGHTS=checkpoints/learned_attenuation_rvp_quality_pretrain_joint2000_001000.pth
CONFIDENCE_MODE=precision
EUROC_ORIGINAL_PROTOCOL=1
STRIDE=2
TRAJECTORY_FILL_STRIDE=1
BUFFER=512
USE_IMU_BIAS_RANDOM_WALK=1
USE_IMU_FULL_COVARIANCE_WEIGHTING=1
IMU_TWO_STAGE_VISUAL_STEPS=3
IMU_TWO_STAGE_FULL_STEPS=2
IMU_TWO_STAGE_GLOBAL_VISUAL_PRE_STEPS=7
IMU_TWO_STAGE_GLOBAL_STEPS=12
USE_IMU_PRECISION_DISCREPANCY=0
USE_IMU_PRECISION_INNOVATION=0
USE_IMU_PRECISION_VISUAL_INNOVATION=0
USE_IMU_PRECISION_EDGE_FUSION=0
```

공개 release의 실행 진입점은 `tools/run_final_v1_03_evaluation.sh`이다. 이 스크립트는 final direct R/V/P checkpoint와 고정된 `3→2→7→12` schedule만 허용한다.

## 11. 예상 질문에 대한 짧은 답

**Q. confidence는 IMU 값을 줄이는 수동 tuning 아닌가?**  
A. 고정 global lambda가 아니라 temporal edge마다 visual context와 IMU quality에서 학습한다. 단, \(c\le1\) 제약은 calibrated sensor covariance보다 강한 가짜 정보를 만들지 않기 위한 안정성 제약이다.

**Q. global bias를 동결했는데 VIO가 아닌가?**  
A. 동결은 training parameter에 대한 말이다. Runtime 15DoF BA는 각 keyframe의 velocity와 local accel/gyro bias를 joint optimization한다. final setting에서 shared global-bias post-refinement만 끈 것이다.

**Q. ConvGRU는 IMU를 받는가?**  
A. 아니다. ConvGRU는 visual tracking memory다. 새 head가 ConvGRU state를 읽고 raw/preintegrated IMU quality와 결합한다.

**Q. original DROID보다 정말 좋은가?**  
A. V1_03 동일 dense Sim(3) 조건에서 0.501% 낮다. 다만 개선폭이 작아 여러 held-out sequence와 seed 반복이 추가로 필요하다.

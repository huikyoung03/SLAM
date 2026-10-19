# 15DoF DROID-IMU Confidence 실험 현황 보고서

작성일: 2026-10-16 (KST)

## 1. 실험 목표

기존 DROID-SLAM의 visual tracking/BA 구조와 MotionFilter 동작은 유지한 채,
15DoF visual-inertial BA에서 **각 IMU preintegration factor의 신뢰도**를
학습한다. 목표는 전역 IMU factor scale을 수동으로 작게 정한 뒤 confidence를
추가로 곱하는 이중 조절을 제거하고, IMU factor를 기본 강도(1.0)로 넣은 뒤
관측 구간별 신뢰도로만 감쇠하는 것이다.

## 2. 현재 모델 구성

| 항목 | 현재 설정 |
| --- | --- |
| 상태 | pose 6DoF + local velocity 3 + local accel bias 3 + local gyro bias 3 = 15DoF |
| 전역 IMU factor | 고정 1.0 (`imu_confidence_only_weighting`) |
| 학습 대상 | scalar DVI confidence head 및 preintegrated-IMU quality fusion MLP (418 parameters) |
| confidence | `sigmoid(visual GRU logit + IMU-quality logit)` |
| IMU quality 입력 | interval duration, angular/acceleration/position proxy, rotation/velocity/position information |
| 동결 대상 | visual backbone, visual GRU/update weights, global gyro/accelerometer bias |
| BA 내부 상태 | local velocity/local bias는 각 unroll의 DBA에서 최적화되는 latent state이며 네트워크 파라미터가 아님 |
| unroll | pose-only 3회 + full 15DoF IMU DBA 4회 (총 7회) |

confidence가 1이면 covariance-relative IMU information이 그대로 BA에 들어가며,
confidence가 작을수록 해당 temporal IMU factor만 약해진다. 전역 `lambda_imu=0.002`
같은 별도 축소 계수는 이 실험 경로에서 사용하지 않는다.

## 3. 데이터와 protocol

| 용도 | 시퀀스 | 상태 |
| --- | --- | --- |
| 학습 | MH_03_medium | cam0 2,631 frames, IMU 26,302 samples, 준비 완료 |
| 학습 | MH_04_difficult | cam0 1,976 frames, IMU 19,753 samples, 다운로드·MCAP 변환 완료 |
| 중간 검증 / early stop | V1_02_medium | prepared input, finite-difference IMU prior, GT 준비 완료 |

카메라 후보 stream은 EuRoC 20 Hz 원본에 `stride=2`를 적용한 10 Hz이다. 이는
기존 DROID EuRoC 평가와 비교 가능한 protocol을 유지하기 위한 선택이다.
MotionFilter의 visual threshold는 2.4이며, IMU가 MotionFilter의 선택을 강제하지
않는다 (`use_imu_filter=False`).

## 4. MotionFilter manifest가 필요한 이유

실제 DROID는 10 Hz 후보 프레임 전체를 BA keyframe으로 쓰지 않는다. visual
MotionFilter가 충분한 visual motion이 있는 프레임만 선택하므로 선택된 키프레임
사이의 실제 시간 간격은 가변적이다.

각 학습 시퀀스의 manifest는 다음을 고정한다.

1. 동결된 visual DROID/MotionFilter가 선택한 원본 frame index와 timestamp
2. 연속 선택 키프레임 사이의 실제 IMU 구간
3. 그 구간에 대해 합성되는 preintegrated IMU measurement/covariance

따라서 training clip은 random raw-frame gap이 아니라 **실제 DROID가 선택한 연속
keyframe 7개**로 구성된다. MotionFilter 자체나 visual weight를 학습하는 작업은
아니다.

## 5. 2026-10-16 현재 실행 상태

### 5.1 진행 중

- `tools/export_euroc_motion_filter_manifests.sh`가 실행 중이다.
- `MH_03_medium.csv` manifest 생성 완료: 16,558 bytes.
- 현재 `MH_04_difficult`에 대해 MotionFilter manifest를 생성 중이다.
- manifest 출력은 sequence 실행이 정상 종료된 뒤 기록된다.

### 5.2 아직 시작하지 않은 작업

- MH03+MH04 confidence 학습은 아직 시작하지 않았다.
- 2k/4k/6k/8k V1_02 intermediate evaluation도 아직 시작하지 않았다.

학습 실행의 선행 조건은 두 파일이 모두 존재하는 것이다.

```text
/home/ubuntu/SLAM/datasets/euroc_stage3/motion_filter_manifests_stride2/MH_03_medium.csv
/home/ubuntu/SLAM/datasets/euroc_stage3/motion_filter_manifests_stride2/MH_04_difficult.csv
```

## 6. 새 학습 및 early-stop 설계

총 learning-rate schedule은 8,000 steps이다. GPU 메모리 충돌 없이 V1_02 전체
평가를 수행하기 위해 학습과 평가는 별도 프로세스로 순차 실행한다.

```text
2k training -> process exit / GPU release -> V1_02 evaluation
    -> pass: optimizer + scheduler state resume -> 4k training
    -> decline: stop
4k -> V1_02 -> 6k
6k -> V1_02 -> 8k
```

각 stage에서 다음이 저장된다.

```text
checkpoints/<RUN_NAME>_002000.pth
checkpoints/<RUN_NAME>_004000.pth
checkpoints/<RUN_NAME>_006000.pth
checkpoints/<RUN_NAME>_008000.pth

checkpoints/<RUN_NAME>_trainstate_XXXXXX.pth
```

`trainstate`에는 model, Adam optimizer, OneCycleLR scheduler, RNG state가
저장된다. 따라서 다음 2k stage는 독립 fine-tuning이 아니라 동일한 8k schedule의
continuation이다.

### Early-stop 조건

검증은 고정된 V1_02 medium에서 metric SE(3) ATE RMSE로 판단한다.

1. 현재 checkpoint의 tracked pose coverage가 과거 최고 coverage의 95% 미만이면 중단
2. 현재 SE(3) ATE가 이전 최고 ATE보다 0.002 m(2 mm)를 초과해 악화되면 중단
3. 2 mm 이하는 수치적 평가 변동으로 보고 다음 stage를 허용

필요 시 `EARLY_STOP_DETERIORATION_TOLERANCE_M=0`으로 설정하여, 아주 작은
악화도 즉시 중단하는 strict mode를 사용할 수 있다.

## 7. 기존 단일-MH03 결과 (참고용)

이전 true 15DoF confidence-only, MH03 단독 학습과 V1_02 평가 결과는 다음과 같다.
새 실험은 데이터와 temporal sampling distribution이 바뀌므로 직접적인 성능
비교값이 아니라 출발점으로만 사용한다.

| checkpoint | V1_02 SE(3) ATE RMSE | tracked poses | mean confidence |
| --- | ---: | ---: | ---: |
| 1k | 0.189939 m | 274 | 0.9476 |
| 2k | 0.189887 m | 274 | 0.9289 |
| 3k | 0.189638 m | 274 | 0.9247 |

3k가 해당 과거 run에서 가장 낮았지만, 개선 폭은 작았다. 따라서 이번 실험은
MH04 확장과 실제 MotionFilter temporal distribution이 confidence 학습의 일반화에
주는 영향을 검증하는 실험이다.

## 8. 다음 실행 순서

1. MH04 manifest 생성 완료를 확인한다.
2. 2k/4k/6k/8k early-stop controller를 실행한다.
3. controller가 생성한 `outputs/<RUN_NAME>/early_stop_v1_02/state.json`에서
   checkpoint별 ATE, coverage, continue/stop 결정을 확인한다.
4. 최종 선택 checkpoint 하나만 별도의 held-out 평가 protocol에 사용한다.

실행기는 `tools/run_15dof_confidence_early_stop.sh`, 검증기는
`tools/validate_15dof_confidence_checkpoint.sh`, 중단 판정기는
`tools/monitor_checkpoint_early_stop.py`에 있다.

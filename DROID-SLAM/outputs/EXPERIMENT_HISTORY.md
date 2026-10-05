# 실험 이력과 결과 요약

이 폴더는 중간 실험의 결과를 추적하기 위한 보존 목록이다. 각 실험 폴더에는
`summary.md`/`summary.csv`, 비교표, checkpoint 선택 결과, 또는 실행 로그를 남겼다.
대용량 reconstruction (`.pth`), PLY, 상세 debug CSV는 필요 시 휴지통에서 복구할 수
있도록 별도 보관했다.

## 1. 초기 visual/IMU ablation (2026-09-02)

- `euroc_ablation_perf_450_20260902_135803`
- `euroc_ablation_stride2_10fps_225_20260902_143204`
- `euroc_fullseq_stride2_compare_20260902_151813`

visual-only, rotation/full IMU BA, confidence weighting의 초기 비교 결과다.

## 2. Jacobian·center residual 확인 (2026-09-03)

- `euroc_after_finetune_center_v2`
- `imu_bias_jacobian_check`
- `euroc_midcheck`

IMU residual, Jacobian, center convention을 점검한 중간 결과다.

## 3. 9DoF·adaptive 초기 비교 (2026-09-18 ~ 2026-09-21)

- `euroc_v1_03_9dof_compare_20260918`
- `euroc_9dof_adaptive_smoke_20260918`
- `euroc_v1_03_adaptive_recal_ft_2000_20260921`

9DoF IMU factor와 adaptive gate의 초기 연결·비교 결과다.

## 4. Noise/covariance/confidence 탐색 (2026-09-21 ~ 2026-09-25)

- `euroc_heldout_v2_01_*`, `euroc_heldout_v2_03_*`
- `euroc_heldout_noise_model_report_20260922.md`

raw-reference, no-limit, learned PV, covariance fix, fixed-bias, noise
fine-tuning, confidence calibration 단계의 V2_01/V2_03 평가 요약이다.

## 5. 체크포인트 선택·교차 시퀀스 검증 (2026-09-28 ~ 2026-09-29)

- `euroc_checkpoint_selection_V2_01_easy_droid_cam0_20260928_223138/selection.txt`
- `euroc_checkpoint_selection_v2_01_hospital_20260928`
- `euroc_crossseq_v2_03_hospital_8k_20260928`

axis-fix 2k/4k/6k/8k 후보를 V2_01에서 비교해 6k를 고정한 과정과
V2_03 교차 검증 결과다.

## 6. Shared-bias 및 matched-control (2026-09-29 ~ 2026-10-01)

- `euroc_v2_03_shared_bias_20260929_132338`
- `euroc_v2_03_full_joint_shared_bias_20260930_131538`
- `euroc_v2_03_matched_control_20261001_160056`
- `_smoke_outer_compare.md`, `_smoke_reintegration_compare.md`

fixed-bias, joint shared-bias, outer-relinearization control의 비교 결과다.
현재 실행 중인 V2_03 검증은 `full_joint_shared_bias`의 visual-only reconstruction을
재사용하므로 해당 파일은 원본 형태로 유지한다.

## 7. 최신 low-rate/VI initialization 검증 (2026-10-02 ~ 2026-10-05)

- `euroc_v2_03_lowrate_bias_20261002_141128`
- `euroc_v2_01_lowrate_bias_20261005_022251`
- `euroc_v2_01_viinitfix_20261005_101230`
- `euroc_v2_03_matched_livebias_20261005_172721` (실행 중)

저주기 bias refinement와 VI initialization 수정을 검증하는 최신 결과다.

## 체크포인트 보존 원칙

`checkpoints/`에는 학습 계열별 `final` checkpoint와, 선택 실험에 필요한
axis-fix 2k/4k/6k/8k 후보를 보존한다. 나머지 epoch 중간본은 휴지통에 있어,
재개 학습이나 특정 epoch 재평가가 필요할 때 복구할 수 있다.

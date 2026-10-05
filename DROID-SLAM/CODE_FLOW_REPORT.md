# DROID-SLAM 코드 흐름 분석 보고서

작성일: 2026-07-20  
분석 대상: `/home/ubuntu/DROID-SLAM`

## 1. 요약

이 저장소는 DROID-SLAM의 Python 추론/학습 코드, CUDA backend 확장, 데이터셋/평가 스크립트, 그리고 로컬로 추가된 IMU 실험 도구를 포함한다.

전체 추론 흐름은 다음 한 줄로 정리할 수 있다.

```text
image_stream
  -> Droid 또는 DroidAsync 생성
  -> MotionFilter로 keyframe 후보 선별
  -> DroidFrontend로 local factor graph 최적화
  -> DroidBackend로 global refinement
  -> PoseTrajectoryFiller로 skipped frame pose 보간/보정
  -> trajectory/reconstruction 저장
```

핵심은 `DepthVideo`에 선택된 프레임의 이미지, feature, pose, inverse depth, intrinsics를 쌓고, `FactorGraph`가 프레임 간 reprojection factor를 구성한 뒤, neural update operator와 CUDA bundle adjustment를 반복해 pose/depth를 갱신하는 구조다.

## 2. 디렉터리 구조와 역할

```text
DROID-SLAM/
  demo.py                         추론/demo 진입점
  train.py                        TartanAir 기반 DDP 학습 진입점
  setup.py                        droid_backends CUDA 확장 빌드
  droid_slam/
    droid.py                      동기식 SLAM orchestrator
    droid_async.py                frontend/backend 분리 비동기 orchestrator
    depth_video.py                SLAM 상태 버퍼
    motion_filter.py              입력 프레임 선별 및 feature 추출
    droid_frontend.py             online/local factor graph 최적화
    droid_backend.py              종료/비동기 global refinement
    factor_graph.py               factor 추가/삭제/update/BA 호출
    droid_net.py                  feature encoder + recurrent update network
    trajectory_filler.py          non-keyframe trajectory 보간/보정
    geom/                         projection, BA, graph, loss
    modules/                      encoder, correlation, ConvGRU
    data_readers/                 학습/stream 데이터 로더
    visualizer/                   moderngl 실시간 시각화
    imu_prior.py, imu_residual.py IMU 실험용 보조 코드
  src/                            droid_backends C++/CUDA 구현
  evaluation_scripts/             TUM, EuRoC, ETH3D, TartanAir 평가
  tools/                          다운로드, IMU, reconstruction 분석 도구
  thirdparty/                     lietorch, pytorch_scatter, tartanair_tools
```

## 3. 추론 실행 단계

### 3.1 `demo.py`: 입력 준비와 전체 실행

`demo.py`는 이미지 폴더와 calibration 파일을 받아 SLAM을 수행하는 기본 진입점이다.

1. `sys.path.append('droid_slam')`로 내부 모듈을 import한다.
2. `image_stream(imagedir, calib, stride)`가 이미지를 정렬된 순서로 읽는다.
3. calibration 파일에서 `fx, fy, cx, cy` 및 distortion 계수를 읽는다.
4. distortion 계수가 있으면 OpenCV `undistort`를 수행한다.
5. 입력 이미지를 대략 `384 * 512` 픽셀 수로 resize하고, H/W를 8의 배수로 crop한다.
6. resize 비율에 맞게 intrinsics를 보정한다.
7. 첫 프레임에서 실제 `args.image_size`를 결정하고 `Droid` 또는 `DroidAsync`를 만든다.
8. 매 프레임마다 `droid.track(t, image, intrinsics=intrinsics)`를 호출한다.
9. 입력이 끝나면 `droid.terminate(image_stream(...))`로 backend refinement와 trajectory filling을 수행한다.
10. `--reconstruction_path`가 있으면 `save_reconstruction()`으로 `tstamps`, `images`, `disps`, `poses`, `intrinsics`를 `.pth`로 저장한다.

실행상 핵심 분기:

```python
droid = DroidAsync(args) if args.asynchronous else Droid(args)
droid.track(t, image, intrinsics=intrinsics)
traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))
```

## 4. 동기식 SLAM 흐름: `Droid`

`droid_slam/droid.py`의 `Droid` 클래스는 동기식 실행 orchestrator다.

### 4.1 초기화 단계

`Droid.__init__()`은 다음 객체를 순서대로 만든다.

1. `DroidNet` 생성 및 `droid.pth` weight 로드
2. `DepthVideo` 생성
3. `MotionFilter` 생성
4. `DroidFrontend` 생성
5. `DroidBackend` 생성
6. optional visualizer process 시작
7. `PoseTrajectoryFiller` 생성

```text
Droid(args)
  -> load_weights()
  -> DepthVideo()
  -> MotionFilter()
  -> DroidFrontend()
  -> DroidBackend()
  -> PoseTrajectoryFiller()
```

### 4.2 프레임별 처리

`Droid.track()`은 gradient 없이 다음을 수행한다.

```text
track(t, image, intrinsics)
  -> MotionFilter.track()
       충분한 motion이면 DepthVideo.append()
  -> DroidFrontend.__call__()
       warmup 이후 local graph update
```

즉, `track()`은 모든 입력 프레임을 바로 최적화하지 않는다. `MotionFilter`가 선택한 프레임만 `DepthVideo`에 들어가고, frontend는 `DepthVideo.counter`가 증가한 경우에만 의미 있는 update를 수행한다.

### 4.3 종료 단계

`Droid.terminate()`는 다음을 수행한다.

1. frontend 삭제
2. CUDA cache 정리
3. `DroidBackend(steps=7)` 실행
4. `DroidBackend(steps=12)` 실행
5. `PoseTrajectoryFiller(stream)`로 원본 stream의 skipped frame pose까지 생성
6. 내부 pose convention을 뒤집기 위해 `camera_trajectory.inv()` 반환

## 5. 상태 저장소: `DepthVideo`

`droid_slam/depth_video.py`의 `DepthVideo`는 SLAM 전체 상태를 담는 공유 버퍼다.

주요 tensor:

```text
tstamp      [buffer]
images      [buffer, 3, H, W]
poses       [buffer, 7]              # [tx, ty, tz, qx, qy, qz, qw]
disps       [buffer, H/8, W/8]        # inverse depth/disparity
disps_sens  [buffer, H/8, W/8]        # RGB-D/stereo sensor depth가 있을 때
disps_up    [buffer, H, W]            # upsampled disparity
intrinsics  [buffer, 4]               # feature scale 기준 intrinsics
fmaps       [buffer, rig, 128, H/8, W/8]
nets        [buffer, 128, H/8, W/8]
inps        [buffer, 128, H/8, W/8]
```

핵심 method:

- `append()`: 새 selected frame과 feature/context를 버퍼에 추가한다.
- `reproject(ii, jj)`: `geom.projective_ops.projective_transform()`으로 frame `ii`의 points를 frame `jj`에 투영한다.
- `distance(ii, jj)`: CUDA `droid_backends.frame_distance()`로 frame distance를 계산한다.
- `ba(...)`: CUDA `droid_backends.ba()`를 호출해 pose와 disparity를 최적화한다.
- `normalize()`: monocular scale ambiguity를 줄이기 위해 disparity 평균 기준으로 pose/depth scale을 정규화한다.
- `upsample(ix, mask)`: network가 예측한 convex upsampling mask로 full-resolution disparity를 만든다.

## 6. 입력 프레임 선별: `MotionFilter`

`droid_slam/motion_filter.py`의 목적은 “충분히 움직인 프레임만 SLAM keyframe 후보로 넣는 것”이다.

처리 단계:

1. 입력 이미지를 GPU로 옮긴다.
2. BGR을 RGB 순서로 바꾸고 ImageNet mean/std로 normalize한다.
3. `fnet`으로 correlation용 feature map `gmap`을 뽑는다.
4. 첫 프레임이면 `cnet`으로 `net/inp`를 뽑고 identity pose로 바로 `DepthVideo.append()`한다.
5. 두 번째 프레임부터는 이전 selected frame의 `self.fmap`과 현재 `gmap`으로 `CorrBlock`을 만든다.
6. `update` network를 한 번 호출해 대략적인 2D motion `delta`를 예측한다.
7. `delta.norm(dim=-1).mean()`이 `filter_thresh`보다 크면 새 프레임으로 append한다.
8. 작으면 frame을 skip한다.

이 단계는 정확한 pose를 구하는 단계가 아니라, 실시간성을 위해 입력 frame 수를 줄이는 gate 역할이다.

## 7. Online frontend: `DroidFrontend`

`DroidFrontend`는 최근 window에 대해 local factor graph를 유지하고 update한다.

### 7.1 Warmup 초기화

`DepthVideo.counter.value == args.warmup`이 되면 `_initialize()`가 실행된다.

```text
add_neighborhood_factors(t0, t1, r=3)
  -> graph.update() 8회
add_proximity_factors(...)
  -> graph.update() 8회
ready = 1
old factors 일부 inactive 저장
```

이때 초기 pose/depth가 어느 정도 안정화되어 이후 online update가 가능해진다.

### 7.2 새 keyframe update

초기화 이후 새 selected frame이 들어오면 `_update()`가 실행된다.

1. 오래된 active factor를 inactive로 이동한다.
2. `add_proximity_factors()`로 최근 window 주변의 frame pair edge를 추가한다.
3. sensor depth가 있으면 현재 frame disparity에 반영한다.
4. `graph.update(..., use_inactive=True)`를 여러 번 실행한다.
5. frame distance가 너무 작으면 `rm_keyframe()`으로 중복 keyframe을 제거한다.
6. 다음 프레임의 pose/depth 초기값을 설정한다.
7. visualizer가 갱신할 수 있도록 `video.dirty`를 표시한다.

`motion_damping`이 있으면 `_init_next_state()`에서 damped linear velocity model을 사용해 다음 pose 초기값을 만든다.

## 8. Factor graph와 최적화: `FactorGraph`

`droid_slam/factor_graph.py`는 DROID-SLAM의 최적화 루프 중심이다.

### 8.1 Edge/factor 구성

factor는 frame pair `(ii, jj)`로 표현된다.

- `add_neighborhood_factors(t0, t1, r)`: 시간적으로 가까운 프레임을 강제 연결한다.
- `add_proximity_factors(...)`: `DepthVideo.distance()`가 작은 후보를 찾아 bidirectional edge를 추가한다.
- `rm_factors(mask, store=True)`: 오래된 factor를 inactive buffer로 옮기거나 제거한다.
- `rm_keyframe(ix)`: 중복 keyframe을 제거하고 관련 tensor/factor index를 당긴다.

### 8.2 `update()`의 내부 단계

```text
FactorGraph.update()
  -> video.reproject(ii, jj)
  -> motion feature 구성
       [coords1 - coords0, target - coords1]
  -> corr(coords1) sampling
  -> update_op(net, inp, corr, motion, ii, jj)
       returns net, delta, weight, damping, upmask
  -> target = coords1 + delta
  -> weight = predicted confidence
  -> video.ba(target, weight, damping, ii, jj, ...)
  -> optional video.upsample()
```

여기서 neural network는 직접 pose를 예측하지 않는다. 대신 현재 reprojection 좌표를 어느 방향으로 옮기면 좋을지 `delta`, 그 residual을 얼마나 믿을지 `weight`, depth damping `eta`를 예측한다. 실제 pose/depth 갱신은 bundle adjustment가 맡는다.

### 8.3 Backend용 `update_lowmem()`

backend는 전체 trajectory를 대상으로 factor가 많아지므로 `corr_impl="alt"`와 `AltCorrBlock`을 사용한다.

`update_lowmem()`은 edge를 chunk 단위로 나누어 correlation/update를 수행하고, 그 뒤 전체 `video.ba()`를 실행한다. 메모리 사용량을 낮추기 위한 구현이다.

## 9. Bundle adjustment와 CUDA backend

Python 추론의 dense BA는 최종적으로 `DepthVideo.ba()`에서 `droid_backends.ba()`를 호출한다.

`setup.py`는 다음 source로 `droid_backends` 확장을 빌드한다.

```text
src/droid.cpp
src/droid_kernels.cu
src/correlation_kernels.cu
src/altcorr_kernel.cu
```

`src/droid.cpp`가 Python에 노출하는 주요 함수:

- `ba`: dense bundle adjustment
- `frame_distance`: keyframe/proximity 판단용 거리
- `projmap`
- `depth_filter`: reconstruction point filtering
- `iproj`: depth/disparity를 point cloud로 back-project
- `corr_index_forward/backward`
- `altcorr_forward/backward`

학습 코드에서는 differentiable Python BA(`droid_slam/geom/ba.py`)도 사용한다. 이 구현은 Jacobian/residual을 만들고 Schur complement solver로 pose/depth increment를 구한 뒤 lietorch retraction을 적용한다.

## 10. 네트워크 구조: `DroidNet`

`droid_slam/droid_net.py`의 핵심 구성:

```text
DroidNet
  fnet: BasicEncoder(output_dim=128, norm_fn='instance')
  cnet: BasicEncoder(output_dim=256, norm_fn='none')
  update: UpdateModule
```

### 10.1 Feature encoders

`BasicEncoder`는 residual CNN encoder다. 입력 이미지는 1/8 해상도의 feature map으로 내려간다.

- `fnet`: frame matching/correlation volume용 feature
- `cnet`: recurrent update용 context. 출력 256 channel을 `net` 128, `inp` 128로 나눈다.

### 10.2 Correlation

`modules/corr.py`는 두 가지 correlation path를 제공한다.

- `CorrBlock`: full all-pairs correlation volume을 만들고 pyramid sampling을 수행한다. frontend/local update에 적합하다.
- `AltCorrBlock`: feature map 전체를 보관하고 필요한 edge만 CUDA kernel로 correlation sampling한다. backend/low-memory path에 적합하다.

### 10.3 UpdateModule

`UpdateModule`은 RAFT 계열 recurrent update block이다.

입력:

- hidden state `net`
- context input `inp`
- correlation feature `corr`
- motion feature `[flow, residual]`
- edge index `ii, jj`

출력:

- 갱신된 hidden state `net`
- reprojection target offset `delta`
- residual confidence `weight`
- BA damping/depth regularization `eta`
- disparity upsample mask `upmask`

## 11. Backend refinement

`droid_slam/droid_backend.py`에는 동기/비동기 backend가 있다.

### 11.1 동기식 `DroidBackend`

종료 시점에 전체 frame 수 `t`를 기준으로 새 `FactorGraph`를 만들고:

1. monocular이고 sensor depth가 없으면 `video.normalize()`
2. `FactorGraph(..., corr_impl="alt", max_factors=16*t)` 생성
3. `add_proximity_factors()`로 전체 trajectory edge 구성
4. `update_lowmem(steps=...)` 수행
5. edge 제거 후 `dirty` 표시

`Droid.terminate()`는 이 backend를 7 step, 12 step 두 번 호출한다.

### 11.2 비동기식 `DroidAsync`

`DroidAsync`는 두 개의 `DepthVideo`를 사용한다.

```text
video1: frontend/MotionFilter가 실시간으로 쓰는 버퍼
video2: backend process가 global refinement하는 버퍼
```

`backend_process()`는 일정 조건에서 `video1`의 일부 frame을 `video2`로 복사하고, `align_pose_fragements()`로 pose fragment와 scale을 맞춘 뒤 `DroidAsyncBackend`를 반복 실행한다. 종료 시 `video2.ready = 1`로 마지막 iteration을 알리고 process를 join한다.

## 12. Trajectory filling

`droid_slam/trajectory_filler.py`의 `PoseTrajectoryFiller`는 `MotionFilter` 때문에 빠진 non-keyframe pose를 채운다.

처리 흐름:

1. 원본 `image_stream`을 다시 순회한다.
2. 16개씩 chunk를 모은다.
3. 각 timestamp가 `DepthVideo`의 어느 keyframe 사이에 있는지 찾는다.
4. `SE3.log/exp`로 선형 속도 기반 pose interpolation을 한다.
5. 임시 frame을 `DepthVideo` 뒤쪽에 추가한다.
6. keyframe `t0/t1`과 임시 frame 사이 factor를 만든다.
7. `graph.update(..., motion_only=True)` 6회로 pose만 보정한다.
8. 임시 frame 수만큼 counter를 되돌리고 결과 pose를 이어 붙인다.

결과적으로 최종 반환 trajectory는 입력 stream 기준 frame 수에 맞춰진다.

## 13. 학습 흐름

`train.py`는 DDP 기반 학습 진입점이다.

```text
mp.spawn(train, nprocs=args.gpus)
  -> setup_ddp()
  -> DroidNet + DDP
  -> dataset_factory(['tartan'])
  -> DataLoader
  -> frame graph 샘플링
  -> DroidNet.forward()
  -> geodesic/residual/flow loss
  -> backward + optimizer/scheduler
```

학습 데이터는 `data_readers/base.py`의 `RGBDDataset`과 `data_readers/tartan.py`의 `TartanAir`가 담당한다.

`RGBDDataset`은 scene별로 covisibility/frame graph를 계산해 cache에 저장하고, 학습 시 한 기준 frame에서 flow distance가 적당한 주변 frame을 골라 `n_frames` sequence를 만든다.

`DroidNet.forward()`는 추론 path와 비슷한 recurrent update + BA 구조를 differentiable하게 돌린다. 각 step의 pose/depth/residual을 저장하고, `geom/losses.py`에서 다음 loss를 계산한다.

- `geodesic_loss`: 상대 pose 오차
- `residual_loss`: reprojection residual 크기
- `flow_loss`: ground-truth pose/depth가 유도하는 optical flow와 예측 flow 차이

## 14. 평가/시각화/결과 도구

평가 스크립트는 dataset별 stream 생성과 metric 계산을 담당하며, 공통적으로 `Droid` 또는 `DroidAsync`를 사용한다.

- `evaluation_scripts/test_tum.py`: TUM RGB-D monocular evaluation, evo APE 사용
- `evaluation_scripts/test_euroc.py`: EuRoC mono/stereo evaluation, undistort/rectify 포함
- `evaluation_scripts/test_tartanair.py`: TartanAir mono/stereo test scenes evaluation
- `evaluation_scripts/test_eth3d.py`: ETH3D/RGB-D evaluation
- `evaluation_scripts/validate_tartanair.py`: validation split 평가

결과 확인 도구:

- `view_reconstruction.py`: saved reconstruction `.pth`를 Open3D로 시각화
- `export_reconstruction_ply.py`: saved reconstruction을 point cloud `.ply`로 export
- `tools/plot_trajectory_topview.py`: trajectory top-view plot
- `tools/compare_recon_metrics.py`, `tools/compare_nonloop_metrics.py`: reconstruction/trajectory 특성 비교

## 15. IMU 관련 확장 상태

현재 tree에는 IMU 실험용 코드가 추가되어 있다.

### 15.1 `droid_slam/imu_prior.py`

`imu_prior.csv`를 읽고 `dr_x, dr_y, dr_z` rotation vector를 quaternion으로 바꾼다. `ImuRotationAccumulator`는 프레임별 IMU delta rotation을 누적해 translation 0, rotation prior만 있는 SE3 vector를 만들 수 있다.

### 15.2 `droid_slam/imu_residual.py`

`ImuRotationResidual`은 DROID의 CUDA BA objective에 직접 항을 추가하지 않는다. 대신 visual optimization 이후 현재 pose quaternion과 IMU preintegrated relative rotation의 residual을 계산하고, `slerp`로 현재 rotation을 soft regularization한다.

중요한 특징:

- `compose_order`로 `prev_dq`, `dq_prev`를 선택할 수 있다.
- residual이 너무 크면 좌표계/동기화 오류로 보고 무시한다.
- debug CSV를 남길 수 있다.
- 특정 turn 구간으로 보이는 index `35 <= ix <= 45`에 가중치를 키우는 실험 코드가 들어 있다.

### 15.3 `tools/make_imu_selected_session_webp.py`

기존 session의 `imu_prior.csv`를 보고 회전량이 큰 구간은 더 촘촘히, 직선 구간은 stride 기반으로 frame을 선택해 새 WebP session을 만든다. 선택된 frame 사이의 IMU prior는 quaternion 누적으로 합성한다.

### 15.4 `tools/calibrate_imu_camera_rotation.py`

saved reconstruction의 visual relative rotation과 `imu_prior.csv`의 IMU rotation vector를 비교해 `R_imu_to_cam`을 SVD로 추정한다. 추정된 회전 정렬을 적용한 CSV도 저장한다.

### 15.5 현재 통합상 주의점

현재 `run_webp_imu_selected_experiment.sh`는 `demo.py`에 다음 옵션을 넘긴다.

```text
--imu_mode
--imu_prior
--imu_residual
--imu_residual_weight
--imu_residual_window
--imu_residual_compose_order
--imu_residual_debug
```

하지만 현재 `demo.py`의 argparse에는 해당 옵션들이 정의되어 있지 않다. 따라서 이 shell script를 그대로 실행하면 `unrecognized arguments`로 실패할 가능성이 높다.

또한 `imu_residual.py`는 현재 `demo.py`, `droid.py`, `droid_frontend.py`, `droid_backend.py` 어디에서도 import/호출되지 않는다. 즉 IMU residual regularizer는 파일로는 존재하지만 DROID 추론 루프에 아직 연결되지 않은 상태다.

`tools/imu_preintegrate.py`도 후반부 `rows.append({...})`, CSV 저장 부분이 `...` placeholder로 남아 있어 완성된 prior generator로 보기 어렵다.

## 16. 코드 단계별 핵심 호출 그래프

### 16.1 동기 추론

```text
demo.py
  image_stream()
  Droid(args)
    Droid.load_weights()
      DroidNet()
    DepthVideo()
    MotionFilter(net, video)
    DroidFrontend(net, video, args)
      FactorGraph(video, net.update)
    DroidBackend(net, video, args)
    PoseTrajectoryFiller(net, video)

  for each frame:
    Droid.track(t, image, intrinsics)
      MotionFilter.track()
        fnet/cnet
        CorrBlock
        UpdateModule.forward()
        DepthVideo.append()
      DroidFrontend.__call__()
        _initialize() or _update()
          FactorGraph.add_*_factors()
          FactorGraph.update()
            DepthVideo.reproject()
            CorrBlock(...)
            UpdateModule.forward()
            DepthVideo.ba()
              droid_backends.ba()

  Droid.terminate(stream)
    DroidBackend.__call__(7)
      FactorGraph(corr_impl="alt")
      add_proximity_factors()
      update_lowmem()
        AltCorrBlock
        droid_backends.ba()
    DroidBackend.__call__(12)
    PoseTrajectoryFiller(stream)
      interpolate SE3
      motion_only graph update
    return trajectory
```

### 16.2 비동기 추론

```text
demo.py
  DroidAsync(args)
    video1 = DepthVideo(frontend_device)
    video2 = DepthVideo(backend_device)
    MotionFilter(net, video1)
    DroidFrontend(net, video1)
    backend_process(video1, video2)
      copy frontend fragment
      align_pose_fragements()
      DroidAsyncBackend(video2)
      update_lowmem()
    PoseTrajectoryFiller(net, video2)

  main process:
    MotionFilter + Frontend keep updating video1

  backend process:
    periodically refines video2

  terminate:
    video2.ready = 1
    join backend process
    trajectory_filler(video2)
```

## 17. 데이터 convention

현재 코드 기준으로 중요한 convention은 다음과 같다.

- 이미지 입력은 OpenCV BGR로 읽히고 network 입력 전 RGB 순서로 바뀐다.
- network/BA는 원본 이미지의 1/8 해상도에서 동작한다.
- intrinsics도 feature 해상도에 맞춰 `/ 8.0`으로 저장된다.
- pose tensor는 `[tx, ty, tz, qx, qy, qz, qw]` 형태로 보인다. `DepthVideo` identity 초기값이 `[0, 0, 0, 0, 0, 0, 1]`이다.
- monocular는 scale ambiguity가 있으므로 backend에서 `video.normalize()`를 수행한다.
- `Droid.terminate()`와 `DroidAsync.terminate()`는 `camera_trajectory.inv()`를 반환한다.

## 18. 확인된 위험/개선 포인트

1. IMU 실험 스크립트와 `demo.py` 인자 정의가 맞지 않는다. `run_webp_imu_selected_experiment.sh`를 쓰려면 `demo.py`에 IMU args를 추가하고 `ImuRotationResidual.apply()` 호출 위치를 정해야 한다.
2. `tools/imu_preintegrate.py`는 CSV row 생성/저장 부분이 placeholder로 남아 있다.
3. `data_readers/factory.py`는 `ETH3DStream`, `TUMStream`, `EurocStream`, `KITTIStream`을 참조하지만 현재 파일 상단에서 import하지 않는다. 해당 factory path를 실제로 쓰면 `NameError`가 날 수 있다.
4. 일부 평가 코드에서 `np.float`, `np.unicode_` 같은 deprecated NumPy alias가 보인다. 최신 NumPy 환경에서는 호환성 문제가 생길 수 있다.
5. `DroidAsync.__init__()`에서 `load_network(args.weights)`를 한 번 호출한 뒤 다시 frontend device로 재호출한다. 첫 호출은 불필요해 보인다.
6. 평가 스크립트가 evo의 `orientations_quat_wxyz`에 `traj_est[:, 3:]`를 그대로 넘긴다. 내부 pose가 `[qx, qy, qz, qw]`라면 `[qw, qx, qy, qz]` 재정렬이 필요할 수 있으므로 실제 metric 검증 시 확인이 필요하다.

## 19. 전체 아키텍처 관점 결론

이 코드베이스는 “학습된 correspondence update + 전통적 bundle adjustment”를 결합한 구조다. 네트워크는 depth/pose를 직접 end-to-end로 출력하기보다, factor graph의 reprojection residual을 어떻게 보정할지 `delta/weight/damping`을 예측한다. 그리고 실제 geometric consistency는 `DepthVideo.ba()`와 CUDA backend가 보장한다.

따라서 핵심 이해 포인트는 세 가지다.

1. `DepthVideo`가 모든 SLAM 상태의 단일 source of truth다.
2. `FactorGraph.update()`가 neural update와 geometric BA를 연결하는 중심 루프다.
3. `MotionFilter`, `Frontend`, `Backend`, `TrajectoryFiller`는 각각 frame 선별, online local 최적화, global refinement, skipped frame 보정이라는 서로 다른 시간 범위의 문제를 맡는다.

IMU 관련 파일은 실험 도구로 추가되어 있으나, 현재 메인 추론 루프에는 완전히 연결되어 있지 않다. IMU-DROID 통합을 완성하려면 argparse, prior 로딩, residual 적용 시점, reconstruction/debug 저장 흐름을 하나의 실행 path로 묶는 작업이 다음 단계다.

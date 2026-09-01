import sys

# droid_slam 내부 모듈을 import하기 위해 경로 추가
sys.path.append('droid_slam')

import cv2
import numpy as np
from collections import OrderedDict
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# dataset loader 생성 함수
from data_readers.factory import dataset_factory

# Lie group pose 표현
# SO3: rotation group
# SE3: rigid transformation, rotation + translation
# Sim3: similarity transformation, SE3 + scale
from lietorch import SO3, SE3, Sim3

# loss 함수 모듈
from geom import losses
from geom.losses import geodesic_loss, residual_loss, flow_loss

# 학습용 frame graph 생성 함수
from geom.graph_utils import build_frame_graph

# DROID neural network
from droid_net import DroidNet

# 학습 log 저장/출력용 logger
from logger import Logger

# DDP training 관련 모듈
# 여러 GPU를 사용해 분산 학습하기 위한 PyTorch 기능
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(gpu, args):
    """
    Distributed Data Parallel 학습 환경 초기화 함수.

    gpu:
        현재 process가 사용할 GPU 번호.

    args.world_size:
        전체 GPU/process 수.

    역할:
        1. NCCL backend로 distributed process group 초기화
        2. random seed 설정
        3. 현재 process가 사용할 CUDA device 지정
    """

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=gpu
    )

    # 재현성을 위한 torch seed 고정
    torch.manual_seed(0)

    # 현재 process가 사용할 GPU 설정
    torch.cuda.set_device(gpu)


def show_image(image):
    """
    디버깅용 이미지 출력 함수.

    image:
        [3, H, W] 형태의 torch tensor.

    처리:
        [3, H, W] -> [H, W, 3]으로 바꾼 뒤 OpenCV로 표시.
    """

    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()


def build_temporal_graph(num_frames, radius=2):
    """
    temporal graph 생성 함수.

    기존 build_frame_graph는 pose/depth 기반으로 frame 간 covisibility나 flow distance를 보고
    edge를 만들 수 있다.

    하지만 dense depth가 없는 dataset 또는 IMU 중심 학습에서는
    단순히 시간적으로 가까운 frame끼리 edge를 연결하는 graph가 필요할 수 있다.

    num_frames:
        한 training sample 안의 frame 수.

    radius:
        frame index 차이가 radius 이하인 frame끼리 edge 연결.

    예:
        num_frames=7, radius=2이면
        0 -> 1,2
        1 -> 0,2,3
        2 -> 0,1,3,4
        ...
    """

    graph = OrderedDict()

    for i in range(num_frames):
        graph[i] = [
            j for j in range(num_frames)
            if i != j and abs(i - j) <= radius
        ]

    return graph


def freeze_non_imu_parameters(model, use_imu_data, use_imu_ba, use_acc_bias):
    """
    기존 DROID visual/update network 파라미터를 freeze하고,
    IMU 관련 파라미터만 학습 가능하게 만드는 함수.

    이 함수는 DVI-SLAM의 3단계 학습 방식과 유사한 의도로 볼 수 있다.

    DVI-SLAM:
        기존 visual branch를 먼저 학습한 뒤,
        후반 단계에서 IMU confidence branch를 따로 학습.

    이 코드:
        pretrained DROID weight를 불러온 뒤,
        DROID 본체는 freeze하고 IMU bias / IMU confidence / IMU BA weight만 학습 가능하게 함.

    use_imu_data:
        gyro bias 학습 여부와 관련.

    use_imu_ba:
        IMU BA weight 및 update.imu_confidence branch 학습 여부.

    use_acc_bias:
        accelerometer bias 학습 여부.
    """

    # 일단 전체 파라미터 freeze
    for p in model.parameters():
        p.requires_grad_(False)

    # gyro bias는 IMU data를 사용할 때만 학습
    if use_imu_data:
        model.imu_gyro_bias.requires_grad_(True)

    # accelerometer bias는 full IMU BA/loss를 사용할 때만 학습
    if use_acc_bias:
        model.imu_acc_bias.requires_grad_(True)

    # IMU BA를 사용하는 경우:
    #   1. global IMU BA weight
    #   2. update module 내부 IMU confidence branch
    # 를 학습 가능하게 설정
    if use_imu_ba:
        model.imu_ba_log_weight.requires_grad_(True)

        for p in model.update.imu_confidence.parameters():
            p.requires_grad_(True)


def count_trainable_parameters(model):
    """
    현재 학습 가능한 parameter 수를 세는 함수.

    freeze_non_imu 옵션을 썼을 때 실제로 IMU 관련 파라미터만
    학습되는지 확인하는 용도.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, path):
    """
    checkpoint 저장 함수.

    DDP로 감싸진 model은 실제 원본 model이 model.module 안에 있으므로,
    DDP 여부를 확인한 뒤 state_dict를 저장한다.

    이렇게 저장하면 나중에 DDP 없이도 load하기 쉬움.
    """

    module = model.module if hasattr(model, "module") else model
    torch.save(module.state_dict(), path)


def clamp_parameter_norm(param, max_norm):
    """
    bias parameter의 norm을 제한하는 함수.

    IMU bias는 학습 중 값이 과도하게 커질 수 있다.
    특히 gyro/acc bias를 end-to-end로 학습하면 비정상적으로 큰 bias로 loss를 줄이려는
    방향이 생길 수 있으므로 norm clamp를 제공한다.

    max_norm <= 0이면 clamp를 하지 않는다.
    """

    max_norm = float(max_norm)

    if max_norm <= 0.0:
        return

    with torch.no_grad():
        norm = param.norm()

        if norm > max_norm:
            param.mul_(max_norm / norm.clamp_min(1e-12))


def train(gpu, args):
    """
    DROID-SLAM 학습 함수.

    기존 DROID train.py와 비교했을 때 추가된 핵심:
        1. IMU prior를 dataset에서 함께 로드할 수 있음.
        2. DroidNet에 IMU BA 관련 인자를 전달할 수 있음.
        3. IMU rotation loss 또는 full IMU preintegration loss를 추가할 수 있음.
        4. gyro bias / accelerometer bias를 학습할 수 있음.
        5. pretrained DROID를 freeze하고 IMU branch만 학습할 수 있음.
    """

    ############################################################
    # 1. DDP 초기화
    ############################################################

    setup_ddp(gpu, args)

    # random restart를 위한 numpy RNG
    rng = np.random.default_rng(12345)

    # 한 sample에서 사용할 frame 수
    N = args.n_frames

    ############################################################
    # 2. 모델 생성
    ############################################################

    model = DroidNet()
    model.cuda()
    model.train()

    ############################################################
    # 3. checkpoint 로드
    ############################################################

    if args.ckpt is not None:
        # DDP checkpoint에서 "module." prefix가 붙어 있을 수 있으므로 제거
        state_dict = OrderedDict([
            (k.replace("module.", ""), v)
            for (k, v) in torch.load(args.ckpt).items()
        ])

        # strict=False:
        #   기존 DROID checkpoint에는 새로 추가한 IMU parameter가 없을 수 있음.
        #   예: imu_gyro_bias, imu_acc_bias, imu_ba_log_weight, update.imu_confidence 등.
        #
        #   그래서 strict=False로 로드해야 기존 weight는 가져오고,
        #   새 IMU parameter는 새로 초기화된 상태로 둘 수 있다.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if gpu == 0 and (missing or unexpected):
            print(f"[CKPT] missing={missing}, unexpected={unexpected}")

    ############################################################
    # 4. IMU bias 초기값 설정
    ############################################################

    # gyro bias 초기값을 command line에서 직접 주는 경우
    if args.imu_bias_init is not None:
        with torch.no_grad():
            model.imu_gyro_bias.copy_(
                torch.as_tensor(
                    args.imu_bias_init,
                    device="cuda",
                    dtype=model.imu_gyro_bias.dtype
                )
            )

    # accelerometer bias 초기값을 command line에서 직접 주는 경우
    if args.imu_acc_bias_init is not None:
        with torch.no_grad():
            model.imu_acc_bias.copy_(
                torch.as_tensor(
                    args.imu_acc_bias_init,
                    device="cuda",
                    dtype=model.imu_acc_bias.dtype
                )
            )

    ############################################################
    # 5. IMU 사용 모드 결정
    ############################################################

    # full IMU loss 사용 여부
    #
    # use_full_imu_loss:
    #   loss 함수에서 rotation뿐 아니라 position, velocity, bias까지 포함한
    #   full preintegration residual을 사용.
    #
    # use_full_imu_ba and use_imu_loss:
    #   BA 내부에도 full IMU residual을 쓰고, loss도 IMU를 쓰는 경우.
    use_full_imu_loss = args.use_full_imu_loss or (
        args.use_full_imu_ba and args.use_imu_loss
    )

    # IMU 데이터를 dataset에서 로드해야 하는지 여부
    #
    # IMU loss를 쓰거나, IMU BA를 쓰거나, full IMU loss를 쓰면
    # dataset에서 imu_delta, imu_valid가 필요하다.
    use_imu_data = args.use_imu_loss or args.use_imu_ba or use_full_imu_loss

    # accelerometer bias까지 필요한지 여부
    #
    # rotation-only IMU loss는 gyro bias만으로도 가능하지만,
    # full preintegration은 acc bias도 필요하다.
    use_acc_bias = args.use_full_imu_ba or use_full_imu_loss

    # IMU BA를 사용할 때 global IMU BA weight 초기값 설정
    if args.use_imu_ba and not args.eval_only:
        model.set_imu_ba_weight(args.imu_ba_weight)

    ############################################################
    # 6. 사용하지 않는 IMU parameter freeze
    ############################################################

    # IMU 데이터를 사용하지 않으면 gyro/acc bias 학습 비활성화
    if not use_imu_data:
        model.imu_gyro_bias.requires_grad_(False)
        model.imu_acc_bias.requires_grad_(False)

    # acc bias가 필요 없는 모드이면 acc bias freeze
    if not use_acc_bias:
        model.imu_acc_bias.requires_grad_(False)

    # IMU BA를 사용하지 않으면:
    #   - IMU BA weight freeze
    #   - update network 안의 IMU confidence branch freeze
    if not args.use_imu_ba:
        model.imu_ba_log_weight.requires_grad_(False)

        for p in model.update.imu_confidence.parameters():
            p.requires_grad_(False)

    ############################################################
    # 7. 기존 DROID 파라미터 freeze 옵션
    ############################################################

    if args.freeze_non_imu:
        freeze_non_imu_parameters(
            model,
            use_imu_data=use_imu_data,
            use_imu_ba=args.use_imu_ba,
            use_acc_bias=use_acc_bias,
        )

        if gpu == 0:
            if args.ckpt is None:
                print(
                    "[WARN] --freeze_non_imu is set without --ckpt; "
                    "IMU heads train on random DROID features."
                )

            print(
                f"[TRAIN] freeze_non_imu=True, "
                f"trainable_parameters={count_trainable_parameters(model)}"
            )

    ############################################################
    # 8. eval only 모드
    ############################################################

    if args.eval_only:
        model.eval()

    ############################################################
    # 9. Dataset 생성
    ############################################################

    # args.datasets가 있으면 해당 dataset 사용,
    # 없으면 기본 tartan 사용
    dataset_names = args.datasets if args.datasets is not None else ["tartan"]

    db = dataset_factory(
        dataset_names,
        datapath=args.datapath,
        n_frames=args.n_frames,
        fmin=args.fmin,
        fmax=args.fmax,

        # 추가된 IMU 관련 dataset 옵션
        use_imu=use_imu_data,
        imu_prior_name=args.imu_prior_name,
        imu_require=args.imu_require,
    )

    # dataset에 dense depth가 있는지 확인
    #
    # TartanAir처럼 dense depth/disparity가 있으면 flow_loss 사용 가능.
    # 실제 수집 데이터처럼 dense depth가 없으면 flow_loss 계산이 어려움.
    has_dense_depth = getattr(db, "has_dense_depth", True)

    # flow loss를 학습할지 여부
    train_flow_loss = (
        has_dense_depth
        and not args.disable_flow_loss
        and args.w3 != 0.0
    )

    if gpu == 0 and not has_dense_depth:
        print("[DATA] dense depth is unavailable; using temporal graph and disabling flow_loss")

    # flow loss를 학습하지 않는 경우 upmask branch를 freeze
    #
    # upmask는 disparity upsampling에 사용되며 flow_loss/disp supervision과 관련이 깊다.
    # dense depth가 없으면 이 branch를 굳이 학습시키지 않는다.
    if not train_flow_loss:
        for p in model.update.agg.upmask.parameters():
            p.requires_grad_(False)

    ############################################################
    # 10. DDP wrapping
    ############################################################

    model = DDP(
        model,
        device_ids=[gpu],
        find_unused_parameters=False
    )

    ############################################################
    # 11. Distributed DataLoader 생성
    ############################################################

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        db,
        shuffle=True,
        num_replicas=args.world_size,
        rank=gpu
    )

    train_loader = DataLoader(
        db,
        batch_size=args.batch,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )

    ############################################################
    # 12. Optimizer / Scheduler 생성
    ############################################################

    if args.eval_only:
        optimizer = None
        scheduler = None

    else:
        # requires_grad=True인 parameter만 optimizer에 넣는다.
        #
        # freeze_non_imu=True이면 여기에는 IMU 관련 parameter만 들어간다.
        trainable_params = [
            p for p in model.parameters()
            if p.requires_grad
        ]

        if len(trainable_params) == 0:
            raise ValueError(
                "no trainable parameters; check --freeze_non_imu / IMU flags"
            )

        optimizer = torch.optim.Adam(
            trainable_params,
            lr=args.lr,
            weight_decay=1e-5
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            args.lr,
            args.steps,
            pct_start=0.01,
            cycle_momentum=False
        )

    ############################################################
    # 13. Logger 및 training loop 준비
    ############################################################

    logger = Logger(args.name, scheduler, sum_freq=args.log_freq)

    should_keep_training = True
    total_steps = 0

    ############################################################
    # 14. Main training loop
    ############################################################

    while should_keep_training:
        for i_batch, item in enumerate(train_loader):

            ####################################################
            # 14-1. gradient 초기화
            ####################################################

            if optimizer is not None:
                optimizer.zero_grad()

            ####################################################
            # 14-2. batch item CUDA 이동
            ####################################################

            # item은 dataset에 따라 길이가 달라질 수 있다.
            #
            # 기본:
            #   images, poses, disps, intrinsics
            #
            # IMU 사용 시:
            #   images, poses, disps, intrinsics, imu_delta, imu_valid
            item = [
                x.to("cuda") if torch.is_tensor(x) else x
                for x in item
            ]

            images, poses, disps, intrinsics = item[:4]

            # IMU preintegration delta
            # 예: frame 간 ΔR, Δp, Δv, dt 등으로 구성될 수 있음.
            imu_delta = item[4] if len(item) > 4 else None

            # IMU edge가 유효한지 나타내는 mask
            imu_valid = item[5] if len(item) > 5 else None

            ####################################################
            # 14-3. pose convention 변환
            ####################################################

            # dataset pose가 world-to-camera라면,
            # DROID 내부 학습에서는 camera-to-world 형태로 쓰기 위해 inverse.
            Ps = SE3(poses).inv()

            # 초기 pose estimate는 identity 형태로 생성
            Gs = SE3.IdentityLike(Ps)

            ####################################################
            # 14-4. frame graph 생성
            ####################################################

            if not has_dense_depth:
                # dense depth가 없으면 build_frame_graph를 쓰기 어렵기 때문에
                # 시간적으로 가까운 frame끼리만 연결
                graph = build_temporal_graph(N)

            elif np.random.rand() < 0.5:
                # pose/depth/intrinsics 기반으로 학습용 frame graph 생성
                graph = build_frame_graph(
                    poses,
                    disps,
                    intrinsics,
                    num=args.edges
                )

            else:
                # 절반 확률로 단순 temporal graph 사용
                graph = build_temporal_graph(N)

            ####################################################
            # 14-5. 초기 pose / disparity 설정
            ####################################################

            # 첫 번째 pose는 GT pose로 고정
            Gs.data[:, 0] = Ps.data[:, 0].clone()

            # 나머지 frame pose는 두 번째 GT pose로 초기화
            # 기존 DROID 학습 방식과 동일한 구조
            Gs.data[:, 1:] = Ps.data[:, [1]].clone()

            # 초기 disparity는 1로 설정
            # DROID는 1/8 해상도 disparity를 최적화하므로
            # disps[:,:,3::8,3::8]와 같은 shape의 tensor를 사용
            disp0 = torch.ones_like(disps[:, :, 3::8, 3::8])

            ####################################################
            # 14-6. Random restart loop
            ####################################################

            # restart_prob 확률 구조:
            #   최소 1번은 실행되고,
            #   r < restart_prob이면 추가 restart가 이어질 수 있음.
            r = 0

            while r < args.restart_prob:
                r = rng.random()

                # projection은 1/8 feature resolution에서 수행되므로 intrinsics도 /8
                intrinsics0 = intrinsics / 8.0

                # full IMU BA와 full IMU loss를 모두 사용하는 경우
                # model forward에서 imu_motions를 반환받음
                return_imu_motion = args.use_full_imu_ba and use_full_imu_loss

                ################################################
                # 14-6-1. DroidNet forward
                ################################################

                with torch.set_grad_enabled(not args.eval_only):
                    model_output = model(
                        Gs,
                        images,
                        disp0,
                        intrinsics0,
                        graph,
                        num_steps=args.iters,
                        fixedp=2,

                        # IMU bias 반환 여부
                        return_imu_bias=use_imu_data,

                        # IMU BA를 사용할 때만 forward에 imu_delta/imu_valid 전달
                        imu_delta=imu_delta if args.use_imu_ba else None,
                        imu_valid=imu_valid if args.use_imu_ba else None,

                        # None이면 model 내부 learnable weight 사용 가능
                        # use_imu_ba가 아니면 0으로 꺼둠
                        imu_ba_weight=None if args.use_imu_ba else 0.0,

                        # IMU residual clipping / confidence 관련 설정
                        imu_ba_max_residual=args.imu_max_residual,
                        imu_confidence_floor=args.imu_conf_floor,
                        return_imu_confidence=args.use_imu_ba,

                        # full IMU BA 관련 설정
                        use_full_imu_ba=args.use_full_imu_ba,
                        imu_full_pos_weight=args.imu_full_pos_weight,
                        imu_full_vel_weight=args.imu_full_vel_weight,
                        imu_full_bias_weight=args.imu_full_bias_weight,
                        imu_velocity_init=args.imu_velocity_init,
                        imu_motion_prior_weight=args.imu_motion_prior_weight,
                        imu_local_bias_prior_weight=args.imu_local_bias_prior_weight,
                        imu_gravity=args.imu_gravity,
                        return_imu_motion=return_imu_motion,
                    )

                ################################################
                # 14-6-2. model output unpacking
                ################################################

                imu_confidences = []
                imu_motions = []

                if args.use_imu_ba:
                    if return_imu_motion:
                        (
                            poses_est,
                            disps_est,
                            residuals,
                            gyro_bias,
                            imu_confidences,
                            imu_motions,
                        ) = model_output
                    else:
                        (
                            poses_est,
                            disps_est,
                            residuals,
                            gyro_bias,
                            imu_confidences,
                        ) = model_output

                elif use_imu_data:
                    if return_imu_motion:
                        (
                            poses_est,
                            disps_est,
                            residuals,
                            gyro_bias,
                            imu_motions,
                        ) = model_output
                    else:
                        (
                            poses_est,
                            disps_est,
                            residuals,
                            gyro_bias,
                        ) = model_output

                else:
                    poses_est, disps_est, residuals = model_output
                    gyro_bias = None

                ################################################
                # 14-6-3. 기존 DROID visual loss 계산
                ################################################

                # pose geodesic loss
                geo_loss, geo_metrics = losses.geodesic_loss(
                    Ps,
                    poses_est,
                    graph,
                    do_scale=False
                )

                # reprojection residual loss
                res_loss, res_metrics = losses.residual_loss(
                    residuals
                )

                # flow loss
                # dense depth가 있을 때만 계산
                if train_flow_loss:
                    flo_loss, flo_metrics = losses.flow_loss(
                        Ps,
                        disps,
                        poses_est,
                        disps_est,
                        intrinsics,
                        graph
                    )
                else:
                    flo_loss = torch.zeros((), device=images.device)
                    flo_metrics = {
                        "f_error": 0.0,
                        "1px": 0.0
                    }

                ################################################
                # 14-6-4. IMU loss 계산
                ################################################

                if args.use_imu_loss and use_full_imu_loss:
                    # full IMU preintegration loss
                    #
                    # rotation뿐 아니라 position, velocity, bias residual까지 포함.
                    # DVI-SLAM의 정식 IMU factor에 가까운 방향.
                    imu_loss, imu_metrics = losses.imu_full_preintegration_loss(
                        poses_est,
                        imu_delta,
                        imu_valid=imu_valid,
                        imu_motions=imu_motions if len(imu_motions) > 0 else None,
                        gyro_bias=gyro_bias,
                        accel_bias=model.module.imu_acc_bias,
                        gamma=args.imu_gamma,
                        max_residual=args.imu_max_residual,
                        smooth_beta=args.imu_smooth_beta,
                        pos_weight=args.imu_loss_pos_weight,
                        vel_weight=args.imu_loss_vel_weight,
                        rot_weight=args.imu_loss_rot_weight,
                        bias_weight=args.imu_loss_bias_weight,
                        gravity=args.imu_gravity,
                    )

                elif args.use_imu_loss:
                    # rotation-only IMU loss
                    #
                    # gyro preintegration 기반으로 frame 간 relative rotation이
                    # pose_est의 relative rotation과 일치하도록 학습.
                    #
                    # full 상태 v, ba, bg를 모두 다루기 전의 가벼운 IMU loss.
                    imu_loss, imu_metrics = losses.imu_rotation_bias_loss(
                        poses_est,
                        imu_delta,
                        imu_valid=imu_valid,
                        gyro_bias=gyro_bias,
                        gamma=args.imu_gamma,
                        max_residual=args.imu_max_residual,
                        smooth_beta=args.imu_smooth_beta,
                    )

                else:
                    # IMU loss를 쓰지 않는 경우 0으로 둠
                    imu_loss = torch.zeros((), device=images.device)

                    imu_metrics = {
                        "imu_full_loss": 0.0,
                        "imu_pos_loss": 0.0,
                        "imu_vel_loss": 0.0,
                        "imu_rot_loss": 0.0,
                        "imu_bias_loss": 0.0,
                        "imu_pos_error": 0.0,
                        "imu_vel_error": 0.0,
                        "imu_rot_error": 0.0,
                        "imu_bias_error": 0.0,
                        "imu_edges": 0.0,
                    }

                ################################################
                # 14-6-5. bias regularization loss
                ################################################

                bias_loss = torch.zeros((), device=images.device)

                # gyro bias norm regularization
                if gyro_bias is not None and args.w_imu_bias != 0.0:
                    bias_loss = bias_loss + gyro_bias.norm()

                # acc bias norm regularization
                if use_acc_bias and args.w_imu_bias != 0.0:
                    bias_loss = bias_loss + model.module.imu_acc_bias.norm()

                ################################################
                # 14-6-6. total loss
                ################################################

                loss = (
                    args.w1 * geo_loss
                    + args.w2 * res_loss
                    + args.w3 * flo_loss
                    + args.w_imu * imu_loss
                    + args.w_imu_bias * bias_loss
                )

                if not args.eval_only:
                    loss.backward()

                ################################################
                # 14-6-7. random restart용 초기값 갱신
                ################################################

                # 마지막 iteration pose를 다음 restart의 초기 pose로 사용
                Gs = poses_est[-1].detach()

                # 마지막 disparity를 1/8 해상도 초기값으로 사용
                disp0 = disps_est[-1][:, :, 3::8, 3::8].detach()

            ####################################################
            # 14-7. metrics 정리
            ####################################################

            metrics = {}
            metrics.update(geo_metrics)
            metrics.update(res_metrics)
            metrics.update(flo_metrics)
            metrics.update(imu_metrics)

            ####################################################
            # 14-8. IMU bias metric 기록
            ####################################################

            if use_imu_data:
                bias = model.module.imu_gyro_bias.detach()

                metrics["imu_bias_norm"] = bias.norm().item()
                metrics["imu_bias_x"] = bias[0].item()
                metrics["imu_bias_y"] = bias[1].item()
                metrics["imu_bias_z"] = bias[2].item()

                if use_acc_bias:
                    acc_bias = model.module.imu_acc_bias.detach()

                    metrics["imu_acc_bias_norm"] = acc_bias.norm().item()
                    metrics["imu_acc_bias_x"] = acc_bias[0].item()
                    metrics["imu_acc_bias_y"] = acc_bias[1].item()
                    metrics["imu_acc_bias_z"] = acc_bias[2].item()

            ####################################################
            # 14-9. IMU confidence metric 기록
            ####################################################

            if args.use_imu_ba and len(imu_confidences) > 0:
                conf = imu_confidences[-1].detach()

                metrics["imu_conf_mean"] = conf.mean().item()
                metrics["imu_conf_min"] = conf.min().item()
                metrics["imu_conf_max"] = conf.max().item()

                # learnable global IMU BA weight
                metrics["imu_ba_weight"] = (
                    model.module.get_imu_ba_weight().detach().item()
                )

            ####################################################
            # 14-10. optimizer update
            ####################################################

            if not args.eval_only:
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.clip
                )

                optimizer.step()

                # 학습 후 gyro/acc bias norm clamp
                clamp_parameter_norm(
                    model.module.imu_gyro_bias,
                    args.imu_gyro_bias_max_norm
                )

                clamp_parameter_norm(
                    model.module.imu_acc_bias,
                    args.imu_acc_bias_max_norm
                )

                scheduler.step()

            ####################################################
            # 14-11. step 증가 및 logging
            ####################################################

            total_steps += 1

            if gpu == 0:
                logger.push(metrics)

            ####################################################
            # 14-12. checkpoint 저장
            ####################################################

            if (
                not args.eval_only
                and args.save_freq > 0
                and total_steps % args.save_freq == 0
                and gpu == 0
            ):
                PATH = 'checkpoints/%s_%06d.pth' % (
                    args.name,
                    total_steps
                )

                save_checkpoint(model, PATH)
                print(f"[CKPT] saved {PATH}")

            ####################################################
            # 14-13. 종료 조건
            ####################################################

            if total_steps >= args.steps:
                should_keep_training = False
                break

    ############################################################
    # 15. final checkpoint 저장
    ############################################################

    if not args.eval_only and args.save_final and gpu == 0:
        PATH = 'checkpoints/%s_final_%06d.pth' % (
            args.name,
            total_steps
        )

        save_checkpoint(model, PATH)
        print(f"[CKPT] saved {PATH}")

    ############################################################
    # 16. DDP 종료
    ############################################################

    dist.destroy_process_group()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    ############################################################
    # 1. 기본 실험 설정
    ############################################################

    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--datasets', nargs='+', help='lists of datasets for training')
    parser.add_argument('--datapath', default='datasets/TartanAir', help="path to dataset directory")
    parser.add_argument('--gpus', type=int, default=4)

    ############################################################
    # 2. 기본 학습 hyperparameter
    ############################################################

    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--iters', type=int, default=15)
    parser.add_argument('--steps', type=int, default=250000)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--n_frames', type=int, default=7)

    ############################################################
    # 3. 기존 DROID loss weight
    ############################################################

    # pose geodesic loss weight
    parser.add_argument('--w1', type=float, default=10.0)

    # residual loss weight
    parser.add_argument('--w2', type=float, default=0.01)

    # flow loss weight
    parser.add_argument('--w3', type=float, default=0.05)

    ############################################################
    # 4. 추가된 IMU loss weight
    ############################################################

    # IMU preintegration loss weight
    parser.add_argument('--w_imu', type=float, default=0.05)

    # gyro/acc bias regularization weight
    parser.add_argument('--w_imu_bias', type=float, default=1e-4)

    ############################################################
    # 5. dataset sampling / graph 설정
    ############################################################

    parser.add_argument('--fmin', type=float, default=8.0)
    parser.add_argument('--fmax', type=float, default=96.0)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--edges', type=int, default=24)
    parser.add_argument('--restart_prob', type=float, default=0.2)

    ############################################################
    # 6. freeze / logging / checkpoint 옵션
    ############################################################

    parser.add_argument(
        '--freeze_non_imu',
        action='store_true',
        help=(
            'freeze pretrained DROID visual/update weights and train only '
            'IMU confidence, IMU BA scale, and IMU biases'
        ),
    )

    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--save_final', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--disable_flow_loss', action='store_true')

    ############################################################
    # 7. IMU BA 관련 옵션
    ############################################################

    # IMU residual을 BA 단계에 넣을지 여부
    parser.add_argument('--use_imu_ba', action='store_true')

    parser.add_argument(
        '--use_full_imu_ba',
        action='store_true',
        help='extend training BA with pose+velocity+accel-bias+gyro-bias IMU residuals',
    )

    # IMU BA residual의 global scale 초기값
    parser.add_argument(
        '--imu_ba_weight',
        type=float,
        default=0.05,
        help='initial value for learnable global IMU BA information scale',
    )

    # learned IMU confidence에 최소값을 섞어줄 때 사용
    parser.add_argument(
        '--imu_conf_floor',
        type=float,
        default=0.0,
        help='minimum value mixed into learned GRU IMU confidence',
    )

    ############################################################
    # 8. IMU loss 관련 옵션
    ############################################################

    # IMU loss를 학습 loss에 추가할지 여부
    parser.add_argument('--use_imu_loss', action='store_true')

    # dataset 안에서 IMU prior 파일 이름
    parser.add_argument('--imu_prior_name', type=str, default='imu_prior.csv')

    # IMU prior가 없는 sample을 허용하지 않을지 여부
    parser.add_argument('--imu_require', action='store_true')

    # iteration별 loss weight decay/gamma
    parser.add_argument('--imu_gamma', type=float, default=0.9)

    # IMU residual clipping 값
    parser.add_argument('--imu_max_residual', type=float, default=0.5)

    # smooth L1 loss beta
    parser.add_argument('--imu_smooth_beta', type=float, default=0.05)

    ############################################################
    # 9. IMU bias 초기값
    ############################################################

    # gyro bias 초기값
    parser.add_argument('--imu_bias_init', type=float, nargs=3, default=None)

    # accelerometer bias 초기값
    parser.add_argument('--imu_acc_bias_init', type=float, nargs=3, default=None)

    ############################################################
    # 10. full IMU BA 내부 weight
    ############################################################

    parser.add_argument('--imu_full_pos_weight', type=float, default=0.05)
    parser.add_argument('--imu_full_vel_weight', type=float, default=0.05)
    parser.add_argument('--imu_full_bias_weight', type=float, default=0.001)
    parser.add_argument(
        '--imu_gravity',
        type=float,
        nargs=3,
        default=None,
        metavar=('GX', 'GY', 'GZ'),
        help='optional gravity vector in training pose units for full IMU residuals',
    )

    # velocity 초기화 방식
    #
    # zero:
    #   velocity를 0으로 초기화
    #
    # pose:
    #   pose 차분으로 velocity를 초기화
    parser.add_argument('--imu_velocity_init', choices=['zero', 'pose'], default='pose')

    # IMU motion state에 대한 prior weight
    parser.add_argument('--imu_motion_prior_weight', type=float, default=0.0)

    # local bias smoothness/prior weight
    parser.add_argument('--imu_local_bias_prior_weight', type=float, default=0.0)

    ############################################################
    # 11. full IMU loss 관련 옵션
    ############################################################

    parser.add_argument(
        '--use_full_imu_loss',
        action='store_true',
        help='train IMU loss with position, velocity, rotation, and bias preintegration residuals',
    )

    parser.add_argument('--imu_loss_pos_weight', type=float, default=0.05)
    parser.add_argument('--imu_loss_vel_weight', type=float, default=0.05)
    parser.add_argument('--imu_loss_rot_weight', type=float, default=1.0)
    parser.add_argument('--imu_loss_bias_weight', type=float, default=0.001)

    ############################################################
    # 12. IMU bias norm clamp
    ############################################################

    parser.add_argument('--imu_gyro_bias_max_norm', type=float, default=0.0)
    parser.add_argument('--imu_acc_bias_max_norm', type=float, default=0.0)

    ############################################################
    # 13. argument parsing 및 옵션 보정
    ############################################################

    args = parser.parse_args()

    # full IMU loss를 켜면 일반 IMU loss flag도 자동으로 켬
    if args.use_full_imu_loss:
        args.use_imu_loss = True

    # full IMU BA를 켜면 일반 IMU BA flag도 자동으로 켬
    if args.use_full_imu_ba:
        args.use_imu_ba = True

    # DDP 전체 process 수
    args.world_size = args.gpus

    print(args)

    ############################################################
    # 14. checkpoint directory 생성
    ############################################################

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    ############################################################
    # 15. DDP 환경 변수 설정
    ############################################################

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    ############################################################
    # 16. multi-GPU training 시작
    ############################################################

    mp.spawn(
        train,
        nprocs=args.gpus,
        args=(args,)
    )

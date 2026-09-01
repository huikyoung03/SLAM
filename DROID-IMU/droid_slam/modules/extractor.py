import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Residual Block (ResNet Basic Block)
# ============================================================
class ResidualBlock(nn.Module):
    """
    ResNet에서 사용하는 기본 Residual Block

    구조
        Conv3x3
            ↓
        Normalization
            ↓
          ReLU
            ↓
        Conv3x3
            ↓
        Normalization
            ↓
      Skip Connection(+)
            ↓
          ReLU

    입력과 출력을 더하여(Residual Learning)
    깊은 네트워크에서도 Gradient Vanishing을 줄인다.
    """

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        # 첫 번째 3x3 Convolution
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            stride=stride
        )

        # 두 번째 3x3 Convolution
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

        # GroupNorm 그룹 개수
        num_groups = planes // 8

        # ----------------------------------------------------
        # Normalization 선택
        # ----------------------------------------------------
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(
                num_groups=num_groups,
                num_channels=planes
            )

            self.norm2 = nn.GroupNorm(
                num_groups=num_groups,
                num_channels=planes
            )

            if stride != 1:
                self.norm3 = nn.GroupNorm(
                    num_groups=num_groups,
                    num_channels=planes
                )

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)

            if stride != 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)

            if stride != 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()

            if stride != 1:
                self.norm3 = nn.Sequential()

        # ----------------------------------------------------
        # Downsampling
        #
        # stride=2인 경우
        # Feature Map 크기가 줄어들므로
        # Skip Connection도 동일한 크기로 맞춘다.
        # ----------------------------------------------------
        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride
                ),
                self.norm3
            )

    def forward(self, x):

        # Skip Connection 저장
        y = x

        # Conv -> Norm -> ReLU
        y = self.relu(
            self.norm1(
                self.conv1(y)
            )
        )

        # Conv -> Norm -> ReLU
        y = self.relu(
            self.norm2(
                self.conv2(y)
            )
        )

        # 크기가 달라지는 경우 Skip도 Downsample
        if self.downsample is not None:
            x = self.downsample(x)

        # Residual Learning
        return self.relu(x + y)


# ============================================================
# Bottleneck Block
# ============================================================
class BottleneckBlock(nn.Module):
    """
    ResNet Bottleneck Block

    1x1 -> 3x3 -> 1x1

    계산량을 줄이면서 깊은 네트워크를 만들기 위한 구조이다.

    현재 BasicEncoder에서는 사용되지 않는다.
    """

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()

        # 채널 축소
        self.conv1 = nn.Conv2d(
            in_planes,
            planes // 4,
            kernel_size=1
        )

        # 공간 Feature 추출
        self.conv2 = nn.Conv2d(
            planes // 4,
            planes // 4,
            kernel_size=3,
            padding=1,
            stride=stride
        )

        # 채널 복원
        self.conv3 = nn.Conv2d(
            planes // 4,
            planes,
            kernel_size=1
        )

        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        # Normalization 선택
        ...
        # (ResidualBlock과 동일)

    def forward(self, x):

        y = x

        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


# 기본 Channel 수
DIM = 32


# ============================================================
# Basic Encoder
# ============================================================
class BasicEncoder(nn.Module):
    """
    DROID-SLAM의 Feature Encoder

    입력
        RGB Image

    출력
        Dense Feature Map

    입력
        (B,N,3,H,W)

    출력
        (B,N,C,H/8,W/8)

    여기서 추출된 Feature가 이후

        Correlation Volume
        Update Network

    의 입력으로 사용된다.
    """

    def __init__(
            self,
            output_dim=128,
            norm_fn='batch',
            dropout=0.0,
            multidim=False):

        super(BasicEncoder, self).__init__()

        self.norm_fn = norm_fn
        self.multidim = multidim

        # 첫 번째 Normalization
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(
                num_groups=8,
                num_channels=DIM
            )

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(DIM)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # ----------------------------------------------------
        # Stem
        #
        # RGB -> 32 Channel
        #
        # stride=2
        # H,W -> H/2,W/2
        # ----------------------------------------------------
        self.conv1 = nn.Conv2d(
            3,
            DIM,
            kernel_size=7,
            stride=2,
            padding=3
        )

        self.relu1 = nn.ReLU(inplace=True)

        # Residual Stage 1
        self.in_planes = DIM
        self.layer1 = self._make_layer(
            DIM,
            stride=1
        )

        # Residual Stage 2
        self.layer2 = self._make_layer(
            2 * DIM,
            stride=2
        )

        # Residual Stage 3
        self.layer3 = self._make_layer(
            4 * DIM,
            stride=2
        )

        # ----------------------------------------------------
        # 최종 Feature 차원 변환
        # ----------------------------------------------------
        self.conv2 = nn.Conv2d(
            4 * DIM,
            output_dim,
            kernel_size=1
        )

        # Multi-scale Feature 사용 시
        if self.multidim:
            ...

        # Dropout
        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = None

        # Weight Initialization
        for m in self.modules():

            # He Initialization
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )

            # Normalization 초기화
            elif isinstance(
                m,
                (
                    nn.BatchNorm2d,
                    nn.InstanceNorm2d,
                    nn.GroupNorm
                )
            ):

                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        """
        Residual Block 두 개를 하나의 Stage로 생성한다.
        """

        layer1 = ResidualBlock(
            self.in_planes,
            dim,
            self.norm_fn,
            stride=stride
        )

        layer2 = ResidualBlock(
            dim,
            dim,
            self.norm_fn,
            stride=1
        )

        self.in_planes = dim

        return nn.Sequential(layer1, layer2)

    def forward(self, x):
        """
        입력
            (B,N,3,H,W)

        출력
            (B,N,128,H/8,W/8)
        """

        # Batch와 Frame을 합쳐 CNN 처리
        b, n, c1, h1, w1 = x.shape

        x = x.view(
            b * n,
            c1,
            h1,
            w1
        )

        # Stem
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # ResNet Backbone
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Feature Projection
        x = self.conv2(x)

        # 다시 (B,N,C,H,W) 형태로 복원
        _, c2, h2, w2 = x.shape

        return x.view(
            b,
            n,
            c2,
            h2,
            w2
        )
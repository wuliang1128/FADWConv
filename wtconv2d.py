import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import wavelet

'''
原始代码解析：
输入 x (B,C,H,W)
    |
    ├─→ base_conv (标准深度卷积) → 保留
    |
    └─→ Level 1 分解: [LL1, LH1, HL1, HH1] 
              | (各自做深度卷积)
              | 存贮: LL1_conv, High1_conv
              |
              └─→ 仅取 LL1 进入 Level 2 分解: [LL2, LH2, HL2, HH2]
                         | (各自做深度卷积)
                         存贮: LL2_conv, High2_conv
                         |
                         重构开始 (从 Level 2 开始):
                         LL2_conv + 0 → 逆变换 → 上采样到 Level 1 尺度
                                     ↓
                         LL1_conv + (上述结果) → 与 High1_conv 拼接 → 逆变换 → 最终 x_tag
    |
最终输出 = base_conv(x) + x_tag

# 级联机制（关键）：每一轮只取 curr_x_ll（LL）进入下一层循环。这意味着：
# 第1层：分解原图 → 得到 LL1, LH1, HL1, HH1
# 第2层：只分解 LL1 → 得到 LL2, LH2, HL2, HH2
# 因此，高频子带只产生一次，而低频子带会被逐级“递归”分解。这正好对应论文中“参数随层级对数增长”的实现。

'''

'''
改进1  调整高低频信息的比例，提升模型效果

思路1：能否增加一个参数 α，低频信息*α + （1-α）高频信息来调整高频和低频的信息比例
WTConv2d 的初始化部分，它其实已经实现了分路径、可学习的比例调节：
   （1）双路径独立缩放：原始路径（base_conv）权重初始为 1.0，小波路径（wavelet_convs）权重初始仅为 0.1。这相当于在训练初期，让网络先依赖标准卷积，随着训练进行，再慢慢加大小波路径的贡献。
   （2）细粒度（Per-channel）：_ScaleModule 的参数维度是 [1, C, 1, 1]，意味着它为每个通道独立学习一个缩放系数，而不是全局一个标量。这比你的全局 α 灵活得多。

鉴于甲骨文拓片噪声极多（高频干扰），而字形形状（低频）是识别的核心依据。我们希望在最终融合时，给小波路径（富含低频形状信息）更大的初始权重，同时抑制高频噪声的过度放大。
修改 forward 的最终融合部分，引入一个可学习的门控融合参数，而不是修改小波内部
'''



class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1


        self.wt_filter, self.iwt_filter = wavelet.create_2d_wavelet_filter(wt_type, in_channels, in_channels,torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])      # 初始化为 1.0

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]    #小波路径（wavelet_convs）权重初始仅为 0.1。
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None


    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            # 核心逻辑：wavelet_2d_transform 返回的张量 curr_x 形状为 (B, C, 4, H/2, W/2)。维度 dim=2（大小为4）固定对应 4 个子带，索引规则为：
            # [ :, :, 0, :, : ] → 低频子带（LL）（全局轮廓）
            # [ :, :, 1, :, : ] → 水平高频（LH）
            # [ :, :, 2, :, : ] → 垂直高频（HL）
            # [ :, :, 3, :, : ] → 对角高频（HH）
            curr_x = wavelet.wavelet_2d_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :] # 提取低频 LL 全局轮廓）

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4]) #先将 (B, C, 4, H, W) 展平为 (B, C*4, H, W)
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))           #执行分组卷积（groups=C*4），等价于对每个子带独立进行空间卷积。
            curr_x_tag = curr_x_tag.reshape(shape_x)  # 还原为 (B, C, 4, H/2, W/2)  即将卷积后的 4 个子带拆开

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])   # 存卷积后的 LL
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])  # 存卷积后的 3个高频

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):   # 融合逻辑：从最深层（第 wt_levels 层）开始向上重构。
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll        # 关键：将高层（深层）的输出加到当前层  特征叠加：将深层提取的 LL 特征（感受野大）加到浅层提取的 LL 特征（感受野小）上，再一起做逆变换（IWT）

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = wavelet.inverse_2d_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        # base_conv 是常规的深度卷积，负责捕捉原始尺度的特征（作为“短路”连接）。
        # x_tag 是经过小波域处理、具备更大感受野的特征。
        # 两者相加，既保留了标准 CNN 的细节捕捉能力，又融入了 WTConv 的全局低频信息。
        x = self.base_scale(self.base_conv(x))    # 原始输入的直接卷积（保留高频纹理）
        x = x + x_tag                             # 加上小波域处理后的结果

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import wavelet   # 确保你的 wavelet 工具模块存在

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        # 小波滤波器（固定不可训练）
        self.wt_filter, self.iwt_filter = wavelet.create_2d_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # 原始路径（标准深度卷积）
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])  # 初始化为 1.0

        # ---------- 改动点 1：拆分高低频的 ScaleModule ----------
        # 低频（LL）缩放，每个 level 一个，初始化为 1.0（强调形状）
        self.wavelet_scale_LL = nn.ModuleList(
            [_ScaleModule([1, in_channels, 1, 1], init_scale=1.0) for _ in range(self.wt_levels)]
        )
        # 高频（LH+HL+HH）缩放，每个 level 一个，初始化为 0.05（压低噪声）
        self.wavelet_scale_H = nn.ModuleList(
            [_ScaleModule([1, in_channels * 3, 1, 1], init_scale=0.05) for _ in range(self.wt_levels)]
        )

        # 小波域的卷积（仍然作用于所有子带，分组卷积）
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )

        # ---------- 改动点 2（可选）：最终融合的可学习参数 ----------
        # 用于控制 base_conv 和 wavelet 路径的融合比例（sigmoid 约束在 0~1）
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))  # 初始 0.5

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll_in_levels = []   # 存储每一级卷积后的 LL
        x_h_in_levels = []    # 存储每一级卷积后的高频（三个子带合并）
        shapes_in_levels = []

        curr_x_ll = x

        # ---------- 前向分解与卷积 ----------
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            # 若尺寸为奇数，填充以保证可整除
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            # 小波变换：输出 (B, C, 4, H/2, W/2)
            curr_x = wavelet.wavelet_2d_transform(curr_x_ll, self.wt_filter)
            # 展平为 (B, C*4, H/2, W/2) 以便分组卷积
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])

            # 执行小波域卷积（对每个子带独立）
            curr_x_tag = self.wavelet_convs[i](curr_x_tag)

            # ---------- 改动点 1 的应用：拆分 LL 和高频分别缩放 ----------
            # 拆出 LL 部分（前 C 个通道）和高频部分（后 3C 个通道）
            curr_x_ll_tag = curr_x_tag[:, :self.in_channels, :, :]          # (B, C, H, W)
            curr_x_h_tag = curr_x_tag[:, self.in_channels:, :, :]          # (B, 3C, H, W)

            # 分别缩放
            curr_x_ll_tag = self.wavelet_scale_LL[i](curr_x_ll_tag)
            curr_x_h_tag = self.wavelet_scale_H[i](curr_x_h_tag)

            # 合并回 (B, 4C, H, W)
            curr_x_tag = torch.cat([curr_x_ll_tag, curr_x_h_tag], dim=1)

            # 还原为 (B, C, 4, H, W)
            curr_x_tag = curr_x_tag.reshape(shape_x)

            # 存储 LL 和高频（用于后续重构）
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])          # LL
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])        # 3个高频合并

            # 更新 curr_x_ll 为下一级的输入（仅 LL）
            curr_x_ll = curr_x[:, :, 0, :, :]

        # ---------- 逆向重构（逐级叠加） ----------
        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            # 深层 LL 特征加到当前层 LL（跨层级融合）
            curr_x_ll = curr_x_ll + next_x_ll

            # 拼接 LL 和三个高频，准备逆变换
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)  # (B, C, 4, H, W)
            next_x_ll = wavelet.inverse_2d_wavelet_transform(curr_x, self.iwt_filter)

            # 裁剪回原始填充前的尺寸
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        # ---------- 原始路径 ----------
        x_base = self.base_scale(self.base_conv(x))

        # ---------- 改动点 2：可学习的最终融合 ----------
        # 使用 sigmoid 将 fusion_alpha 约束在 (0,1)，然后加权求和
        alpha = torch.sigmoid(self.fusion_alpha)
        x = alpha * x_base + (1 - alpha) * x_tag

        # 下采样（如果 stride > 1）
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


# 辅助缩放模块（与原代码一致）
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)
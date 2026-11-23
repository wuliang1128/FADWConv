#FA-WTConv v2：双分支频率感知小波卷积

import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import wavelet

# 辅助缩放模块（与原代码一致）
class _ScaleModule(nn.Module):
    """逐通道可学习缩放模块"""
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return self.weight * x


class FA_WTConv2dv2(nn.Module):
    """
    Frequency-Aware Wavelet Convolution v2 (双分支设计)

    相比 v1 的关键改进：
      1. 并行双分支（卷积 + 小波）而非单向替换 → 保留空间定位
      2. 空间-通道双重门控替代全局标量 α → 逐像素/逐通道自适应
      3. 高频分支初始权重降至 0.05 → 训练初期抑制噪声
      4. 残差连接 → 稳定训练、便于梯度回传
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 bias=True, wt_levels=1, wt_type='db1', reduction=8):
        super().__init__()
        assert in_channels == out_channels, \
            f"FA_WTConv2d requires in_channels == out_channels, got {in_channels} vs {out_channels}"

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        # ========== 分支 A：深度可分离卷积（保留空间定位） ==========
        self.base_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size,
                      padding=kernel_size // 2, stride=1, dilation=1,
                      groups=in_channels, bias=bias),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        # ========== 分支 B：小波频率分支 ==========
        self.wt_filter, self.iwt_filter = wavelet.create_2d_wavelet_filter(
            wt_type, in_channels, in_channels, torch.float
        )
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size,
                      padding=kernel_size // 2, stride=1,
                      groups=in_channels * 4, bias=False)
            for _ in range(wt_levels)
        ])

        # LL 与 H 差异化初始化：LL 保形，H 抑噪
        self.scale_LL = nn.ModuleList([
            _ScaleModule([1, in_channels, 1, 1], init_scale=0.1)
            for _ in range(wt_levels)
        ])
        self.scale_H = nn.ModuleList([
            _ScaleModule([1, in_channels * 3, 1, 1], init_scale=0.05)
            for _ in range(wt_levels)
        ])

        # ========== 空间-通道门控 ==========
        # 空间门控：输出 (B, C, H, W)，逐像素决定融合权重
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      padding=1, groups=in_channels, bias=False),
            nn.Sigmoid()
        )

        # 通道门控：输出 (B, 2C, 1, 1)，逐通道重加权
        hidden = max(in_channels // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels * 2, 1, bias=False),
            nn.Sigmoid()
        )

        # 输出融合后的 1×1 投影（保持通道一致性）
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.out_bn = nn.BatchNorm2d(in_channels)

        if stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def _wavelet_branch(self, x):
        """小波分支前向"""
        x_ll_in_levels, x_h_in_levels, shapes_in_levels = [], [], []
        curr_x_ll = x

        # -------- 分解 + 频域卷积 --------
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            # 保证偶数尺寸
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, pads)

            curr_x = wavelet.wavelet_2d_transform(curr_x_ll, self.wt_filter)
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4,
                                         shape_x[3], shape_x[4])

            curr_x_tag = self.wavelet_convs[i](curr_x_tag)

            # 拆分 LL / H 分别缩放
            ll_tag = self.scale_LL[i](curr_x_tag[:, :self.in_channels])
            h_tag = self.scale_H[i](curr_x_tag[:, self.in_channels:])
            curr_x_tag = torch.cat([ll_tag, h_tag], dim=1).reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4])
            curr_x_ll = curr_x[:, :, 0]

        # -------- 逆变换重构 --------
        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = wavelet.inverse_2d_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        return next_x_ll

    def forward(self, x):
        identity = x

        # 双分支
        x_base = self.base_conv(x)                      # (B, C, H, W)
        x_tag = self._wavelet_branch(x)                 # (B, C, H, W)

        # 拼接用于门控生成
        cat = torch.cat([x_base, x_tag], dim=1)         # (B, 2C, H, W)

        # 空间门控（逐像素）：g_s ∈ (0,1)^(C,H,W)
        g_s = self.spatial_gate(cat)

        # 通道门控（逐通道）：g_c ∈ (0,1)^(2C,1,1)
        g_c = self.channel_gate(cat)
        g_c_base, g_c_tag = g_c.chunk(2, dim=1)

        # 融合：空间门控决定"在哪混合"，通道门控决定"如何重加权"
        x_out = g_s * x_base * g_c_base + (1 - g_s) * x_tag * g_c_tag

        # 残差 + 输出投影
        x_out = self.out_bn(self.out_proj(x_out)) + identity

        if self.do_stride is not None:
            x_out = self.do_stride(x_out)

        return x_out
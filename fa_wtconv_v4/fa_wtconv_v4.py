import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import wavelet
from .lightweight_sab import SubbandAttention
from .dual_path_wavelet import DualPathWaveletDecomposition


def _gn(c):
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return self.weight * x


class FA_WTConv2dv4(nn.Module):
    """
    FADG-WTConv v4 = v3 + 可学习小波分解（双路径融合）
    相比 v3 的改进：
      1. 双路径小波分解（固定 Haar + 可学习）
      2. 子带级融合门控
      3. 其余模块沿用 v3：SAB、LL/H 独立缩放、空间-通道双重门控、残差、GroupNorm
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 bias=True, wt_levels=1, reduction=8, use_sab=True,
                 use_learnable_wavelet=True, learnable_init_dev=0.01):
        super().__init__()
        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.use_sab = use_sab
        self.use_learnable_wavelet = use_learnable_wavelet

        # ====== 分支 A：空间分支 ======
        self.base_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size,
                      padding=kernel_size // 2, groups=in_channels, bias=bias),
            _gn(in_channels),
            nn.SiLU(inplace=True),
        )

        # ====== 分支 B：小波分支 ======
        # IWT 固定 Haar 重构滤波器
        wt_filter, iwt_filter = wavelet.create_2d_wavelet_filter(
            'haar', in_channels, in_channels, torch.float
        )
        self.register_buffer('wt_filter_fixed', wt_filter)
        self.register_buffer('iwt_filter_fixed', iwt_filter)

        # 双路径可学习小波分解
        self.dual_wt = nn.ModuleList([
            DualPathWaveletDecomposition(
                in_channels,
                init_deviation=learnable_init_dev,
                use_learnable=use_learnable_wavelet
            )
            for _ in range(wt_levels)
        ])

        # 小波域分组卷积
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size,
                      padding=kernel_size // 2, groups=in_channels * 4, bias=False)
            for _ in range(wt_levels)
        ])

        # SAB 子带注意力
        if use_sab:
            self.subband_attn = nn.ModuleList([
                SubbandAttention(in_channels) for _ in range(wt_levels)
            ])
        else:
            self.subband_attn = None

        # LL/H 差异化缩放
        self.scale_LL = nn.ModuleList([
            _ScaleModule([1, in_channels, 1, 1], init_scale=0.1)
            for _ in range(wt_levels)
        ])
        self.scale_H = nn.ModuleList([
            _ScaleModule([1, in_channels * 3, 1, 1], init_scale=0.1)
            for _ in range(wt_levels)
        ])

        # ====== 空间-通道双重门控 ======
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            _gn(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        hidden = max(in_channels // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels * 2, 1, bias=False),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.out_bn = _gn(in_channels)

        if stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def _wavelet_branch(self, x):
        x_ll_in_levels, x_h_in_levels, shapes_in_levels = [], [], []
        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            # 双路径分解
            curr_x = self.dual_wt[i](curr_x_ll)  # (B, C, 4, H/2, W/2)
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4,
                                         shape_x[3], shape_x[4])

            curr_x_tag = self.wavelet_convs[i](curr_x_tag)

            if self.subband_attn is not None:
                curr_x_tag = self.subband_attn[i](curr_x_tag)

            ll_tag = self.scale_LL[i](curr_x_tag[:, :self.in_channels])
            h_tag = self.scale_H[i](curr_x_tag[:, self.in_channels:])
            curr_x_tag = torch.cat([ll_tag, h_tag], dim=1).reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4])
            curr_x_ll = curr_x[:, :, 0]

        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = wavelet.inverse_2d_wavelet_transform(
                curr_x, self.iwt_filter_fixed
            )
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        return next_x_ll

    def forward(self, x):
        identity = x
        x_base = self.base_conv(x)
        x_tag = self._wavelet_branch(x)

        cat = torch.cat([x_base, x_tag], dim=1)

        g_s = self.spatial_gate(cat)   # (B,1,H,W)
        g_c = self.channel_gate(cat)   # (B,2C,1,1)
        g_c_base, g_c_tag = g_c.chunk(2, dim=1)

        x_base_w = g_c_base * x_base
        x_tag_w = g_c_tag * x_tag
        x_out = g_s * x_base_w + (1 - g_s) * x_tag_w
        x_out = self.out_bn(self.out_proj(x_out)) + identity

        if self.do_stride is not None:
            x_out = self.do_stride(x_out)
        return x_out

    def get_subband_fusion_weights(self):
        weights = []
        for i in range(self.wt_levels):
            weights.append(self.dual_wt[i].get_fusion_weights())
        return torch.stack(weights, dim=0)  # (L, 4)


class FADG_WTConvV4Wrapper(nn.Module):
    """YOLOv8 接口包装器。"""
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True,
                 wt_levels=1, use_learnable_wavelet=True):
        super().__init__()
        self.c1, self.c2 = c1, c2

        if c1 != c2 or s > 1:
            self.proj = nn.Sequential(
                nn.Conv2d(c1, c2, 1, stride=s, bias=False),
                _gn(c2),
                nn.SiLU(inplace=True) if act else nn.Identity(),
            )
            inner_stride = 1
        else:
            self.proj = nn.Identity()
            inner_stride = 1

        self.fa_wt = FA_WTConv2dv4(
            c2, c2, kernel_size=k, stride=inner_stride,
            bias=False, wt_levels=wt_levels,
            use_learnable_wavelet=use_learnable_wavelet
        )

    def forward(self, x):
        x = self.proj(x)
        return self.fa_wt(x)
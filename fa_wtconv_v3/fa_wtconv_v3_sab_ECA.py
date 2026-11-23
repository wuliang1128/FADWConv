import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import wavelet
from .lightweight_sab import SubbandAttention


def _gn(c):
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return self.weight * x


class FA_WTConv2dv3(nn.Module):
    """
    FA-WTConv v3 = v2 + 子带注意力（SAB）
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 bias=True, wt_levels=1, wt_type='db1', reduction=8,
                 use_sab=True):
        super().__init__()
        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.use_sab = use_sab

        # 分支 A：深度卷积（ 用 GN）
        self.base_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size,
                      padding=kernel_size // 2, groups=in_channels, bias=bias),
            _gn(in_channels),                     # GN
            nn.SiLU(inplace=True),
        )

        # 分支 B：小波频率分支
        self.wt_filter, self.iwt_filter = wavelet.create_2d_wavelet_filter(
            wt_type, in_channels, in_channels, torch.float
        )
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size,
                      padding=kernel_size // 2, groups=in_channels * 4,
                      bias=False)
            for _ in range(wt_levels)
        ])

        # SAB（每个 level 一个）
        if use_sab:
            self.subband_attn = nn.ModuleList([
                SubbandAttention(in_channels) for _ in range(wt_levels)
            ])
        else:
            self.subband_attn = None

        # LL/H 差异化缩放（继承 v2）
        self.scale_LL = nn.ModuleList([
            _ScaleModule([1, in_channels, 1, 1], init_scale=0.1)
            for _ in range(wt_levels)
        ])
        self.scale_H = nn.ModuleList([
            _ScaleModule([1, in_channels * 3, 1, 1], init_scale=0.1)
            for _ in range(wt_levels)
        ])

        # 跨域融合门控（用 GN）
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            _gn(in_channels),                     #  GN
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),  # 输出 C 通道 对应于 SAB中的 ECA C个通道
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
        self.out_bn = _gn(in_channels)             #  GN

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

            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, pads)

            curr_x = wavelet.wavelet_2d_transform(curr_x_ll, self.wt_filter)
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
                curr_x, self.iwt_filter
            )
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        return next_x_ll

    def forward(self, x):
        identity = x

        x_base = self.base_conv(x)
        x_tag = self._wavelet_branch(x)

        cat = torch.cat([x_base, x_tag], dim=1)

        g_s = self.spatial_gate(cat)  # (B, C, H, W)
        g_c = self.channel_gate(cat)  # (B, 2C, 1, 1)
        g_c_base, g_c_tag = g_c.chunk(2, dim=1)

        # 先做通道重加权，再做空间融合
        x_base_w = g_c_base * x_base
        x_tag_w = g_c_tag * x_tag
        x_out = g_s * x_base_w + (1 - g_s) * x_tag_w
        x_out = self.out_bn(self.out_proj(x_out)) + identity

        if self.do_stride is not None:
            x_out = self.do_stride(x_out)
        return x_out


class FA_WTConvV3Wrapper(nn.Module):
    """适配 YOLO 的包装器"""
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, wt_levels=1):
        super().__init__()
        self.c1, self.c2 = c1, c2

        if c1 != c2 or s > 1:
            self.proj = nn.Sequential(
                nn.Conv2d(c1, c2, 1, stride=s, bias=False),
                _gn(c2),                          #  GN
                nn.SiLU(inplace=True) if act else nn.Identity(),
            )
            inner_stride = 1
        else:
            self.proj = nn.Identity()
            inner_stride = 1

        self.fa_wt = FA_WTConv2dv3(
            c2, c2,
            kernel_size=k, stride=inner_stride,
            bias=False, wt_levels=wt_levels
        )

    def forward(self, x):
        x = self.proj(x)
        return self.fa_wt(x)
# SAB 模块设计（轻量化改造）

# WaveDown 的原始 SAB 参数较多（对 C=512 约 1M），我们做了三点轻量化改造：

# 改造	             原始 SAB	       本方案
# 通道注意力	     1×1 卷积（两轮）	ECA（1D 卷积，参数≈3）
# 空间注意力	     无	               7×7 卷积（参数=98）
# 参数总量（C=512）	 ~1M	           ~22K

import math
import torch
import torch.nn as nn


class SubbandAttention(nn.Module):
    """
    轻量级子带注意力（SAB）。

    输入：拼接后的 4 个子带 (B, 4C, H, W)
    输出：通道-空间双重调制后的子带特征 (B, 4C, H, W)

    设计要点：
      1. 深度卷积捕获局部上下文
      2. 1×1 卷积通道注意力，保留空间维度
      3. Max+Mean 池化 + 7×7 卷积做空间注意力
      4. 残差形式 y = x + x * ca * sa，保证梯度稳定
    """
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        c4 = channels * 4

        # ---- 局部上下文：深度卷积 ----
        self.dw = nn.Conv2d(c4, c4, kernel_size=3, padding=1,
                            groups=c4, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(c4 // 4, 32),
                                 num_channels=c4)

        # ---- 通道注意力（保留空间维度）1x1卷积 参数量较大  ----
        #self.channel_conv = nn.Sequential(
        #    nn.Conv2d(c4, c4 // 4, kernel_size=1, bias=False),
        #   nn.SiLU(inplace=True),
        #    nn.Conv2d(c4 // 4, c4, kernel_size=1, bias=False),
        #    nn.Sigmoid()
        #)

        # ---- 通道注意力（保留空间维度）ECA 参数量较小  ----
        t = int(abs(math.log2(c4) + 1) / 2)
        eca_k = t if t % 2 else t + 1
        self.eca_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=eca_k,
                          padding=eca_k // 2, bias=False)

        # ---- 空间注意力 ----
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                      padding=kernel_size // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, 4C, H, W)
        ctx = self.norm(self.dw(x))
        b, c, h, w = ctx.shape

        # ---- 直接用 1×1 卷积从完整特征图学权重 参数量大 ----
        #ca = self.channel_conv(ctx)  # (B, 4C, H, W)
        # ---- ECA ----
        ca = self.eca_pool(ctx).squeeze(-1).transpose(-1, -2)
        ca = self.eca_conv(ca).transpose(-1, -2).unsqueeze(-1)
        ca = self.sigmoid(ca)

        # ---- 空间注意力 ----
        pooled = torch.cat([ctx.max(dim=1, keepdim=True)[0],
                            ctx.mean(dim=1, keepdim=True)], dim=1)  # (B, 2, H, W)
        sa = self.sigmoid(self.spatial_conv(pooled))  # (B, 1, H, W)

        # ---- 残差双重调制 ----
        return x + x * ca * sa
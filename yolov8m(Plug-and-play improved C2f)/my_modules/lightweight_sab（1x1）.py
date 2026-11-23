import math
import torch
import torch.nn as nn


class SubbandAttention(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        c4 = channels * 4

        # ---- 局部上下文：深度卷积 ----
        self.dw = nn.Conv2d(c4, c4, kernel_size=3, padding=1,
                            groups=c4, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(c4 // 4, 32),
                                 num_channels=c4)

        # ---- 通道注意力（保留空间维度） ----
        self.channel_conv = nn.Sequential(
            nn.Conv2d(c4, c4 // 4, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(c4 // 4, c4, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # ---- 空间注意力 ----
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                      padding=kernel_size // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, 4C, H, W)
        ctx = self.norm(self.dw(x))
        b, c, h, w = ctx.shape

        # ---- 直接用 1×1 卷积从完整特征图学权重 ----
        ca = self.channel_conv(ctx)  # (B, 4C, H, W)

        # ---- 空间注意力 ----
        pooled = torch.cat([ctx.max(dim=1, keepdim=True)[0],
                            ctx.mean(dim=1, keepdim=True)], dim=1)  # (B, 2, H, W)
        sa = self.sigmoid(self.spatial_conv(pooled))  # (B, 1, H, W)

        # ---- 残差双重调制 ----
        return x + x * ca * sa
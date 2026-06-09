import torch
import torch.nn as nn

from models.GBC import BottConv, get_norm_layer


class CAB(nn.Module):
    """Continuity-Aware Bottleneck used in the CACrackNet encoder.

    CAB combines a local convolution branch with horizontal and vertical strip
    pooling branches, then applies channel interaction and a residual
    connection.
    """

    def __init__(self, in_channels, norm_type="GN"):
        super().__init__()
        norm_groups = max(in_channels // 16, 1)

        self.local_branch = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, norm_groups),
            nn.ReLU(),
        )

        self.h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.h_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, (1, 3), padding=(0, 1), bias=False),
            nn.GroupNorm(max(in_channels // 4 // 16, 1), in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, (1, 3), padding=(0, 1), bias=False),
        )

        self.v_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.v_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, (3, 1), padding=(1, 0), bias=False),
            nn.GroupNorm(max(in_channels // 4 // 16, 1), in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, (3, 1), padding=(1, 0), bias=False),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1, bias=False),
            get_norm_layer(norm_type, in_channels, norm_groups),
            nn.ReLU(),
        )

        self.channel_mix = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, norm_groups),
            nn.ReLU(),
        )

        self.out_conv = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, max(16, norm_groups)),
            nn.ReLU(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        del b, c
        residual = x

        x_local = self.local_branch(x)

        x_h = self.h_pool(x)
        x_h = self.h_conv(x_h)
        x_h = x_h.expand(-1, -1, h, -1)

        x_v = self.v_pool(x)
        x_v = self.v_conv(x_v)
        x_v = x_v.expand(-1, -1, -1, w)

        x_fused = self.fuse(torch.cat([x_local, x_h, x_v], dim=1))
        x_channel = self.channel_mix(x)
        x = x_fused * x_channel
        x = self.out_conv(x)
        return x + residual

"""Spatial-Channel Gated Fusion (SCGF).

The original training code named this module ``AGCF``. The class name is kept
for checkpoint compatibility, while the paper refers to the same mechanism as
SCGF.
"""

import torch
import torch.nn as nn


class AGCF(nn.Module):
    """
    Spatial-Channel Gated Fusion.

    The gate is computed from the guidance feature using a spatial branch and a
    channel branch, then used to interpolate guidance and base features.

    Args:
        channels (int): 输入特征的通道数
        reduction (int): 通道缩减比例，默认为4
    
    Input:
        base_feat: 基础特征 [B, C, H, W]
        guidance_feat: 引导特征 [B, C, H, W]
    
    Output:
        fused_feat: 融合后的特征 [B, C, H, W]
    
    Example:
        >>> agcf = AGCF(channels=256, reduction=4)
        >>> base = torch.randn(2, 256, 32, 32)
        >>> guidance = torch.randn(2, 256, 32, 32)
        >>> output = agcf(base, guidance)
        >>> print(output.shape)  # torch.Size([2, 256, 32, 32])
    """
    
    def __init__(self, channels, reduction=4):
        super(AGCF, self).__init__()
        
        self.channels = channels
        self.reduction = reduction
        
        # ===== 空间门控（来自 AF，验证有效）=====
        # 用 guidance 的激活强度，这是 AF 成功的关键
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        
        # Channel gate from guidance statistics.
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, base_feat, guidance_feat):
        """
        前向传播
        
        Args:
            base_feat: 基础特征 [B, C, H, W]
            guidance_feat: 引导特征 [B, C, H, W]
        
        Returns:
            融合后的特征 [B, C, H, W]
        """
        # 1. Spatial gate from guidance activations.
        max_act = torch.max(guidance_feat, dim=1, keepdim=True)[0]  # [B, 1, H, W]
        avg_act = torch.mean(guidance_feat, dim=1, keepdim=True)    # [B, 1, H, W]
        spatial_gate = self.spatial_gate(torch.cat([max_act, avg_act], dim=1))  # [B, 1, H, W]
        
        # 2. Channel gate from global guidance statistics.
        channel_gate = self.channel_gate(guidance_feat)  # [B, C, 1, 1]
        
        # 3. Joint spatial-channel gate.
        gate = spatial_gate * channel_gate  # [B, C, H, W] 广播
        
        # 4. Adaptive interpolation.
        fused = gate * guidance_feat + (1 - gate) * base_feat
        
        return fused
    
    def extra_repr(self):
        """返回模块的额外信息"""
        return f'channels={self.channels}, reduction={self.reduction}'


class ActivationGuidedChannelFusion(AGCF):
    """Backward-compatible alias."""
    pass


if __name__ == "__main__":
    # 测试代码
    print("Testing AGCF Module...")
    
    # 创建模块
    agcf = AGCF(channels=256, reduction=4)
    
    # 创建测试数据
    batch_size = 2
    channels = 256
    height, width = 32, 32
    
    base_feat = torch.randn(batch_size, channels, height, width)
    guidance_feat = torch.randn(batch_size, channels, height, width)
    
    # 前向传播
    output = agcf(base_feat, guidance_feat)
    
    # 打印结果
    print(f"Input shape: {base_feat.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Module parameters: {sum(p.numel() for p in agcf.parameters())}")
    print(f"Module info: {agcf}")
    
    print("\nTest passed!")

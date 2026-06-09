'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import torch.nn as nn

class BottConv(nn.Module):
    """
    瓶颈卷积模块，实现了深度可分离卷积的变体。
    
    网络结构:
    1. 第一个逐点卷积(1x1)降低通道数到mid_channels
    2. 深度卷积(depthwise)处理空间信息
    3. 第二个逐点卷积(1x1)调整通道数到out_channels
    
    这种结构大幅减少参数量和计算量，同时保持良好的表示能力。
    """
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        # 第一个1x1卷积，降低通道数
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        # 深度卷积，每个通道单独处理，保持通道数不变
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        # 第二个1x1卷积，调整到目标通道数
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        # 通过第一个逐点卷积
        x = self.pointwise_1(x)
        # 通过深度卷积
        x = self.depthwise(x)
        # 通过第二个逐点卷积
        x = self.pointwise_2(x)
        return x


def get_norm_layer(norm_type, channels, num_groups):
    """
    获取指定类型的标准化层。
    
    参数:
        norm_type: 标准化类型，'GN'表示GroupNorm，其他值使用InstanceNorm3d
        channels: 通道数
        num_groups: 组归一化的组数
        
    返回:
        配置好的标准化层
    """
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.InstanceNorm3d(channels)


class GBC(nn.Module):
    """
    全局上下文模块(Global Block Context)
    
    网络结构:
    1. 包含4个处理块，采用残差连接结构
    2. 前两个块使用3x3卷积捕获局部上下文
    3. 第三个块使用1x1卷积处理通道信息
    4. 通过特征相乘(x1*x2)实现特征交互
    5. 第四个块进一步处理交互后的特征
    6. 最后添加残差连接保持信息流动
    
    这种设计结合了局部和全局上下文信息，增强了特征表示能力。
    """
    def __init__(self, in_channels, norm_type='GN'):
        super(GBC, self).__init__()

        # 第一个处理块: 3x3卷积+标准化+激活
        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),  # 3x3卷积，保持通道数不变
            get_norm_layer(norm_type, in_channels, in_channels // 16),      # 标准化层
            nn.ReLU()                                                       # 激活函数
        )

        # 第二个处理块: 3x3卷积+标准化+激活
        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),  # 3x3卷积，保持通道数不变
            get_norm_layer(norm_type, in_channels, in_channels // 16),      # 标准化层
            nn.ReLU()                                                       # 激活函数
        )

        # 第三个处理块: 1x1卷积+标准化+激活
        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),  # 1x1卷积，处理通道信息
            get_norm_layer(norm_type, in_channels, in_channels // 16),      # 标准化层
            nn.ReLU()                                                       # 激活函数
        )

        # 第四个处理块: 1x1卷积+标准化+激活
        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),  # 1x1卷积，处理通道信息
            get_norm_layer(norm_type, in_channels, 16),                     # 标准化层
            nn.ReLU()                                                       # 激活函数
        )

    def forward(self, x):
        # 保存输入作为残差连接
        residual = x

        # 通过第一个和第二个块处理特征
        x1 = self.block1(x)
        x1 = self.block2(x1)
        
        # 通过第三个块处理特征
        x2 = self.block3(x)
        
        # 特征交互: 相乘操作
        x = x1 * x2
        
        # 通过第四个块进一步处理
        x = self.block4(x)

        # 添加残差连接
        return x + residual

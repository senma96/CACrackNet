import torch
import torch.nn as nn
import torch.nn.functional as F

class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, patch_size=8):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = patch_size

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        # 使用一个卷积来动态生成频域权重，而不是固定的参数
        self.fft_conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x_in = self.project_in(x)
        x1, x2 = self.dwconv(x_in).chunk(2, dim=1)
        
        # Gated-MLP 部分
        x_gelu = F.gelu(x1)

        # FFT 分支
        b, c, h, w = x2.shape
        h_pad = (self.patch_size - h % self.patch_size) % self.patch_size
        w_pad = (self.patch_size - w % self.patch_size) % self.patch_size
        
        x2_padded = F.pad(x2, (0, w_pad, 0, h_pad), mode='reflect')
        
        x2_fft = torch.fft.rfft2(x2_padded, norm='ortho')
        
        # 通过卷积生成与输入相关的动态权重
        weights = self.fft_conv(x2)
        weights_padded = F.pad(weights, (0, w_pad, 0, h_pad), mode='reflect')
        weights_fft = torch.fft.rfft2(weights_padded, norm='ortho')

        # 在频域中相乘
        x_fft_filtered = x2_fft * weights_fft
        
        x_filtered = torch.fft.irfft2(x_fft_filtered, s=(h + h_pad, w + w_pad), norm='ortho')
        
        # 裁剪回原始尺寸
        x_filtered = x_filtered[:, :, :h, :w]

        # 门控机制融合
        x = x_gelu * x_filtered
        x = self.project_out(x)
        
        return x

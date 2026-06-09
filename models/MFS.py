import torch
import torch.nn as nn

from models.DySample import DySample
from models.Edffn import EDFFN
from models.GBC import BottConv, GBC
from models.agcf import AGCF


class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x)


class MFS(nn.Module):
    """CACrackNet decoder.

    The release version fixes the decoder to the paper configuration:
    multi-scale projection, DySample alignment, deep-to-shallow SCGF cascade,
    GBC refinement, BottConv compression, and EDFFN refinement.
    """

    def __init__(self, embedding_dim=8):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.linear_c4 = MLP(input_dim=128, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=64, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=32, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=16, embed_dim=embedding_dim)

        # SCGF is implemented by the AGCF class name for checkpoint compatibility.
        self.agcf_c4 = AGCF(channels=embedding_dim, reduction=4)
        self.agcf_c3 = AGCF(channels=embedding_dim, reduction=4)
        self.agcf_c2 = AGCF(channels=embedding_dim, reduction=4)

        self.GBC_C = GBC(embedding_dim * 4)
        self.GBC_8 = GBC(8, norm_type="IN")

        self.linear_fuse = BottConv(embedding_dim * 4, embedding_dim, embedding_dim // 8, kernel_size=1, padding=0, stride=1)
        self.deffn = EDFFN(dim=embedding_dim, ffn_expansion_factor=2.66, bias=False)

        self.linear_pred = BottConv(embedding_dim, 1, 1, kernel_size=1)
        self.linear_pred_1 = nn.Conv2d(1, 1, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)

        self.DySample_C_2 = DySample(embedding_dim, scale=2)
        self.DySample_C_4 = DySample(embedding_dim, scale=4)
        self.DySample_C_8 = DySample(embedding_dim, scale=8)

    def _project(self, feature, linear):
        b, c, h, w = feature.shape
        return linear(feature.reshape(b, c, h * w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)

    def forward(self, inputs):
        c4, c3, c2, c1 = inputs

        out_c4 = self.DySample_C_8(self._project(c4, self.linear_c4))
        out_c3 = self.DySample_C_4(self._project(c3, self.linear_c3))
        out_c3 = self.agcf_c4(out_c3, out_c4)

        out_c2 = self.DySample_C_2(self._project(c2, self.linear_c2))
        out_c2 = self.agcf_c3(out_c2, out_c3)

        out_c1 = self._project(c1, self.linear_c1)
        out_c1 = self.agcf_c2(out_c1, out_c2)

        fused_features = torch.cat([out_c4, out_c3, out_c2, out_c1], dim=1)
        out_c = self.GBC_C(fused_features)
        out_c = self.linear_fuse(out_c)
        out_c = self.deffn(out_c)
        out_c = self.dropout(out_c)
        return self.linear_pred_1(self.linear_pred(out_c))

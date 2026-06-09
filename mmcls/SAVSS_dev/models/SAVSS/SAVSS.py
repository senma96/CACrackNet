from typing import Sequence

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcls.SAVSS_dev.models.modules.patch_embed import ConvPatchEmbed, to_2tuple
from models.Edffn import EDFFN
from models.GBC import BottConv, GBC
from models.GBC_Variants import CAB


def trunc_normal_(tensor, mean=0.0, std=1.0):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std)


class ModuleList(nn.ModuleList):
    pass


def build_norm_layer(norm_cfg, num_features, postfix=""):
    if norm_cfg is None:
        return f"norm{postfix}", nn.Identity()
    norm_type = norm_cfg.get("type", "LN")
    eps = norm_cfg.get("eps", 1e-6)
    if norm_type != "LN":
        raise ValueError(f"Unsupported norm layer in release build: {norm_type}")
    return f"ln{postfix}", nn.LayerNorm(num_features, eps=eps)


def resize_pos_embed(pos_embed, src_shape, dst_shape, mode="bicubic", num_extra_tokens=0):
    if src_shape == dst_shape:
        return pos_embed
    if num_extra_tokens:
        extra_tokens = pos_embed[:, :num_extra_tokens]
        pos_tokens = pos_embed[:, num_extra_tokens:]
    else:
        extra_tokens = None
        pos_tokens = pos_embed

    b, _, c = pos_tokens.shape
    pos_tokens = pos_tokens.reshape(b, src_shape[0], src_shape[1], c).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(pos_tokens, size=dst_shape, mode=mode, align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(b, dst_shape[0] * dst_shape[1], c)
    if extra_tokens is not None:
        pos_tokens = torch.cat([extra_tokens, pos_tokens], dim=1)
    return pos_tokens


class ExtendedSAVSSLayer(nn.Module):
    """Release implementation of the paper encoder block.

    The published checkpoint uses CAB in the encoder. Other experimental
    variants from the research workspace are intentionally omitted.
    """

    def __init__(
        self,
        embed_dims,
        use_rms_norm=False,
        with_dwconv=False,
        drop_path_rate=0.0,
        mamba_cfg=None,
        gbc_type="cab",
        use_deffn=False,
        deffn_expansion_factor=2.66,
    ):
        super().__init__()
        del use_rms_norm, with_dwconv, drop_path_rate, mamba_cfg
        if gbc_type != "cab":
            self.freq_module = GBC(embed_dims)
        else:
            self.freq_module = CAB(embed_dims)
        self.use_deffn = use_deffn
        if use_deffn:
            self.deffn = EDFFN(dim=embed_dims, ffn_expansion_factor=deffn_expansion_factor, bias=False)

    def forward(self, x, hw_shape=None):
        original_shape = x.shape
        if len(x.shape) == 3:
            b, _, c = x.shape
            if hw_shape is None:
                raise ValueError("hw_shape is required for token input.")
            h, w = hw_shape
            x = x.transpose(1, 2).view(b, c, h, w)

        x = self.freq_module(x)

        if self.use_deffn:
            x = self.deffn(x)

        if len(original_shape) == 3:
            b, c, h, w = x.shape
            x = x.view(b, c, h * w).transpose(1, 2)
        return x


class SAVSS(nn.Module):
    """Pure-PyTorch CACrackNet encoder used by the release checkpoint."""

    arch_zoo = {
        "Crack": {
            "patch_size": 8,
            "embed_dims": 256,
            "num_layers": 4,
            "num_convs_patch_embed": 2,
            "layers_with_dwconv": [],
            "layer_cfgs": {
                "use_rms_norm": False,
                "mamba_cfg": {
                    "d_state": 16,
                    "expand": 2,
                    "conv_size": 7,
                    "dt_init": "random",
                    "conv_bias": True,
                    "bias": True,
                    "default_hw_shape": (512 // 8, 512 // 8),
                },
            },
        }
    }

    def __init__(
        self,
        img_size=224,
        in_channels=3,
        arch=None,
        patch_size=16,
        embed_dims=192,
        num_layers=20,
        num_convs_patch_embed=1,
        with_pos_embed=True,
        out_indices=-1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_cfg=dict(type="LN", eps=1e-6),
        final_norm=True,
        interpolate_mode="bicubic",
        layer_cfgs=dict(),
        layers_with_dwconv=[],
        init_cfg=None,
        test_cfg=dict(),
        convert_syncbn=False,
        freeze_patch_embed=False,
        gbc_type="cab",
        use_deffn=False,
        deffn_expansion_factor=2.66,
        **kwargs,
    ):
        super().__init__()
        del init_cfg, test_cfg, convert_syncbn, kwargs

        self.img_size = to_2tuple(img_size)
        self.arch = arch
        self.freeze_patch_embed = freeze_patch_embed

        if self.arch is None:
            self.embed_dims = embed_dims
            self.num_layers = num_layers
            self.patch_size = patch_size
            self.num_convs_patch_embed = num_convs_patch_embed
            self.layers_with_dwconv = layers_with_dwconv
            layer_template = layer_cfgs
        else:
            if self.arch not in self.arch_zoo:
                raise KeyError(f"Unsupported arch: {self.arch}")
            arch_cfg = self.arch_zoo[self.arch]
            self.embed_dims = arch_cfg["embed_dims"]
            self.num_layers = arch_cfg["num_layers"]
            self.patch_size = arch_cfg["patch_size"]
            self.num_convs_patch_embed = arch_cfg["num_convs_patch_embed"]
            self.layers_with_dwconv = arch_cfg["layers_with_dwconv"]
            layer_template = arch_cfg["layer_cfgs"]

        self.with_pos_embed = with_pos_embed
        self.interpolate_mode = interpolate_mode
        self.patch_embed = ConvPatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            num_convs=self.num_convs_patch_embed,
            patch_size=self.patch_size,
            stride=self.patch_size,
        )
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        if with_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))
            trunc_normal_(self.pos_embed, std=0.02)

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        if not isinstance(out_indices, Sequence):
            raise TypeError("out_indices must be a sequence or int")
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            if not 0 <= out_indices[i] <= self.num_layers:
                raise ValueError(f"Invalid out index: {index}")
        self.out_indices = out_indices

        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        if isinstance(layer_template, dict):
            layer_cfgs = [copy.deepcopy(layer_template) for _ in range(self.num_layers)]

        self.layers = ModuleList()
        for i in range(self.num_layers):
            cfg = layer_cfgs[i]
            self.layers.append(
                ExtendedSAVSSLayer(
                    embed_dims=self.embed_dims,
                    use_rms_norm=cfg.get("use_rms_norm", False),
                    with_dwconv=i in self.layers_with_dwconv,
                    drop_path_rate=dpr[i],
                    mamba_cfg=cfg.get("mamba_cfg", {}),
                    gbc_type=gbc_type,
                    use_deffn=use_deffn,
                    deffn_expansion_factor=deffn_expansion_factor,
                )
            )

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        for i in out_indices:
            if i != self.num_layers - 1:
                norm_layer = build_norm_layer(norm_cfg, self.embed_dims)[1] if norm_cfg is not None else nn.Identity()
                self.add_module(f"norm_layer{i}", norm_layer)

        self.conv256to128 = BottConv(256, 128, 32, kernel_size=1, stride=1, padding=0)
        self.conv256to64 = BottConv(256, 64, 16, kernel_size=1, stride=1, padding=0)
        self.conv256to32 = BottConv(256, 32, 8, kernel_size=1, stride=1, padding=0)
        self.conv256to16 = BottConv(256, 16, 4, kernel_size=1, stride=1, padding=0)

        self.gn128 = nn.GroupNorm(num_channels=128, num_groups=8)
        self.gn64 = nn.GroupNorm(num_channels=64, num_groups=4)
        self.gn32 = nn.GroupNorm(num_channels=32, num_groups=2)
        self.gn16 = nn.GroupNorm(num_channels=16, num_groups=2)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        if self.with_pos_embed:
            trunc_normal_(self.pos_embed, std=0.02)
        self.set_freeze_patch_embed()

    def set_freeze_patch_embed(self):
        if self.freeze_patch_embed:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

    def forward(self, x):
        x, patch_resolution = self.patch_embed(x)

        if self.with_pos_embed:
            pos_embed = resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=0,
            )
            x = x + pos_embed

        x = self.drop_after_pos(x)
        outs = []

        for i, layer in enumerate(self.layers):
            x = layer(x, hw_shape=patch_resolution)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                b, _, c = x.shape
                patch_token = x.reshape(b, *patch_resolution, c)
                if i != self.num_layers - 1:
                    norm_layer = getattr(self, f"norm_layer{i}")
                    patch_token = norm_layer(patch_token)
                patch_token = patch_token.permute(0, 3, 1, 2)

                if i == self.out_indices[0]:
                    patch_token_mid = self.gn128(self.conv256to128(patch_token))
                    outs.append(F.interpolate(patch_token_mid, size=(64, 64), mode="bilinear", align_corners=False))
                elif i == self.out_indices[1]:
                    patch_token_mid = self.gn64(self.conv256to64(patch_token))
                    outs.append(F.interpolate(patch_token_mid, size=(128, 128), mode="bilinear", align_corners=False))
                elif i == self.out_indices[2]:
                    patch_token_mid = self.gn32(self.conv256to32(patch_token))
                    outs.append(F.interpolate(patch_token_mid, size=(256, 256), mode="bilinear", align_corners=False))
                elif i == self.out_indices[3]:
                    patch_token_mid = self.gn16(self.conv256to16(patch_token))
                    outs.append(F.interpolate(patch_token_mid, size=(512, 512), mode="bilinear", align_corners=False))

        return outs

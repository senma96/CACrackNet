import torch
from torch import nn

from mmcls.SAVSS_dev.models.SAVSS.SAVSS import SAVSS
from models.MFS import MFS


class Decoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.MFS = MFS(embedding_dim=8)

    def forward(self, samples):
        return self.MFS(self.backbone(samples))


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, dims=(-2, -1)):
        super().__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        return 1 - dc.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_ratio=0.83, dice_ratio=0.17):
        super().__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()
        self.bce_ratio = bce_ratio
        self.dice_ratio = dice_ratio

    def forward(self, y_pred, y_true):
        bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        return self.bce_ratio * bce + self.dice_ratio * dice


def build(args):
    device = torch.device(args.device)
    args.device = device

    backbone = SAVSS(
        arch="Crack",
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.2,
        final_norm=True,
        convert_syncbn=True,
        gbc_type="cab",
        use_deffn=False,
    )
    model = Decoder(backbone)
    criterion = BCEDiceLoss(
        bce_ratio=getattr(args, "BCELoss_ratio", 0.83),
        dice_ratio=getattr(args, "DiceLoss_ratio", 0.17),
    ).to(device)
    return model, criterion

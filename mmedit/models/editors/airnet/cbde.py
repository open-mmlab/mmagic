# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine.model import BaseModule

from mmedit.registry import MODELS
from .moco import MoCo


@MODELS.register_module()
class CBDE(BaseModule):
    """Contrastive-Based Degradation Encoder.

    Args:
        batch_size (int): Batch size of inputs
        dim (int): Dimension of compact feature vectors.
            Defaults to 128.
    """

    def __init__(self, batch_size, dim=128):
        super(CBDE, self).__init__()

        # Encoder
        self.E = MoCo(base_encoder=ResEncoder, dim=dim, K=batch_size * dim)

    def forward(self, x_query, x_key):
        # degradation-aware represenetion learning
        outputs = self.E(x_query, x_key)

        return outputs


# Components
class ResBlock(nn.Module):
    """ResBlock for ResEncoder.

    Args:
        in_feat (int): Channel dimension of input features.
        out_feat (int): Channel dimension of input features.
        stride (int): Stride for convolution.
            Defaults to 1.
    """

    def __init__(self, in_feat, out_feat, stride=1):
        super(ResBlock, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(
                in_feat,
                out_feat,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False),
            nn.BatchNorm2d(out_feat),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(
                out_feat, out_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_feat, out_feat, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_feat))

    def forward(self, x):
        return nn.LeakyReLU(0.1, True)(self.backbone(x) + self.shortcut(x))


class ResEncoder(nn.Module):
    """ResEncoder for MoCo as backbone."""

    def __init__(self):
        super(ResEncoder, self).__init__()

        self.E_pre = ResBlock(in_feat=3, out_feat=64, stride=1)
        self.E = nn.Sequential(
            ResBlock(in_feat=64, out_feat=128, stride=2),
            ResBlock(in_feat=128, out_feat=256, stride=2),
            nn.AdaptiveAvgPool2d(1))

        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        """Forward function.

        Args:
            x: input tensor
        Return:
            inter: the feature after of the first block
            fea: the feature after all blocks
            out: mlp output layer for fea
        """
        inter = self.E_pre(x)
        fea = self.E(inter).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out, inter

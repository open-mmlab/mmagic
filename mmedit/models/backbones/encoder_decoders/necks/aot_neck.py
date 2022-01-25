# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class AOTBlockNeck(nn.Module):
    """Dilation backbone used in AOT-GAN model.

    This implementation follows:
    Aggregated Contextual Transformations for High-Resolution Image Inpainting

    Args:
        in_channels (int, optional): Channel number of input feature.
            Default: 256.
        dilation_rates (Tuple[int], optional): The dilation rates used
        for AOT block. Default: (1, 2, 4, 8).
        num_aotblock (int, optional): Number of AOT blocks. Default: 8.
        act_cfg (dict, optional): Config dict for activation layer,
            "relu" by default.
        kwargs (keyword arguments).
    """

    def __init__(self,
                 in_channels=256,
                 dilation_rates=(1, 2, 4, 8),
                 num_aotblock=8,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super().__init__()

        self.dilation_rates = list(dilation_rates)

        self.model = nn.Sequential(*[(AOTBlock(
            in_channels=in_channels,
            dilation_rates=self.dilation_rates,
            act_cfg=act_cfg,
        )) for _ in range(0, num_aotblock)])

    def forward(self, x):
        x = self.model(x)
        return x


class AOTBlock(nn.Module):
    """AOT Block which constitutes the dilation backbone.

    This implementation follows:
    Aggregated Contextual Transformations for High-Resolution Image Inpainting

    The AOT Block adopts the split-transformation-merge strategy:
    Splitting: A kernel with 256 output channels is split into four
               64-channel sub-kernels.
    Transforming: Each sub-kernel performs a different transformation with
                  a different dilation rate.
    Splitting: Sub-kernels with different receptive fields are merged.

    Args:
        in_channels (int, optional): Channel number of input feature.
            Default: 256.
        dilation_rates (Tuple[int]): The dilation rates used for AOT block.
            Default (1, 2, 4, 8).
        act_cfg (dict, optional): Config dict for activation layer,
            "relu" by default.
        kwargs (keyword arguments).
    """

    def __init__(self,
                 in_channels=256,
                 dilation_rates=(1, 2, 4, 8),
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super().__init__()
        self.dilation_rates = dilation_rates
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad2d(dilation_rate),
                ConvModule(
                    in_channels,
                    in_channels // 4,
                    kernel_size=3,
                    dilation=dilation_rate,
                    act_cfg=act_cfg)) for dilation_rate in self.dilation_rates
        ])

        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            ConvModule(in_channels, in_channels, 3, dilation=1, act_cfg=None))

        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            ConvModule(in_channels, in_channels, 3, dilation=1, act_cfg=None))

    def normalize(self, x):
        mean = x.mean((2, 3), keepdim=True)
        std = x.std((2, 3), keepdim=True) + 1e-9
        x = 2 * (x - mean) / std - 1
        x = 5 * x
        return x

    def forward(self, x):

        dilate_x = [
            self.blocks[i](x) for i in range(0, len(self.dilation_rates))
        ]
        dilate_x = torch.cat(dilate_x, 1)
        dilate_x = self.fuse(dilate_x)
        mask = self.normalize(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + dilate_x * mask

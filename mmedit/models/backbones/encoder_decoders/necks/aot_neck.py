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
        in_channels (int): Channel number of input feature.
        dilation_rates (str): The dilation rates used
        for AOT block. Default: 1+2+4+8.
        num_aotblock (int): Number of AOT blocks. Default: 8.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
        kwargs (keyword arguments).
    """

    def __init__(self,
                 in_channels=256,
                 dilation_rates='1+2+4+8',
                 num_aotblock=8,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super().__init__()

        self.dilation_rates = list(map(int, list(dilation_rates.split('+'))))
        self.num_aotblock = num_aotblock

        self.model = nn.Sequential()

        for i in range(self.num_aotblock):
            self.model.add_module(
                f'aotblock{i + 1}',
                AOTBlock(
                    in_channels=in_channels,
                    dilation_rates=self.dilation_rates,
                    act_cfg=act_cfg,
                ))

    def forward(self, x):
        return self.model(x)


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
        in_channels (int): Channel number of input feature.
        dilation_rates (str): The dilation rates used for AOT block.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
        kwargs (keyword arguments).
    """

    def __init__(self,
                 in_channels,
                 dilation_rates,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super().__init__()
        self.dilation_rates = dilation_rates

        for i, dilation_rate in enumerate(dilation_rates):
            self.__setattr__(
                f'block{i + 1}',
                nn.Sequential(
                    nn.ReflectionPad2d(dilation_rate),
                    ConvModule(
                        in_channels,
                        in_channels // 4,
                        kernel_size=3,
                        dilation=dilation_rate,
                        act_cfg=act_cfg)))

        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            ConvModule(in_channels, in_channels, 3, dilation=1, act_cfg=None))

        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            ConvModule(in_channels, in_channels, 3, dilation=1, act_cfg=None))

    def norm(self, x):
        mean = x.mean((2, 3), keepdim=True)
        std = x.std((2, 3), keepdim=True) + 1e-9
        x = 2 * (x - mean) / std - 1
        x = 5 * x
        return x

    def forward(self, x):
        dilate_x = [
            self.__getattr__(f'block{i + 1}')(x)
            for i in range(len(self.dilation_rates))
        ]
        dilate_x = torch.cat(dilate_x, 1)
        dilate_x = self.fuse(dilate_x)
        mask = self.norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + dilate_x * mask

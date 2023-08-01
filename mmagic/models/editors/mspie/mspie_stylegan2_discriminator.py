# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from mmengine.model import BaseModule

from mmagic.registry import MODELS
from ..stylegan1 import EqualLinearActModule
from ..stylegan2 import ConvDownLayer, ModMBStddevLayer, ResBlock


@MODELS.register_module()
class MSStyleGAN2Discriminator(BaseModule):
    """StyleGAN2 Discriminator.

    The architecture of this discriminator is proposed in StyleGAN2. More
    details can be found in: Analyzing and Improving the Image Quality of
    StyleGAN CVPR2020.

    Args:
        in_size (int): The input size of images.
        channel_multiplier (int, optional): The multiplier factor for the
            channel number. Defaults to 2.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 3, 3, 1].
        mbstd_cfg (dict, optional): Configs for minibatch-stddev layer.
            Defaults to dict(group_size=4, channel_groups=1).
    """

    def __init__(self,
                 in_size,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 mbstd_cfg=dict(group_size=4, channel_groups=1),
                 with_adaptive_pool=False,
                 pool_size=(2, 2)):
        super().__init__()
        self.with_adaptive_pool = with_adaptive_pool
        self.pool_size = pool_size

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        log_size = int(np.log2(in_size))
        in_channels = channels[in_size]
        convs = [ConvDownLayer(3, channels[in_size], 1)]

        for i in range(log_size, 2, -1):
            out_channel = channels[2**(i - 1)]
            convs.append(ResBlock(in_channels, out_channel, blur_kernel))

            in_channels = out_channel

        self.convs = nn.Sequential(*convs)
        self.mbstd_layer = ModMBStddevLayer(**mbstd_cfg)

        self.final_conv = ConvDownLayer(in_channels + 1, channels[4], 3)

        if self.with_adaptive_pool:
            self.adaptive_pool = nn.AdaptiveAvgPool2d(pool_size)
            linear_in_channels = channels[4] * pool_size[0] * pool_size[1]
        else:
            linear_in_channels = channels[4] * 4 * 4

        self.final_linear = nn.Sequential(
            EqualLinearActModule(
                linear_in_channels,
                channels[4],
                act_cfg=dict(type='fused_bias')),
            EqualLinearActModule(channels[4], 1),
        )

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Predict score for the input image.
        """
        x = self.convs(x)

        x = self.mbstd_layer(x)
        x = self.final_conv(x)
        if self.with_adaptive_pool:
            x = self.adaptive_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)

        return x

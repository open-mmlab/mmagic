# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mmedit.registry import MODULES
from ..pggan import (EqualizedLRConvDownModule, EqualizedLRConvModule,
                     MiniBatchStddevLayer)
from .stylegan1_modules import Blur, EqualLinearActModule


@MODULES.register_module('StyleGANv1Discriminator')
@MODULES.register_module()
class StyleGAN1Discriminator(nn.Module):
    """StyleGAN1 Discriminator.

    The architecture of this discriminator is proposed in StyleGAN1. More
    details can be found in: A Style-Based Generator Architecture for
    Generative Adversarial Networks CVPR2019.

    Args:
        in_size (int): The input size of images.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 2, 1].
        mbstd_cfg (dict, optional): Configs for minibatch-stddev layer.
            Defaults to dict(group_size=4).
    """

    def __init__(self,
                 in_size,
                 blur_kernel=[1, 2, 1],
                 mbstd_cfg=dict(group_size=4)):
        super().__init__()

        self.with_mbstd = mbstd_cfg is not None
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16,
        }

        log_size = int(np.log2(in_size))
        self.log_size = log_size
        in_channels = channels[in_size]

        self.convs = nn.ModuleList()
        self.from_rgb = nn.ModuleList()

        for i in range(log_size, 2, -1):
            out_channel = channels[2**(i - 1)]
            self.from_rgb.append(
                EqualizedLRConvModule(
                    3,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    act_cfg=dict(type='LeakyReLU', negative_slope=0.2)))
            self.convs.append(
                nn.Sequential(
                    EqualizedLRConvModule(
                        in_channels,
                        out_channel,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                        norm_cfg=None,
                        act_cfg=dict(type='LeakyReLU', negative_slope=0.2)),
                    Blur(blur_kernel, pad=(1, 1)),
                    EqualizedLRConvDownModule(
                        out_channel,
                        out_channel,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act_cfg=None),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)))

            in_channels = out_channel

        self.from_rgb.append(
            EqualizedLRConvModule(
                3,
                in_channels,
                kernel_size=3,
                padding=0,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.2)))
        self.convs.append(
            nn.Sequential(
                EqualizedLRConvModule(
                    in_channels + 1,
                    512,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                    norm_cfg=None,
                    act_cfg=dict(type='LeakyReLU', negative_slope=0.2)),
                EqualizedLRConvModule(
                    512,
                    512,
                    kernel_size=4,
                    padding=0,
                    bias=True,
                    norm_cfg=None,
                    act_cfg=None),
            ))

        if self.with_mbstd:
            self.mbstd_layer = MiniBatchStddevLayer(**mbstd_cfg)

        self.final_linear = nn.Sequential(EqualLinearActModule(channels[4], 1))

        self.n_layer = len(self.convs)

    def forward(self, input, transition_weight=1., curr_scale=-1):
        """Forward function.

        Args:
            input (torch.Tensor): Input image tensor.
            transition_weight (float, optional): The weight used in resolution
                transition. Defaults to 1..
            curr_scale (int, optional): The resolution scale of image tensor.
                -1 means the max resolution scale of the StyleGAN1.
                Defaults to -1.

        Returns:
            torch.Tensor: Predict score for the input image.
        """
        curr_log_size = self.log_size if curr_scale < 0 else int(
            np.log2(curr_scale))
        step = curr_log_size - 2
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            # minibatch standard deviation
            if i == 0:
                out = self.mbstd_layer(out)

            out = self.convs[index](out)

            if i > 0:
                if i == step and 0 <= transition_weight < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - transition_weight
                           ) * skip_rgb + transition_weight * out

        out = out.view(out.shape[0], -1)
        out = self.final_linear(out)
        return out

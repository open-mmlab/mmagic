# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmagic.registry import MODELS
from .wgan_gp_module import ConvLNModule, WGANDecisionHead


@MODELS.register_module()
class WGANGPDiscriminator(BaseModule):
    r"""Discriminator for WGANGP.

    Implementation Details for WGANGP discriminator the same as training
    configuration (a) described in PGGAN paper:
    PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION
    https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf # noqa

    #. Adopt convolution architecture specified in appendix A.2;
    #. Add layer normalization to all conv3x3 and conv4x4 layers;
    #. Use LeakyReLU in the discriminator except for the final output layer;
    #. Initialize all weights using He’s initializer.

    Args:
        in_channel (int): The channel number of the input image.
        in_scale (int): The scale of the input image.
        conv_module_cfg (dict, optional): Config for the convolution module
            used in this discriminator. Defaults to None.
        init_cfg (dict, optional): Initialization config dict.
    """
    _default_channels_per_scale = {
        '4': 512,
        '8': 512,
        '16': 256,
        '32': 128,
        '64': 64,
        '128': 32
    }
    _default_conv_module_cfg = dict(
        conv_cfg=None,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        norm_cfg=dict(type='LN2d'),
        order=('conv', 'norm', 'act'))

    _default_upsample_cfg = dict(type='nearest', scale_factor=2)

    def __init__(self,
                 in_channel,
                 in_scale,
                 conv_module_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # set initial params
        self.in_channel = in_channel
        self.in_scale = in_scale
        self.conv_module_cfg = deepcopy(self._default_conv_module_cfg)
        if conv_module_cfg is not None:
            self.conv_module_cfg.update(conv_module_cfg)
        # set from_rgb head
        self.from_rgb = ConvModule(
            3,
            kernel_size=1,
            out_channels=self._default_channels_per_scale[str(self.in_scale)],
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2))
        # set conv_blocks
        self.conv_blocks = nn.ModuleList()

        log2scale = int(np.log2(self.in_scale))
        for i in range(log2scale, 2, -1):
            self.conv_blocks.append(
                ConvLNModule(
                    self._default_channels_per_scale[str(2**i)],
                    self._default_channels_per_scale[str(2**i)],
                    feature_shape=(self._default_channels_per_scale[str(2**i)],
                                   2**i, 2**i),
                    **self.conv_module_cfg))
            self.conv_blocks.append(
                ConvLNModule(
                    self._default_channels_per_scale[str(2**i)],
                    self._default_channels_per_scale[str(2**(i - 1))],
                    feature_shape=(self._default_channels_per_scale[str(
                        2**(i - 1))], 2**i, 2**i),
                    **self.conv_module_cfg))
            self.conv_blocks.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.decision = WGANDecisionHead(
            self._default_channels_per_scale['4'],
            self._default_channels_per_scale['4'],
            1,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
            norm_cfg=self.conv_module_cfg['norm_cfg'])

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Fake or real image tensor.

        Returns:
            torch.Tensor: Prediction for the reality of the input image.
        """
        # noise vector to 2D feature
        x = self.from_rgb(x)
        for conv in self.conv_blocks:
            x = conv(x)
        x = self.decision(x)
        return x

# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from functools import partial

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mmedit.registry import MODULES
from .pggan_modules import (EqualizedLRConvDownModule, EqualizedLRConvModule,
                            MiniBatchStddevLayer, PGGANDecisionHead)


@MODULES.register_module()
class PGGANDiscriminator(nn.Module):
    """Discriminator for PGGAN.

    Args:
        in_scale (int): The scale of the input image.
        label_size (int, optional): Size of the label vector. Defaults to
            0.
        base_channels (int, optional): The basic channel number of the
            generator. The other layers contains channels based on this
            number. Defaults to 8192.
        max_channels (int, optional): Maximum channels for the feature
            maps in the discriminator block. Defaults to 512.
        in_channels (int, optional): Number of channels in input images.
            Defaults to 3.
        channel_decay (float, optional): Decay for channels of feature
            maps. Defaults to 1.0.
        mbstd_cfg (dict, optional): Configs for minibatch-stddev layer.
            Defaults to dict(group_size=4).
        fused_convdown (bool, optional): Whether use fused downconv.
            Defaults to True.
        conv_module_cfg (dict, optional): Config for the convolution
            module used in this generator. Defaults to None.
        fused_convdown_cfg (dict, optional): Config for the fused downconv
            module used in this discriminator. Defaults to None.
        fromrgb_layer_cfg (dict, optional): Config for the fromrgb layer.
            Defaults to None.
        downsample_cfg (dict, optional): Config for the downsampling
            operation. Defaults to None.
    """
    _default_fromrgb_cfg = dict(
        conv_cfg=None,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        norm_cfg=None,
        order=('conv', 'act', 'norm'))

    _default_conv_module_cfg = dict(
        kernel_size=3,
        padding=1,
        stride=1,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2))

    _default_convdown_cfg = dict(
        kernel_size=3,
        padding=1,
        stride=2,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2))

    def __init__(self,
                 in_scale,
                 label_size=0,
                 base_channels=8192,
                 max_channels=512,
                 in_channels=3,
                 channel_decay=1.0,
                 mbstd_cfg=dict(group_size=4),
                 fused_convdown=True,
                 conv_module_cfg=None,
                 fused_convdown_cfg=None,
                 fromrgb_layer_cfg=None,
                 downsample_cfg=None):
        super().__init__()
        self.in_scale = in_scale
        self.in_log2_scale = int(np.log2(self.in_scale))
        self.label_size = label_size
        self.base_channels = base_channels
        self.max_channels = max_channels
        self.in_channels = in_channels
        self.channel_decay = channel_decay
        self.with_mbstd = mbstd_cfg is not None

        self.fused_convdown = fused_convdown

        self.conv_module_cfg = deepcopy(self._default_conv_module_cfg)
        if conv_module_cfg is not None:
            self.conv_module_cfg.update(conv_module_cfg)

        if self.fused_convdown:
            self.fused_convdown_cfg = deepcopy(self._default_convdown_cfg)
            if fused_convdown_cfg is not None:
                self.fused_convdown_cfg.update(fused_convdown_cfg)

        self.fromrgb_layer_cfg = deepcopy(self._default_fromrgb_cfg)
        if fromrgb_layer_cfg:
            self.fromrgb_layer_cfg.update(fromrgb_layer_cfg)

        # setup conv blocks
        self.conv_blocks = nn.ModuleList()
        self.fromrgb_layers = nn.ModuleList()

        for s in range(2, self.in_log2_scale + 1):
            self.fromrgb_layers.append(
                self._get_fromrgb_layer(self.in_channels, s))

            self.conv_blocks.extend(
                self._get_convdown_block(self._num_out_channels(s - 1), s))

        # setup downsample layer
        self.downsample_cfg = deepcopy(downsample_cfg)
        if self.downsample_cfg is None or self.downsample_cfg.get(
                'type', None) == 'avgpool':
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        elif self.downsample_cfg.get('type', None) in ['nearest', 'bilinear']:
            self.downsample = partial(
                F.interpolate,
                mode=self.downsample_cfg.pop('type'),
                **self.downsample_cfg)
        else:
            raise NotImplementedError(
                'We have not supported the downsampling with type'
                f' {downsample_cfg}.')

        # setup minibatch stddev layer
        if self.with_mbstd:
            self.mbstd_layer = MiniBatchStddevLayer(**mbstd_cfg)
            # minibatch stddev layer will concatenate an additional feature map
            # in channel dimension.
            decision_in_channels = self._num_out_channels(1) * 16 + 16
        else:
            decision_in_channels = self._num_out_channels(1) * 16

        # setup decision layer
        self.decision = PGGANDecisionHead(decision_in_channels,
                                          self._num_out_channels(0),
                                          1 + self.label_size)

    def _num_out_channels(self, log_scale: int) -> int:
        """Calculate the number of output channels of the current network from
        logarithm of current scale.

        Args:
            log_scale (int): The logarithm of the current scale.

        Returns:
            int: The number of output channels.
        """
        return min(
            int(self.base_channels / (2.0**(log_scale * self.channel_decay))),
            self.max_channels)

    def _get_fromrgb_layer(self, in_channels: int,
                           log2_scale: int) -> nn.Module:
        """Get the 'fromrgb' layer from logarithm of current scale.

        Args:
            in_channels (int): The number of input channels.
            log2_scale (int): The logarithm of the current scale.

        Returns:
            nn.Module: The built from-rgb layer.
        """
        return EqualizedLRConvModule(in_channels,
                                     self._num_out_channels(log2_scale - 1),
                                     **self.fromrgb_layer_cfg)

    def _get_convdown_block(self, in_channels: int,
                            log2_scale: int) -> nn.Module:
        """Get the downsample layer from logarithm of current scale.

        Args:
            in_channels (int): The number of input channels.
            log2_scale (int): The logarithm of the current scale.

        Returns:
            nn.Module: The built Conv layer.
        """
        modules = []
        if log2_scale == 2:
            modules.append(
                EqualizedLRConvModule(in_channels,
                                      self._num_out_channels(log2_scale - 1),
                                      **self.conv_module_cfg))
        else:
            modules.append(
                EqualizedLRConvModule(in_channels,
                                      self._num_out_channels(log2_scale - 1),
                                      **self.conv_module_cfg))

            if self.fused_convdown:
                cfg_ = dict(downsample=dict(type='fused_pool'))
                cfg_.update(self.fused_convdown_cfg)
            else:
                cfg_ = dict(downsample=self.downsample)
                cfg_.update(self.conv_module_cfg)
            modules.append(
                EqualizedLRConvDownModule(
                    self._num_out_channels(log2_scale - 1),
                    self._num_out_channels(log2_scale - 2), **cfg_))
        return modules

    def forward(self, x, transition_weight=1., curr_scale=-1):
        """Forward function.

        Args:
            x (torch.Tensor): Input image tensor.
            transition_weight (float, optional): The weight used in resolution
                transition. Defaults to 1.0.
            curr_scale (int, optional): The scale for the current inference or
                training. Defaults to -1.

        Returns:
            Tensor: Predict score for the input image.
        """
        curr_log2_scale = self.in_log2_scale if curr_scale < 4 else int(
            np.log2(curr_scale))

        original_img = x

        x = self.fromrgb_layers[curr_log2_scale - 2](x)

        for s in range(curr_log2_scale, 2, -1):
            x = self.conv_blocks[2 * s - 5](x)
            x = self.conv_blocks[2 * s - 4](x)
            if s == curr_log2_scale:
                img_down = self.downsample(original_img)
                y = self.fromrgb_layers[curr_log2_scale - 3](img_down)
                x = y + transition_weight * (x - y)

        if self.with_mbstd:
            x = self.mbstd_layer(x)

        x = self.decision(x)

        if self.label_size > 0:
            return x[:, :1], x[:, 1:]

        return x

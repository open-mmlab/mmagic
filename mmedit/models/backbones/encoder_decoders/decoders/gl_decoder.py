# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class GLDecoder(nn.Module):
    """Decoder used in Global&Local model.

    This implementation follows:
    Globally and locally Consistent Image Completion

    Args:
        in_channels (int): Channel number of input feature.
        norm_cfg (dict): Config dict to build norm layer.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
        out_act (str): Output activation type, "clip" by default. Noted that
            in our implementation, we clip the output with range [-1, 1].
    """

    def __init__(self,
                 in_channels=256,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 out_act='clip'):
        super().__init__()
        self.dec1 = ConvModule(
            in_channels,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.dec2 = ConvModule(
            256,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.dec3 = ConvModule(
            256,
            128,
            kernel_size=4,
            stride=2,
            padding=1,
            conv_cfg=dict(type='Deconv'),
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.dec4 = ConvModule(
            128,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.dec5 = ConvModule(
            128,
            64,
            kernel_size=4,
            stride=2,
            padding=1,
            conv_cfg=dict(type='Deconv'),
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.dec6 = ConvModule(
            64,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.dec7 = ConvModule(
            32,
            3,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=None,
            act_cfg=None)

        if out_act == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif out_act == 'clip':
            self.output_act = partial(torch.clamp, min=-1, max=1.)
        else:
            raise ValueError(
                f'{out_act} activation for output has not be supported.')

    def forward(self, x):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """
        for i in range(7):
            x = getattr(self, f'dec{i + 1}')(x)
        x = self.output_act(x)
        return x

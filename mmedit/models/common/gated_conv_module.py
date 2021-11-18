# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_activation_layer


class SimpleGatedConvModule(nn.Module):
    """Simple Gated Convolutional Module.

    This module is a simple gated convolutional module. The detailed formula
    is:

    .. math::
        y = \\phi(conv1(x)) * \\sigma(conv2(x)),

    where `phi` is the feature activation function and `sigma` is the gate
    activation function. In default, the gate activation function is sigmoid.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): The number of channels of the output feature. Note
            that `out_channels` in the conv module is doubled since this module
            contains two convolutions for feature and gate separately.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        feat_act_cfg (dict): Config dict for feature activation layer.
        gate_act_cfg (dict): Config dict for gate activation layer.
        kwargs (keyword arguments): Same as `ConvModule`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 feat_act_cfg=dict(type='ELU'),
                 gate_act_cfg=dict(type='Sigmoid'),
                 **kwargs):
        super().__init__()
        # the activation function should specified outside conv module
        kwargs_ = copy.deepcopy(kwargs)
        kwargs_['act_cfg'] = None
        self.with_feat_act = feat_act_cfg is not None
        self.with_gate_act = gate_act_cfg is not None

        self.conv = ConvModule(in_channels, out_channels * 2, kernel_size,
                               **kwargs_)

        if self.with_feat_act:
            self.feat_act = build_activation_layer(feat_act_cfg)

        if self.with_gate_act:
            self.gate_act = build_activation_layer(gate_act_cfg)

    def forward(self, x):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """
        x = self.conv(x)
        x, gate = torch.split(x, x.size(1) // 2, dim=1)
        if self.with_feat_act:
            x = self.feat_act(x)
        if self.with_gate_act:
            gate = self.gate_act(gate)
        x = x * gate

        return x

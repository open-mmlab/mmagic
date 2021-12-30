# Copyright (c) OpenMMLab. All rights reserved.
import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer

from mmedit.models.common import SimpleGatedConvModule
from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class DeepFillDecoder(nn.Module):
    """Decoder used in DeepFill model.

    This implementation follows:
    Generative Image Inpainting with Contextual Attention

    Args:
        in_channels (int): The number of input channels.
        conv_type (str): The type of conv module. In DeepFillv1 model, the
            `conv_type` should be 'conv'. In DeepFillv2 model, the `conv_type`
            should be 'gated_conv'.
        norm_cfg (dict): Config dict to build norm layer. Default: None.
        act_cfg (dict): Config dict for activation layer, "elu" by default.
        out_act_cfg (dict): Config dict for output activation layer. Here, we
            provide commonly used `clamp` or `clip` operation.
        channel_factor (float): The scale factor for channel size.
            Default: 1.
        kwargs (keyword arguments).
    """
    _conv_type = dict(conv=ConvModule, gated_conv=SimpleGatedConvModule)

    def __init__(self,
                 in_channels,
                 conv_type='conv',
                 norm_cfg=None,
                 act_cfg=dict(type='ELU'),
                 out_act_cfg=dict(type='clip', min=-1., max=1.),
                 channel_factor=1.,
                 **kwargs):
        super().__init__()
        self.with_out_activation = out_act_cfg is not None

        conv_module = self._conv_type[conv_type]
        channel_list = [128, 128, 64, 64, 32, 16, 3]
        channel_list = [int(x * channel_factor) for x in channel_list]
        # dirty code for assign output channel with 3
        channel_list[-1] = 3
        for i in range(7):
            kwargs_ = copy.deepcopy(kwargs)
            if i == 6:
                act_cfg = None
                if conv_type == 'gated_conv':
                    kwargs_['feat_act_cfg'] = None
            self.add_module(
                f'dec{i + 1}',
                conv_module(
                    in_channels,
                    channel_list[i],
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **kwargs_))
            in_channels = channel_list[i]

        if self.with_out_activation:
            act_type = out_act_cfg['type']
            if act_type == 'clip':
                act_cfg_ = copy.deepcopy(out_act_cfg)
                act_cfg_.pop('type')
                self.out_act = partial(torch.clamp, **act_cfg_)
            else:
                self.out_act = build_activation_layer(out_act_cfg)

    def forward(self, input_dict):
        """Forward Function.

        Args:
            input_dict (dict | torch.Tensor): Input dict with middle features
                or torch.Tensor.

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h, w).
        """
        if isinstance(input_dict, dict):
            x = input_dict['out']
        else:
            x = input_dict
        for i in range(7):
            x = getattr(self, f'dec{i + 1}')(x)
            if i in (1, 3):
                x = F.interpolate(x, scale_factor=2)

        if self.with_out_activation:
            x = self.out_act(x)
        return x

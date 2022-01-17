# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmedit.models.common import MaskConvModule
from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class PConvDecoder(nn.Module):
    """Decoder with partial conv.

    About the details for this architecture, pls see:
    Image Inpainting for Irregular Holes Using Partial Convolutions

    Args:
        num_layers (int): The number of convolutional layers. Default: 7.
        interpolation (str): The upsample mode. Default: 'nearest'.
        conv_cfg (dict): Config for convolution module. Default:
            {'type': 'PConv', 'multi_channel': True}.
        norm_cfg (dict): Config for norm layer. Default:
            {'type': 'BN'}.
    """

    def __init__(self,
                 num_layers=7,
                 interpolation='nearest',
                 conv_cfg=dict(type='PConv', multi_channel=True),
                 norm_cfg=dict(type='BN')):
        super().__init__()
        self.num_layers = num_layers
        self.interpolation = interpolation

        for i in range(4, num_layers):
            name = f'dec{i+1}'
            self.add_module(
                name,
                MaskConvModule(
                    512 + 512,
                    512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='LeakyReLU', negative_slope=0.2)))

        self.dec4 = MaskConvModule(
            512 + 256,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2))

        self.dec3 = MaskConvModule(
            256 + 128,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2))

        self.dec2 = MaskConvModule(
            128 + 64,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2))

        self.dec1 = MaskConvModule(
            64 + 3,
            3,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=None,
            act_cfg=None)

    def forward(self, input_dict):
        """Forward Function.

        Args:
            input_dict (dict | torch.Tensor): Input dict with middle features
                or torch.Tensor.

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h, w).
        """
        hidden_feats = input_dict['hidden_feats']
        hidden_masks = input_dict['hidden_masks']
        h_key = 'h{:d}'.format(self.num_layers)
        h, h_mask = hidden_feats[h_key], hidden_masks[h_key]

        for i in range(self.num_layers, 0, -1):
            enc_h_key = f'h{i-1}'
            dec_l_key = f'dec{i}'

            h = F.interpolate(h, scale_factor=2, mode=self.interpolation)
            h_mask = F.interpolate(
                h_mask, scale_factor=2, mode=self.interpolation)

            h = torch.cat([h, hidden_feats[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, hidden_masks[enc_h_key]], dim=1)

            h, h_mask = getattr(self, dec_l_key)(h, h_mask)

        return h, h_mask

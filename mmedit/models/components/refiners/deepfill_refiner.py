# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmedit.models.builder import build_component
from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class DeepFillRefiner(nn.Module):
    """Refiner used in DeepFill model.

    This implementation follows:
    Generative Image Inpainting with Contextual Attention.

    Args:
        encoder_attention (dict): Config dict for encoder used in branch
            with contextual attention module.
        encoder_conv (dict): Config dict for encoder used in branch with
            just convolutional operation.
        dilation_neck (dict): Config dict for dilation neck in branch with
            just convolutional operation.
        contextual_attention (dict): Config dict for contextual attention
            neck.
        decoder (dict): Config dict for decoder used to fuse and decode
            features.
    """

    def __init__(self,
                 encoder_attention=dict(
                     type='DeepFillEncoder', encoder_type='stage2_attention'),
                 encoder_conv=dict(
                     type='DeepFillEncoder', encoder_type='stage2_conv'),
                 dilation_neck=dict(
                     type='GLDilationNeck',
                     in_channels=128,
                     act_cfg=dict(type='ELU')),
                 contextual_attention=dict(
                     type='ContextualAttentionNeck', in_channels=128),
                 decoder=dict(type='DeepFillDecoder', in_channels=256)):
        super().__init__()
        self.encoder_attention = build_component(encoder_attention)
        self.encoder_conv = build_component(encoder_conv)
        self.contextual_attention_neck = build_component(contextual_attention)
        self.dilation_neck = build_component(dilation_neck)
        self.decoder = build_component(decoder)

    def forward(self, x, mask):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).
            mask (torch.Tensor): Input tensor with shape of (n, 1, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """
        # conv branch
        encoder_dict = self.encoder_conv(x)
        conv_x = self.dilation_neck(encoder_dict['out'])

        # contextual attention branch
        attention_x = self.encoder_attention(x)['out']
        h_x, w_x = attention_x.shape[-2:]
        # resale mask to a smaller size
        resized_mask = F.interpolate(mask, size=(h_x, w_x))
        attention_x, offset = self.contextual_attention_neck(
            attention_x, resized_mask)

        # concat two branches
        x = torch.cat([conv_x, attention_x], dim=1)
        x = self.decoder(dict(out=x))

        return x, offset

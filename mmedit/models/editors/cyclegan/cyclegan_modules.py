# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule


class ResidualBlockWithDropout(nn.Module):
    """Define a Residual Block with dropout layers.

    Ref:
    Deep Residual Learning for Image Recognition

    A residual block is a conv block with skip connections. A dropout layer is
    added between two common conv modules.

    Args:
        channels (int): Number of channels in the conv layer.
        padding_mode (str): The name of padding layer:
            'reflect' | 'replicate' | 'zeros'.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='IN')`.
        use_dropout (bool): Whether to use dropout layers. Default: True.
    """

    def __init__(self,
                 channels,
                 padding_mode,
                 norm_cfg=dict(type='BN'),
                 use_dropout=True):
        super().__init__()
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the residual block with dropout layers.
        # Only for IN, use bias to follow cyclegan's original implementation.
        use_bias = norm_cfg['type'] == 'IN'

        block = [
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm_cfg=norm_cfg,
                padding_mode=padding_mode)
        ]

        if use_dropout:
            block += [nn.Dropout(0.5)]

        block += [
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm_cfg=norm_cfg,
                act_cfg=None,
                padding_mode=padding_mode)
        ]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        """Forward function. Add skip connections without final ReLU.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = x + self.block(x)
        return out

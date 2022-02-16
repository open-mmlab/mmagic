# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint

from mmedit.models.common import generation_init_weights
from mmedit.models.registry import COMPONENTS
from mmedit.utils import get_root_logger


@COMPONENTS.register_module()
class SoftMaskPatchDiscriminator(nn.Module):
    """A Soft Mask-Guided PatchGAN discriminator.

    Args:
        in_channels (int): Number of channels in input images.
        base_channels (int, optional): Number of channels at the
            first conv layer. Default: 64.
        num_conv (int, optional): Number of stacked intermediate convs
            (excluding input and output conv). Default: 3.
        norm_cfg (dict, optional): Config dict to build norm layer.
            Default: None.
        init_cfg (dict, optional): Config dict for initialization.
            `type`: The name of our initialization method. Default: 'normal'.
            `gain`: Scaling factor for normal, xavier and orthogonal.
            Default: 0.02.
        with_spectral_norm (bool, optional): Whether use spectral norm
            after the conv layers. Default: False.
    """

    def __init__(self,
                 in_channels,
                 base_channels=64,
                 num_conv=3,
                 norm_cfg=None,
                 init_cfg=dict(type='normal', gain=0.02),
                 with_spectral_norm=False):
        super().__init__()

        kernel_size = 4
        padding = 1

        # input layer
        sequence = [
            ConvModule(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                with_spectral_norm=with_spectral_norm)
        ]

        # stacked intermediate layers,
        # gradually increasing the number of filters
        multiplier_in = 1
        multiplier_out = 1
        for n in range(1, num_conv):
            multiplier_in = multiplier_out
            multiplier_out = min(2**n, 8)
            sequence += [
                ConvModule(
                    in_channels=base_channels * multiplier_in,
                    out_channels=base_channels * multiplier_out,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    bias=False,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                    with_spectral_norm=with_spectral_norm)
            ]
        multiplier_in = multiplier_out
        multiplier_out = min(2**num_conv, 8)
        sequence += [
            ConvModule(
                in_channels=base_channels * multiplier_in,
                out_channels=base_channels * multiplier_out,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                with_spectral_norm=with_spectral_norm)
        ]

        # output one-channel prediction map
        sequence += [
            nn.Conv2d(
                base_channels * multiplier_out,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
        ]

        self.model = nn.Sequential(*sequence)
        self.init_type = 'normal' if init_cfg is None else init_cfg.get(
            'type', 'normal')
        self.init_gain = 0.02 if init_cfg is None else init_cfg.get(
            'gain', 0.02)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        return self.model(x)

    def init_weights(self, pretrained=None):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            generation_init_weights(
                self, init_type=self.init_type, init_gain=self.init_gain)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')

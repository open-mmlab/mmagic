# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmedit.models.base_archs import pixel_unshuffle
from mmedit.models.utils import default_init_weights, make_layer
from mmedit.registry import BACKBONES


@BACKBONES.register_module()
class RRDBNet(BaseModule):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN and Real-ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data. # noqa: E501
    Currently, it supports [x1/x2/x4] upsampling scale factor.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (int): Block number in the trunk network. Defaults: 23
        growth_channels (int): Channels for each growth. Default: 32.
        upscale_factor (int): Upsampling factor. Support x1, x2 and x4.
            Default: 4.
    """
    _supported_upscale_factors = [1, 2, 4]

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 num_blocks=23,
                 growth_channels=32,
                 upscale_factor=4,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if upscale_factor in self._supported_upscale_factors:
            in_channels = in_channels * ((4 // upscale_factor)**2)
        else:
            raise ValueError(f'Unsupported scale factor {upscale_factor}. '
                             f'Currently supported ones are '
                             f'{self._supported_upscale_factors}.')

        self.upscale_factor = upscale_factor
        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.body = make_layer(
            RRDB,
            num_blocks,
            mid_channels=mid_channels,
            growth_channels=growth_channels)
        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.upscale_factor in [1, 2]:
            feat = pixel_unshuffle(x, scale=4 // self.upscale_factor)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # upsample
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

    def init_weights(self):
        """Init weights for models."""
        if self.init_cfg:
            super().init_weights()
        else:
            # Use smaller std for better stability and performance. We
            # use 0.1. See more details in "ESRGAN: Enhanced Super-Resolution
            # Generative Adversarial Networks"
            for m in [
                    self.conv_first, self.conv_body, self.conv_up1,
                    self.conv_up2, self.conv_hr, self.conv_last
            ]:
                default_init_weights(m, 0.1)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        growth_channels (int): Channels for each growth. Default: 32.
    """

    def __init__(self, mid_channels=64, growth_channels=32):
        super().__init__()
        for i in range(5):
            out_channels = mid_channels if i == 4 else growth_channels
            self.add_module(
                f'conv{i+1}',
                nn.Conv2d(mid_channels + i * growth_channels, out_channels, 3,
                          1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.init_weights()

    def init_weights(self):
        """Init weights for ResidualDenseBlock.

        Use smaller std for better stability and performance. We empirically
        use 0.1. See more details in "ESRGAN: Enhanced Super-Resolution
        Generative Adversarial Networks"
        """
        for i in range(5):
            default_init_weights(getattr(self, f'conv{i+1}'), 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth. Default: 32.
    """

    def __init__(self, mid_channels, growth_channels=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(mid_channels, growth_channels)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x

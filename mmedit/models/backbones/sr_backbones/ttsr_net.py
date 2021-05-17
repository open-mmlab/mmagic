from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer

from mmedit.models.common import ResidualBlockNoBN, make_layer

# Use partial to specify some default arguments
_norm_conv_layer = partial(
    build_conv_layer, dict(type='Conv2d'), kernel_size=3, padding=1)
_bottleneck_layer = partial(
    build_conv_layer, dict(type='Conv2d'), kernel_size=1, padding=0)


class SFE(nn.Module):
    """Structural Feature Encoder

    Backbone of Texture Transformer Network for Image Super-Resolution.

    Args:
        in_channels (int): Number of channels in the input image
        mid_channels (int): Channel number of intermediate features
        num_blocks (int): Block number in the trunk network
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
    """

    def __init__(self, in_channels, mid_channels, num_blocks, res_scale):
        super().__init__()

        self.num_blocks = num_blocks
        self.conv_first = _norm_conv_layer(in_channels, mid_channels)

        self.body = make_layer(
            ResidualBlockNoBN,
            num_blocks,
            mid_channels=mid_channels,
            res_scale=res_scale)

        self.conv_last = _norm_conv_layer(mid_channels, mid_channels)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        x1 = x = F.relu(self.conv_first(x))
        x = self.body(x)
        x = self.conv_last(x)
        x = x + x1
        return x


class CSFI2(nn.Module):
    """Cross-Scale Feature Integration, include 2 scales.

    Cross-Scale Feature Integration in Texture Transformer Network for
        Image Super-Resolution.

    Args:
        mid_channels (int): Channel number of intermediate features
    """

    def __init__(self, mid_channels):
        super().__init__()
        self.conv12 = _bottleneck_layer(mid_channels, mid_channels)
        self.conv21 = _norm_conv_layer(mid_channels, mid_channels, stride=2)

        self.conv_merge1 = _norm_conv_layer(mid_channels * 2, mid_channels)
        self.conv_merge2 = _norm_conv_layer(mid_channels * 2, mid_channels)

    def forward(self, x1, x2):
        """Forward function.

        Args:
            x1 (Tensor): Input tensor with shape (n, c, h, w).
            x2 (Tensor): Input tensor with shape (n, c, 2h, 2w).

        Returns:
            x1 (Tensor): Output tensor with shape (n, c, h, w).
            x2 (Tensor): Output tensor with shape (n, c, 2h, 2w).
        """

        x12 = F.interpolate(
            x1, scale_factor=2, mode='bicubic', align_corners=False)
        x12 = F.relu(self.conv12(x12))
        x21 = F.relu(self.conv21(x2))

        x1 = F.relu(self.conv_merge1(torch.cat((x1, x21), dim=1)))
        x2 = F.relu(self.conv_merge2(torch.cat((x2, x12), dim=1)))

        return x1, x2


class CSFI3(nn.Module):
    """Cross-Scale Feature Integration, include 3 scales.

    Cross-Scale Feature Integration in Texture Transformer Network for
        Image Super-Resolution.

    Args:
        mid_channels (int): Channel number of intermediate features
    """

    def __init__(self, mid_channels):
        super().__init__()
        self.conv12 = _bottleneck_layer(mid_channels, mid_channels)
        self.conv13 = _bottleneck_layer(mid_channels, mid_channels)

        self.conv21 = _norm_conv_layer(mid_channels, mid_channels, stride=2)
        self.conv23 = _bottleneck_layer(mid_channels, mid_channels)

        self.conv31_1 = _norm_conv_layer(mid_channels, mid_channels, stride=2)
        self.conv31_2 = _norm_conv_layer(mid_channels, mid_channels, stride=2)
        self.conv32 = _norm_conv_layer(mid_channels, mid_channels, stride=2)

        self.conv_merge1 = _norm_conv_layer(mid_channels * 3, mid_channels)
        self.conv_merge2 = _norm_conv_layer(mid_channels * 3, mid_channels)
        self.conv_merge3 = _norm_conv_layer(mid_channels * 3, mid_channels)

    def forward(self, x1, x2, x3):
        """Forward function.

        Args:
            x1 (Tensor): Input tensor with shape (n, c, h, w).
            x2 (Tensor): Input tensor with shape (n, c, 2h, 2w).
            x3 (Tensor): Input tensor with shape (n, c, 4h, 4w).

        Returns:
            x1 (Tensor): Output tensor with shape (n, c, h, w).
            x2 (Tensor): Output tensor with shape (n, c, 2h, 2w).
            x3 (Tensor): Output tensor with shape (n, c, 4h, 4w).
        """

        x12 = F.interpolate(
            x1, scale_factor=2, mode='bicubic', align_corners=False)
        x12 = F.relu(self.conv12(x12))
        x13 = F.interpolate(
            x1, scale_factor=4, mode='bicubic', align_corners=False)
        x13 = F.relu(self.conv13(x13))

        x21 = F.relu(self.conv21(x2))
        x23 = F.interpolate(
            x2, scale_factor=2, mode='bicubic', align_corners=False)
        x23 = F.relu(self.conv23(x23))

        x31 = F.relu(self.conv31_1(x3))
        x31 = F.relu(self.conv31_2(x31))
        x32 = F.relu(self.conv32(x3))

        x1 = F.relu(self.conv_merge1(torch.cat((x1, x21, x31), dim=1)))
        x2 = F.relu(self.conv_merge2(torch.cat((x2, x12, x32), dim=1)))
        x3 = F.relu(self.conv_merge3(torch.cat((x3, x13, x23), dim=1)))

        return x1, x2, x3

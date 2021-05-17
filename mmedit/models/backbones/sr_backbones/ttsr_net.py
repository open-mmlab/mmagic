from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer

from mmedit.models.common import ResidualBlockNoBN, make_layer

# Use partial to specify some default arguments
_conv3x3_layer = partial(
    build_conv_layer, dict(type='Conv2d'), kernel_size=3, padding=1)
_conv1x1_layer = partial(
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
        self.conv_first = _conv3x3_layer(in_channels, mid_channels)

        self.body = make_layer(
            ResidualBlockNoBN,
            num_blocks,
            mid_channels=mid_channels,
            res_scale=res_scale)

        self.conv_last = _conv3x3_layer(mid_channels, mid_channels)

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
    """Cross-Scale Feature Integration between 1x and 2x features.

    Cross-Scale Feature Integration in Texture Transformer Network for
        Image Super-Resolution.
    It is cross-scale feature integration between 1x and 2x features.
        For example, `conv2to1` means conv layer from 2x feature to 1x
        feature. Down-sampling is achieved by conv layer with stride=2,
        and up-sampling is achieved by bicubic interpolate and conv layer.

    Args:
        mid_channels (int): Channel number of intermediate features
    """

    def __init__(self, mid_channels):
        super().__init__()
        self.conv1to2 = _conv1x1_layer(mid_channels, mid_channels)
        self.conv2to1 = _conv3x3_layer(mid_channels, mid_channels, stride=2)

        self.conv_merge1 = _conv3x3_layer(mid_channels * 2, mid_channels)
        self.conv_merge2 = _conv3x3_layer(mid_channels * 2, mid_channels)

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
        x12 = F.relu(self.conv1to2(x12))
        x21 = F.relu(self.conv2to1(x2))

        x1 = F.relu(self.conv_merge1(torch.cat((x1, x21), dim=1)))
        x2 = F.relu(self.conv_merge2(torch.cat((x2, x12), dim=1)))

        return x1, x2


class CSFI3(nn.Module):
    """Cross-Scale Feature Integration between 1x, 2x, and 4x features.

    Cross-Scale Feature Integration in Texture Transformer Network for
        Image Super-Resolution.
    It is cross-scale feature integration between 1x and 2x features.
        For example, `conv2to1` means conv layer from 2x feature to 1x
        feature. Down-sampling is achieved by conv layer with stride=2,
        and up-sampling is achieved by bicubic interpolate and conv layer.

    Args:
        mid_channels (int): Channel number of intermediate features
    """

    def __init__(self, mid_channels):
        super().__init__()
        self.conv1to2 = _conv1x1_layer(mid_channels, mid_channels)
        self.conv1to4 = _conv1x1_layer(mid_channels, mid_channels)

        self.conv2to1 = _conv3x3_layer(mid_channels, mid_channels, stride=2)
        self.conv2to4 = _conv1x1_layer(mid_channels, mid_channels)

        self.conv4to1_1 = _conv3x3_layer(mid_channels, mid_channels, stride=2)
        self.conv4to1_2 = _conv3x3_layer(mid_channels, mid_channels, stride=2)
        self.conv4to2 = _conv3x3_layer(mid_channels, mid_channels, stride=2)

        self.conv_merge1 = _conv3x3_layer(mid_channels * 3, mid_channels)
        self.conv_merge2 = _conv3x3_layer(mid_channels * 3, mid_channels)
        self.conv_merge4 = _conv3x3_layer(mid_channels * 3, mid_channels)

    def forward(self, x1, x2, x4):
        """Forward function.

        Args:
            x1 (Tensor): Input tensor with shape (n, c, h, w).
            x2 (Tensor): Input tensor with shape (n, c, 2h, 2w).
            x4 (Tensor): Input tensor with shape (n, c, 4h, 4w).

        Returns:
            x1 (Tensor): Output tensor with shape (n, c, h, w).
            x2 (Tensor): Output tensor with shape (n, c, 2h, 2w).
            x4 (Tensor): Output tensor with shape (n, c, 4h, 4w).
        """

        x12 = F.interpolate(
            x1, scale_factor=2, mode='bicubic', align_corners=False)
        x12 = F.relu(self.conv1to2(x12))
        x14 = F.interpolate(
            x1, scale_factor=4, mode='bicubic', align_corners=False)
        x14 = F.relu(self.conv1to4(x14))

        x21 = F.relu(self.conv2to1(x2))
        x24 = F.interpolate(
            x2, scale_factor=2, mode='bicubic', align_corners=False)
        x24 = F.relu(self.conv2to4(x24))

        x41 = F.relu(self.conv4to1_1(x4))
        x41 = F.relu(self.conv4to1_2(x41))
        x42 = F.relu(self.conv4to2(x4))

        x1 = F.relu(self.conv_merge1(torch.cat((x1, x21, x41), dim=1)))
        x2 = F.relu(self.conv_merge2(torch.cat((x2, x12, x42), dim=1)))
        x4 = F.relu(self.conv_merge4(torch.cat((x4, x14, x24), dim=1)))

        return x1, x2, x4


class MergeFeatures(nn.Module):
    """Merge Features. Merge 1x, 2x, and 4x features.

    Final module of Texture Transformer Network for Image Super-Resolution.

    """

    def __init__(self, mid_channels, out_channels):
        super().__init__()
        self.conv1to4 = _conv1x1_layer(mid_channels, mid_channels)
        self.conv2to4 = _conv1x1_layer(mid_channels, mid_channels)
        self.conv_merge = _conv3x3_layer(mid_channels * 3, mid_channels)
        self.conv_last1 = _conv3x3_layer(mid_channels, mid_channels // 2)
        self.conv_last2 = _conv1x1_layer(mid_channels // 2, out_channels)

    def forward(self, x1, x2, x4):
        """Forward function.

        Args:
            x1 (Tensor): Input tensor with shape (n, c, h, w).
            x2 (Tensor): Input tensor with shape (n, c, 2h, 2w).
            x4 (Tensor): Input tensor with shape (n, c, 4h, 4w).

        Returns:
            x (Tensor): Output tensor with shape (n, c_out, 4h, 4w).
        """

        x14 = F.interpolate(
            x1, scale_factor=4, mode='bicubic', align_corners=False)
        x14 = F.relu(self.conv1to4(x14))
        x24 = F.interpolate(
            x2, scale_factor=2, mode='bicubic', align_corners=False)
        x24 = F.relu(self.conv2to4(x24))

        x = F.relu(self.conv_merge(torch.cat((x4, x14, x24), dim=1)))
        x = self.conv_last1(x)
        x = self.conv_last2(x)
        x = torch.clamp(x, -1, 1)

        return x

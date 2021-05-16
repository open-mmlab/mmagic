import torch.nn as nn
import torch.nn.functional as F

from mmedit.models.common import ResidualBlockNoBN, make_layer


def norm_conv_layer(in_channels, out_channels, stride=1):
    """Norm conv layer with kernal_size=3.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        stride (int or tuple, optional): Stride of the convolution. Default: 1

    results:
        conv_layer (Conv2d): Conv layer with kernal_size=3.
    """

    conv_layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True)
    return conv_layer


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
        self.conv_head = norm_conv_layer(in_channels, mid_channels)

        self.body = make_layer(
            ResidualBlockNoBN,
            num_blocks,
            mid_channels=mid_channels,
            res_scale=res_scale)

        self.conv_tail = norm_conv_layer(mid_channels, mid_channels)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        x = F.relu(self.conv_head(x))
        x1 = x
        x = self.body(x)
        x = self.conv_tail(x)
        x = x + x1
        return x

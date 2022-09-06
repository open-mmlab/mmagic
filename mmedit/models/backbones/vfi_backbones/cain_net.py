# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint

from mmedit.models.common import make_layer, pixel_unshuffle
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


def get_padding_functions(x, padding=7):
    """Generate padding function for CAIN.

    This function produces two functions to pad and depad a tensor, given the
    number of pixels to be padded. When applying padding and depadding
    sequentially, the original tensor is obtained.

    The generated padding function will pad the given tensor to the 'padding'
    power of 2, i.e., pow(2, 'padding').

    tensor --padding_function--> padded tensor
    padded tensor --depadding_function--> original tensor

    Args:
        x (Tensor): Input tensor.
        padding (int): Padding size.

    Returns:
        padding_function (Function): Padding function.
        depadding_function (Function): Depadding function.
    """

    h, w = x.shape[-2:]
    padding_width, padding_height = 0, 0
    if w != ((w >> padding) << padding):
        padding_width = (((w >> padding) + 1) << padding) - w
    if h != ((h >> padding) << padding):
        padding_height = (((h >> padding) + 1) << padding) - h
    left, right = padding_width // 2, padding_width - padding_width // 2
    up, down = padding_height // 2, padding_height - padding_height // 2
    # print(up, down, left, right)
    if down >= h or right >= w:
        function = nn.ReplicationPad2d
    else:
        function = nn.ReflectionPad2d
    padding_function = function(padding=[left, right, up, down])
    depadding_function = function(
        padding=[0 - left, 0 - right, 0 - up, 0 - down])
    return padding_function, depadding_function


class ConvNormWithReflectionPad(nn.Module):
    """Apply reflection padding, followed by a convolution, which can be
    followed by an optional normalization.

    Args:
        in_channels (int): Channel number of input features.
        out_channels (int): Channel number of output features.
        kernel_size (int): Kernel size of convolution layer.
        norm (str | None): Normalization layer. If it is None, no
            normalization is performed. Default: None.
    """

    def __init__(self, in_channels, out_channels, kernel_size, norm=None):
        super().__init__()

        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, bias=True)

        if norm is None:
            self.norm = None
        elif norm.lower() == 'in':
            self.norm = nn.InstanceNorm2d(
                out_channels, track_running_stats=True)
        elif norm.lower() == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            raise ValueError(f"Invalid value for 'norm': {norm}")

    def forward(self, x):
        """Forward function for ConvNormWithReflectionPad.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor with shape (n, c, h, w).
        """

        out = self.reflection_pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        return out


class ChannelAttentionLayer(nn.Module):
    """Channel Attention (CA) Layer.

    Args:
        mid_channels (int): Channel number of the intermediate features.
        reduction (int): Channel reduction of CA. Default: 16.
    """

    def __init__(self, mid_channels, reduction=16):
        super().__init__()

        # global average pooling: (n, c, h, w) --> (n, c, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # channel reduction.
        self.channel_attention = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels // reduction,
                1,
                padding=0,
                bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels // reduction,
                mid_channels,
                1,
                padding=0,
                bias=True), nn.Sigmoid())

    def forward(self, x):
        """Forward function for ChannelAttentionLayer.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor with shape (n, c, h, w).
        """

        y = self.avg_pool(x)
        y = self.channel_attention(y)
        return x * y


class ResidualChannelAttention(nn.Module):
    """Residual Channel Attention Module.

    Args:
        mid_channels (int): Channel number of the intermediate features.
        kernel_size (int): Kernel size of convolution layers. Default: 3.
        reduction (int): Channel reduction. Default: 16.
        norm (None | function): Norm layer. If None, no norm layer.
            Default: None.
        act (function): activation function. Default: nn.LeakyReLU(0.2, True).
    """

    def __init__(self,
                 mid_channels,
                 kernel_size=3,
                 reduction=16,
                 norm=None,
                 act=nn.LeakyReLU(0.2, True)):
        super().__init__()

        self.body = nn.Sequential(
            ConvNormWithReflectionPad(
                mid_channels, mid_channels, kernel_size, norm=norm), act,
            ConvNormWithReflectionPad(
                mid_channels, mid_channels, kernel_size, norm=norm),
            ChannelAttentionLayer(mid_channels, reduction))

    def forward(self, x):
        """Forward function for ResidualChannelAttention.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor with shape (n, c, h, w).
        """

        out = self.body(x)
        return out + x


class ResidualGroup(nn.Module):
    """Residual Group, consisting of a stack of residual channel attention,
    followed by a convolution.

    Args:
        block_layer (nn.Module): nn.Module class for basic block.
        num_block_layers (int): number of blocks.
        mid_channels (int): Channel number of the intermediate features.
        kernel_size (int): Kernel size of ResidualGroup.
        reduction (int): Channel reduction of CA. Default: 16.
        act (function): activation function. Default: nn.LeakyReLU(0.2, True).
        norm (str | None): Normalization layer. If it is None, no
            normalization is performed. Default: None.
    """

    def __init__(self,
                 block_layer,
                 num_block_layers,
                 mid_channels,
                 kernel_size,
                 reduction,
                 act=nn.LeakyReLU(0.2, True),
                 norm=None):
        super().__init__()

        self.body = make_layer(
            block_layer,
            num_block_layers,
            mid_channels=mid_channels,
            kernel_size=kernel_size,
            reduction=reduction,
            norm=norm,
            act=act)
        self.conv_after_body = ConvNormWithReflectionPad(
            mid_channels, mid_channels, kernel_size, norm=norm)

    def forward(self, x):
        """Forward function for ResidualGroup.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor with shape (n, c, h, w).
        """

        y = self.body(x)
        y = self.conv_after_body(y)
        return x + y


@BACKBONES.register_module()
class CAINNet(nn.Module):
    """CAIN network structure.

    Paper: Channel Attention Is All You Need for Video Frame Interpolation.
    Ref repo: https://github.com/myungsub/CAIN

    Args:
        in_channels (int): Channel number of inputs. Default: 3.
        kernel_size (int): Kernel size of CAINNet. Default: 3.
        num_block_groups (int): Number of block groups. Default: 5.
        num_block_layers (int): Number of blocks in a group. Default: 12.
        depth (int): Down scale depth, scale = 2**depth. Default: 3.
        reduction (int): Channel reduction of CA. Default: 16.
        norm (str | None): Normalization layer. If it is None, no
            normalization is performed. Default: None.
        padding (int): Padding of CAINNet. Default: 7.
        act (function): activate function. Default: nn.LeakyReLU(0.2, True).
    """

    def __init__(self,
                 in_channels=3,
                 kernel_size=3,
                 num_block_groups=5,
                 num_block_layers=12,
                 depth=3,
                 reduction=16,
                 norm=None,
                 padding=7,
                 act=nn.LeakyReLU(0.2, True)):
        super().__init__()

        mid_channels = in_channels * (4**depth)
        self.scale = 2**depth
        self.padding = padding

        self.conv_first = nn.Conv2d(mid_channels * 2, mid_channels,
                                    kernel_size, 1, 1)
        self.body = make_layer(
            ResidualGroup,
            num_block_groups,
            block_layer=ResidualChannelAttention,
            num_block_layers=num_block_layers,
            mid_channels=mid_channels,
            kernel_size=kernel_size,
            reduction=reduction,
            norm=norm,
            act=act)
        self.conv_last = nn.Conv2d(mid_channels, mid_channels, kernel_size, 1,
                                   1)

    def forward(self, imgs, padding_flag=False):
        """Forward function.

        Args:
            imgs (Tensor): Input tensor with shape (n, 2, c, h, w).
            padding_flag (bool): Padding or not. Default: False.

        Returns:
            Tensor: Forward results.
        """

        assert imgs.shape[1] == 2
        x1, x2 = imgs[:, 0], imgs[:, 1]

        mean1 = x1.mean(2, keepdim=True).mean(3, keepdim=True)
        mean2 = x2.mean(2, keepdim=True).mean(3, keepdim=True)
        x1 -= mean1
        x2 -= mean2

        if padding_flag:
            padding_function, depadding_function = get_padding_functions(
                x1, self.padding)
            x1 = padding_function(x1)
            x2 = padding_function(x2)

        x1 = pixel_unshuffle(x1, self.scale)
        x2 = pixel_unshuffle(x2, self.scale)

        x = torch.cat([x1, x2], dim=1)
        x = self.conv_first(x)
        res = self.body(x)
        res += x
        x = self.conv_last(res)
        x = F.pixel_shuffle(x, self.scale)

        if padding_flag:
            x = depadding_function(x)

        x += (mean1 + mean2) / 2
        return x

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError('"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

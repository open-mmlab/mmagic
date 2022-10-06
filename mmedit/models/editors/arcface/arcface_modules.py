# Copyright (c) OpenMMLab. All rights reserved.
from collections import namedtuple

import torch
from torch.nn import (AdaptiveAvgPool2d, BatchNorm2d, Conv2d, MaxPool2d,
                      Module, PReLU, ReLU, Sequential, Sigmoid)

# yapf: disable
"""
ArcFace implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) # isort:skip  # noqa
"""
# yapf: enable


class Flatten(Module):
    """Flatten Module."""

    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    """l2 normalization.

    Args:
        input (torch.Tensor): The input tensor.
        axis (int, optional): Specifies which axis of input to calculate the
            norm across. Defaults to 1.

    Returns:
        Tensor: Tensor after L2 normalization per-instance.
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2):
    """Get a single block config.

    Args:
        in_channel (int): Input channels.
        depth (int): Output channels.
        num_units (int): Number of unit modules.
        stride (int, optional): Conv2d stride. Defaults to 2.

    Returns:
        list: A list of unit modules' config.
    """
    return [Bottleneck(in_channel, depth, stride)
            ] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    """Get block configs of backbone.

    Args:
        num_layers (int): Number of ConvBlock layers in backbone.

    Raises:
        ValueError: `num_layers` must be one of [50, 100, 152].

    Returns:
        list: A list of block configs.
    """
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError(
            'Invalid number of layers: {}. Must be one of [50, 100, 152]'.
            format(num_layers))
    return blocks


class SEModule(Module):
    """Squeeze-and-Excitation Modules.

    Args:
        channels (int): Input channels.
        reduction (int): Intermediate channels reduction ratio.
    """

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels,
            channels // reduction,
            kernel_size=1,
            padding=0,
            bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction,
            channels,
            kernel_size=1,
            padding=0,
            bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        """Forward Function."""
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(Module):
    """Intermediate Resblock of bottleneck.

    Args:
        in_channel (int): Input channels.
        depth (int): Output channels.
        stride (int): Conv2d stride.
    """

    def __init__(self, in_channel, depth, stride):
        """Intermediate Resblock of bottleneck.

        Args:
            in_channel (int): Input channels.
            depth (int): Output channels.
            stride (int): Conv2d stride.
        """
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth), Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        """Forward function."""
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    """Intermediate Resblock of bottleneck with SEModule.

    Args:
        in_channel (int): Input channels.
        depth (int): Output channels.
        stride (int): Conv2d stride.
    """

    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth), Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth), SEModule(depth, 16))

    def forward(self, x):
        """Forward function."""
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

import torch.nn as nn
from mmedit.models.common import ConvModule, build_activation_layer


class BasicBlock(nn.Module):
    """Basic residual block.

    Args:
        in_channels (int): Input channels of the block.
        out_channels (int): Output channels of the block.
        kernel_size (int): Kernel size of the convolution layers.
        stride (int): Stride of the first conv of the block.
        interpolation (nn.Module, optional): Interpolation module for skip
            connection.
        conv_cfg (dict): dictionary to construct convolution layer. If it is
            None, 2d convolution will be applied. Default: None.
        norm_cfg (dict): Config dict for normalization layer. "BN" by default.
        act_cfg (dict): Config dict for activation layer, "ReLU" by default.
        decode (bool): If this is a decode block.
            In decode mode, if stride is 2, the sample layer should be a
            upsample layer and conv1 will be ConvTranspose2d with kernel_size 4
            and padding 1. Default: False.
    """
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 interpolation=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(BasicBlock, self).__init__()
        assert stride == 1 or stride == 2, (
            f'stride other than 1 and 2 is not implemented, got {stride}')

        assert stride != 2 or interpolation is not None, (
            f'if stride is 2, interpolation should be specified')

        self.conv1 = self.build_conv1(in_channels, out_channels, kernel_size,
                                      stride, conv_cfg, norm_cfg, act_cfg)
        self.conv2 = self.build_conv2(in_channels, out_channels, kernel_size,
                                      conv_cfg, norm_cfg)

        self.interpolation = interpolation
        self.activation = build_activation_layer(act_cfg)
        self.stride = stride

    def build_conv1(self, in_channels, out_channels, kernel_size, stride,
                    conv_cfg, norm_cfg, act_cfg):
        return ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def build_conv2(self, in_channels, out_channels, kernel_size, conv_cfg,
                    norm_cfg):
        return ConvModule(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.interpolation is not None:
            identity = self.interpolation(x)

        out += identity
        out = self.activation(out)

        return out

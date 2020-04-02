import torch.nn as nn
from mmcv.cnn.weight_init import constant_init
from mmcv.runner import load_checkpoint
from mmedit.models.common import ConvModule, build_activation_layer
from mmedit.models.registry import COMPONENTS
from mmedit.utils.logger import get_root_logger


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


@COMPONENTS.register_module
class ResNetEnc(nn.Module):
    """ResNet encoder for image matting.

    This class is adopted from https://github.com/Yaoyi-Li/GCA-Matting.
    Implement and pre-train on ImageNet with the tricks from
    https://arxiv.org/abs/1812.01187
    without the mix-up part.

    Args:
        block (str): Type of residual block. Currently only `BasicBlock` is
            implemented.
        layers (list[int]): Number of layers in each block.
        in_channels (int): Number of input channels.
        conv_cfg (dict): dictionary to construct convolution layer. If it is
            ``None``, 2d convolution with be applied. Default: None.
        norm_cfg (dict): Config for norm layers. required keys are `type`.
            Default: dict(type='BN').
        late_downsample (bool): Whether to adopt late downsample strategy,
            Default: False.
    """

    def __init__(self,
                 block,
                 layers,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 late_downsample=False):
        super(ResNetEnc, self).__init__()
        if block == 'BasicBlock':
            block = BasicBlock
        else:
            raise NotImplementedError(f'{block} is not implemented.')

        self.inplanes = 64
        self.midplanes = 64 if late_downsample else 32

        start_stride = [1, 2, 1, 2] if late_downsample else [2, 1, 2, 1]
        self.conv1 = ConvModule(
            in_channels,
            32,
            3,
            stride=start_stride[0],
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            32,
            self.midplanes,
            3,
            stride=start_stride[1],
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            self.midplanes,
            self.inplanes,
            3,
            stride=start_stride[2],
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.layer1 = self._make_layer(block, 64, layers[0], start_stride[3],
                                       conv_cfg, norm_cfg, act_cfg)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, conv_cfg,
                                       norm_cfg, act_cfg)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, conv_cfg,
                                       norm_cfg, act_cfg)
        self.layer4 = self._make_layer(block, 512, layers[3], 2, conv_cfg,
                                       norm_cfg, act_cfg)

        self.out_channels = 512

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            # if pretrained weight is trained on 3-channel images,
            # initialize other channels with zeros
            self.conv1.weight.data[:, 3:, :, :] = 0

            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m.weight, 1)
                    constant_init(m.bias, 0)

            # Zero-initialize the last BN in each residual branch, so that the
            # residual branch starts with zeros, and each residual block
            # behaves like an identity. This improves the model by 0.2~0.3%
            # according to https://arxiv.org/abs/1706.02677
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    constant_init(m.conv2.bn.weight, 0)
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

    def _make_layer(self, block, planes, num_blocks, stride, conv_cfg,
                    norm_cfg, act_cfg):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(2, stride),
                ConvModule(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None))

        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                interpolation=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

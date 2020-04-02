import torch.nn as nn
from mmcv.cnn.weight_init import constant_init
from mmedit.models.common import ConvModule
from mmedit.models.registry import COMPONENTS

from ..encoders.resnet_enc import BasicBlock


class BasicBlockDec(BasicBlock):
    """Basic residual block for decoder.

    For decoder, we use ConvTranspose2d with kernel_size 4 and padding 1 for
    conv1. And the output channel of conv1 is modified from `out_channels` to
    `in_channels`.
    """

    def build_conv1(self, in_channels, out_channels, kernel_size, stride,
                    conv_cfg, norm_cfg, act_cfg):
        if stride == 2:
            conv_cfg = dict(type='Deconv')
            kernel_size = 4
            padding = 1
        else:
            padding = kernel_size // 2

        return ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def build_conv2(self, in_channels, out_channels, kernel_size, conv_cfg,
                    norm_cfg):
        return ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)


@COMPONENTS.register_module
class ResNetDec(nn.Module):
    """ResNet decoder for image matting.

    This class is adopted from https://github.com/Yaoyi-Li/GCA-Matting.

    Args:
        block (str): Type of residual block. Currently only `BasicBlockDec` is
            implemented.
        layers (list[int]): Number of layers in each block.
        in_channels (int): Channel num of input features.
        conv_cfg (dict): dictionary to construct convolution layer. If it is
            None, 2d convolution will be applied. Default: None.
        norm_cfg (dict): Config dict for normalization layer. "BN" by default.
        act_cfg (dict): Config dict for activation layer, "ReLU" by default.
        late_downsample (bool): Whether to adopt late downsample strategy,
            Default: False.
    """

    def __init__(self,
                 block,
                 layers,
                 in_channels,
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 late_downsample=False):
        super(ResNetDec, self).__init__()
        if block == 'BasicBlockDec':
            block = BasicBlockDec
        else:
            raise NotImplementedError(f'{block} is not implemented.')

        self.kernel_size = kernel_size
        self.inplanes = in_channels
        self.midplanes = 64 if late_downsample else 32

        self.layer1 = self._make_layer(block, 256, layers[0], conv_cfg,
                                       norm_cfg, act_cfg)
        self.layer2 = self._make_layer(block, 128, layers[1], conv_cfg,
                                       norm_cfg, act_cfg)
        self.layer3 = self._make_layer(block, 64, layers[2], conv_cfg,
                                       norm_cfg, act_cfg)
        self.layer4 = self._make_layer(block, self.midplanes, layers[3],
                                       conv_cfg, norm_cfg, act_cfg)

        self.conv1 = ConvModule(
            self.midplanes,
            32,
            4,
            stride=2,
            padding=1,
            conv_cfg=dict(type='Deconv'),
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2 = nn.Conv2d(
            32,
            1,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m.weight, 1)
                constant_init(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the
        # residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlockDec):
                constant_init(m.conv2.bn.weight, 0)

    def _make_layer(self, block, planes, num_blocks, conv_cfg, norm_cfg,
                    act_cfg):
        upsample = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
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
                kernel_size=self.kernel_size,
                stride=2,
                interpolation=upsample,
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
                    kernel_size=self.kernel_size,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return x

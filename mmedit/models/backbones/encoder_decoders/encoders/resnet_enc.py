# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer, constant_init
from mmcv.runner import load_checkpoint

from mmedit.models.common import GCAModule
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
        with_spectral_norm (bool): Whether use spectral norm after conv.
            Default: False.
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
                 act_cfg=dict(type='ReLU'),
                 with_spectral_norm=False):
        super().__init__()
        assert stride in (1, 2), (
            f'stride other than 1 and 2 is not implemented, got {stride}')

        assert stride != 2 or interpolation is not None, (
            'if stride is 2, interpolation should be specified')

        self.conv1 = self.build_conv1(in_channels, out_channels, kernel_size,
                                      stride, conv_cfg, norm_cfg, act_cfg,
                                      with_spectral_norm)
        self.conv2 = self.build_conv2(in_channels, out_channels, kernel_size,
                                      conv_cfg, norm_cfg, with_spectral_norm)

        self.interpolation = interpolation
        self.activation = build_activation_layer(act_cfg)
        self.stride = stride

    def build_conv1(self, in_channels, out_channels, kernel_size, stride,
                    conv_cfg, norm_cfg, act_cfg, with_spectral_norm):
        return ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            with_spectral_norm=with_spectral_norm)

    def build_conv2(self, in_channels, out_channels, kernel_size, conv_cfg,
                    norm_cfg, with_spectral_norm):
        return ConvModule(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.interpolation is not None:
            identity = self.interpolation(x)

        out += identity
        out = self.activation(out)

        return out


@COMPONENTS.register_module()
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
            None, 2d convolution will be applied. Default: None.
        norm_cfg (dict): Config dict for normalization layer. "BN" by default.
        act_cfg (dict): Config dict for activation layer, "ReLU" by default.
        with_spectral_norm (bool): Whether use spectral norm after conv.
            Default: False.
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
                 with_spectral_norm=False,
                 late_downsample=False):
        super().__init__()
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
            act_cfg=act_cfg,
            with_spectral_norm=with_spectral_norm)
        self.conv2 = ConvModule(
            32,
            self.midplanes,
            3,
            stride=start_stride[1],
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            with_spectral_norm=with_spectral_norm)
        self.conv3 = ConvModule(
            self.midplanes,
            self.inplanes,
            3,
            stride=start_stride[2],
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            with_spectral_norm=with_spectral_norm)

        self.layer1 = self._make_layer(block, 64, layers[0], start_stride[3],
                                       conv_cfg, norm_cfg, act_cfg,
                                       with_spectral_norm)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, conv_cfg,
                                       norm_cfg, act_cfg, with_spectral_norm)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, conv_cfg,
                                       norm_cfg, act_cfg, with_spectral_norm)
        self.layer4 = self._make_layer(block, 512, layers[3], 2, conv_cfg,
                                       norm_cfg, act_cfg, with_spectral_norm)

        self.out_channels = 512

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            # if pretrained weight is trained on 3-channel images,
            # initialize other channels with zeros
            self.conv1.conv.weight.data[:, 3:, :, :] = 0

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
                    norm_cfg, act_cfg, with_spectral_norm):
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
                    act_cfg=None,
                    with_spectral_norm=with_spectral_norm))

        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                interpolation=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_spectral_norm=with_spectral_norm)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_spectral_norm=with_spectral_norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            Tensor: Output tensor.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


@COMPONENTS.register_module()
class ResShortcutEnc(ResNetEnc):
    """ResNet backbone for image matting with shortcut connection.

    ::

        image ---------------- shortcut[0] --- feat1
          |
        conv1-conv2 ---------- shortcut[1] --- feat2
               |
              conv3-layer1 --- shortcut[2] --- feat3
                      |
                     layer2 -- shortcut[4] --- feat4
                       |
                      layer3 - shortcut[5] --- feat5
                        |
                       layer4 ---------------- out

    Baseline model of Natural Image Matting via Guided Contextual Attention
    https://arxiv.org/pdf/2001.04069.pdf.

    Args:
        block (str): Type of residual block. Currently only `BasicBlock` is
            implemented.
        layers (list[int]): Number of layers in each block.
        in_channels (int): Number of input channels.
        conv_cfg (dict): Dictionary to construct convolution layer. If it is
            None, 2d convolution will be applied. Default: None.
        norm_cfg (dict): Config dict for normalization layer. "BN" by default.
        act_cfg (dict): Config dict for activation layer, "ReLU" by default.
        with_spectral_norm (bool): Whether use spectral norm after conv.
            Default: False.
        late_downsample (bool): Whether to adopt late downsample strategy.
            Default: False.
        order (tuple[str]): Order of `conv`, `norm` and `act` layer in shortcut
            convolution module. Default: ('conv', 'act', 'norm').
    """

    def __init__(self,
                 block,
                 layers,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_spectral_norm=False,
                 late_downsample=False,
                 order=('conv', 'act', 'norm')):
        super().__init__(block, layers, in_channels, conv_cfg, norm_cfg,
                         act_cfg, with_spectral_norm, late_downsample)

        # TODO: rename self.midplanes to self.mid_channels in ResNetEnc
        self.shortcut_in_channels = [in_channels, self.midplanes, 64, 128, 256]
        self.shortcut_out_channels = [32, self.midplanes, 64, 128, 256]

        self.shortcut = nn.ModuleList()
        for in_channel, out_channel in zip(self.shortcut_in_channels,
                                           self.shortcut_out_channels):
            self.shortcut.append(
                self._make_shortcut(in_channel, out_channel, conv_cfg,
                                    norm_cfg, act_cfg, order,
                                    with_spectral_norm))

    def _make_shortcut(self, in_channels, out_channels, conv_cfg, norm_cfg,
                       act_cfg, order, with_spectral_norm):
        return nn.Sequential(
            ConvModule(
                in_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_spectral_norm=with_spectral_norm,
                order=order),
            ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_spectral_norm=with_spectral_norm,
                order=order))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            dict: Contains the output tensor and shortcut feature.
        """
        out = self.conv1(x)
        x1 = self.conv2(out)
        out = self.conv3(x1)

        x2 = self.layer1(out)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        out = self.layer4(x4)

        feat1 = self.shortcut[0](x)
        feat2 = self.shortcut[1](x1)
        feat3 = self.shortcut[2](x2)
        feat4 = self.shortcut[3](x3)
        feat5 = self.shortcut[4](x4)

        return {
            'out': out,
            'feat1': feat1,
            'feat2': feat2,
            'feat3': feat3,
            'feat4': feat4,
            'feat5': feat5,
        }


@COMPONENTS.register_module()
class ResGCAEncoder(ResShortcutEnc):
    """ResNet backbone with shortcut connection and gca module.

    ::

        image ---------------- shortcut[0] -------------- feat1
         |
        conv1-conv2 ---------- shortcut[1] -------------- feat2
               |
             conv3-layer1 ---- shortcut[2] -------------- feat3
                     |
                     | image - guidance_conv ------------ img_feat
                     |             |
                    layer2 --- gca_module - shortcut[4] - feat4
                                    |
                                  layer3 -- shortcut[5] - feat5
                                     |
                                   layer4 --------------- out

    * gca module also requires unknown tensor generated by trimap which is \
    ignored in the above graph.

    Implementation of Natural Image Matting via Guided Contextual Attention
    https://arxiv.org/pdf/2001.04069.pdf.

    Args:
        block (str): Type of residual block. Currently only `BasicBlock` is
            implemented.
        layers (list[int]): Number of layers in each block.
        in_channels (int): Number of input channels.
        conv_cfg (dict): Dictionary to construct convolution layer. If it is
            None, 2d convolution will be applied. Default: None.
        norm_cfg (dict): Config dict for normalization layer. "BN" by default.
        act_cfg (dict): Config dict for activation layer, "ReLU" by default.
        late_downsample (bool): Whether to adopt late downsample strategy.
            Default: False.
        order (tuple[str]): Order of `conv`, `norm` and `act` layer in shortcut
            convolution module. Default: ('conv', 'act', 'norm').
    """

    def __init__(self,
                 block,
                 layers,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_spectral_norm=False,
                 late_downsample=False,
                 order=('conv', 'act', 'norm')):
        super().__init__(block, layers, in_channels, conv_cfg, norm_cfg,
                         act_cfg, with_spectral_norm, late_downsample, order)

        assert in_channels in (4, 6), (
            f'in_channels must be 4 or 6, but got {in_channels}')

        self.trimap_channels = in_channels - 3

        guidance_in_channels = [3, 16, 32]
        guidance_out_channels = [16, 32, 128]

        guidance_head = []
        for in_channel, out_channel in zip(guidance_in_channels,
                                           guidance_out_channels):
            guidance_head += [
                ConvModule(
                    in_channel,
                    out_channel,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_spectral_norm=with_spectral_norm,
                    padding_mode='reflect',
                    order=order)
            ]
        self.guidance_head = nn.Sequential(*guidance_head)

        self.gca = GCAModule(128, 128)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            super().init_weights()
        else:
            raise TypeError('"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            dict: Contains the output tensor, shortcut feature and \
                intermediate feature.
        """
        out = self.conv1(x)
        x1 = self.conv2(out)
        out = self.conv3(x1)

        img_feat = self.guidance_head(x[:, :3, ...])
        if self.trimap_channels == 3:
            unknown = x[:, 4:5, ...]
        else:
            unknown = x[:, 3:, ...].eq(1).float()
        # same as img_feat, downsample to 1/8
        unknown = F.interpolate(unknown, scale_factor=1 / 8, mode='nearest')

        x2 = self.layer1(out)
        x3 = self.layer2(x2)
        x3 = self.gca(img_feat, x3, unknown)
        x4 = self.layer3(x3)
        out = self.layer4(x4)

        # shortcut block
        feat1 = self.shortcut[0](x)
        feat2 = self.shortcut[1](x1)
        feat3 = self.shortcut[2](x2)
        feat4 = self.shortcut[3](x3)
        feat5 = self.shortcut[4](x4)

        return {
            'out': out,
            'feat1': feat1,
            'feat2': feat2,
            'feat3': feat3,
            'feat4': feat4,
            'feat5': feat5,
            'img_feat': img_feat,
            'unknown': unknown
        }

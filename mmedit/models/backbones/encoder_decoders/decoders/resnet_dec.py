import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init

from mmedit.models.common import GCAModule
from mmedit.models.registry import COMPONENTS
from ..encoders.resnet_enc import BasicBlock


class BasicBlockDec(BasicBlock):
    """Basic residual block for decoder.

    For decoder, we use ConvTranspose2d with kernel_size 4 and padding 1 for
    conv1. And the output channel of conv1 is modified from `out_channels` to
    `in_channels`.
    """

    def build_conv1(self, in_channels, out_channels, kernel_size, stride,
                    conv_cfg, norm_cfg, act_cfg, with_spectral_norm):
        """Build conv1 of the block.

        Args:
            in_channels (int): The input channels of the ConvModule.
            out_channels (int): The output channels of the ConvModule.
            kernel_size (int): The kernel size of the ConvModule.
            stride (int): The stride of the ConvModule. If stride is set to 2,
                then ``conv_cfg`` will be overwritten as
                ``dict(type='Deconv')`` and ``kernel_size`` will be overwritten
                as 4.
            conv_cfg (dict): The conv config of the ConvModule.
            norm_cfg (dict): The norm config of the ConvModule.
            act_cfg (dict): The activation config of the ConvModule.
            with_spectral_norm (bool): Whether use spectral norm.

        Returns:
            nn.Module: The built ConvModule.
        """
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
            act_cfg=act_cfg,
            with_spectral_norm=with_spectral_norm)

    def build_conv2(self, in_channels, out_channels, kernel_size, conv_cfg,
                    norm_cfg, with_spectral_norm):
        """Build conv2 of the block.

        Args:
            in_channels (int): The input channels of the ConvModule.
            out_channels (int): The output channels of the ConvModule.
            kernel_size (int): The kernel size of the ConvModule.
            conv_cfg (dict): The conv config of the ConvModule.
            norm_cfg (dict): The norm config of the ConvModule.
            with_spectral_norm (bool): Whether use spectral norm.

        Returns:
            nn.Module: The built ConvModule.
        """
        return ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm)


@COMPONENTS.register_module()
class ResNetDec(nn.Module):
    """ResNet decoder for image matting.

    This class is adopted from https://github.com/Yaoyi-Li/GCA-Matting.

    Args:
        block (str): Type of residual block. Currently only `BasicBlockDec` is
            implemented.
        layers (list[int]): Number of layers in each block.
        in_channels (int): Channel num of input features.
        kernel_size (int): Kernel size of the conv layers in the decoder.
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
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(
                     type='LeakyReLU', negative_slope=0.2, inplace=True),
                 with_spectral_norm=False,
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
                                       norm_cfg, act_cfg, with_spectral_norm)
        self.layer2 = self._make_layer(block, 128, layers[1], conv_cfg,
                                       norm_cfg, act_cfg, with_spectral_norm)
        self.layer3 = self._make_layer(block, 64, layers[2], conv_cfg,
                                       norm_cfg, act_cfg, with_spectral_norm)
        self.layer4 = self._make_layer(block, self.midplanes, layers[3],
                                       conv_cfg, norm_cfg, act_cfg,
                                       with_spectral_norm)

        self.conv1 = ConvModule(
            self.midplanes,
            32,
            4,
            stride=2,
            padding=1,
            conv_cfg=dict(type='Deconv'),
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            with_spectral_norm=with_spectral_norm)

        self.conv2 = ConvModule(
            32,
            1,
            self.kernel_size,
            padding=self.kernel_size // 2,
            act_cfg=None)

    def init_weights(self):
        """Init weights for the module.
        """
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
                    act_cfg, with_spectral_norm):
        upsample = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
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
                kernel_size=self.kernel_size,
                stride=2,
                interpolation=upsample,
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
                    kernel_size=self.kernel_size,
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return x


@COMPONENTS.register_module()
class ResShortcutDec(ResNetDec):
    """ResNet decoder for image matting with shortcut connection.

    ::

        feat1 --------------------------- conv2 --- out
                                       |
        feat2 ---------------------- conv1
                                  |
        feat3 ----------------- layer4
                             |
        feat4 ------------ layer3
                        |
        feat5 ------- layer2
                   |
        out ---  layer1

    Args:
        block (str): Type of residual block. Currently only `BasicBlockDec` is
            implemented.
        layers (list[int]): Number of layers in each block.
        in_channels (int): Channel number of input features.
        kernel_size (int): Kernel size of the conv layers in the decoder.
        conv_cfg (dict): Dictionary to construct convolution layer. If it is
            None, 2d convolution will be applied. Default: None.
        norm_cfg (dict): Config dict for normalization layer. "BN" by default.
        act_cfg (dict): Config dict for activation layer, "ReLU" by default.
        late_downsample (bool): Whether to adopt late downsample strategy,
            Default: False.
    """

    def forward(self, inputs):
        """Forward function of resnet shortcut decoder.

        Args:
            inputs (dict): Output dictionary of the ResNetEnc containing:

              - out (Tensor): Output of the ResNetEnc.
              - feat1 (Tensor): Shortcut connection from input image.
              - feat2 (Tensor): Shortcut connection from conv2 of ResNetEnc.
              - feat3 (Tensor): Shortcut connection from layer1 of ResNetEnc.
              - feat4 (Tensor): Shortcut connection from layer2 of ResNetEnc.
              - feat5 (Tensor): Shortcut connection from layer3 of ResNetEnc.

        Returns:
            Tensor: Output tensor.
        """
        feat1 = inputs['feat1']
        feat2 = inputs['feat2']
        feat3 = inputs['feat3']
        feat4 = inputs['feat4']
        feat5 = inputs['feat5']
        x = inputs['out']

        x = self.layer1(x) + feat5
        x = self.layer2(x) + feat4
        x = self.layer3(x) + feat3
        x = self.layer4(x) + feat2
        x = self.conv1(x) + feat1
        x = self.conv2(x)

        return x


@COMPONENTS.register_module()
class ResGCADecoder(ResShortcutDec):
    """ResNet decoder with shortcut connection and gca module.

    ::

        feat1 ---------------------------------------- conv2 --- out
                                                    |
        feat2 ----------------------------------- conv1
                                               |
        feat3 ------------------------------ layer4
                                          |
        feat4, img_feat -- gca_module - layer3
                        |
        feat5 ------- layer2
                   |
        out ---  layer1

    * gca module also requires unknown tensor generated by trimap which is \
    ignored in the above graph.

    Args:
        block (str): Type of residual block. Currently only `BasicBlockDec` is
            implemented.
        layers (list[int]): Number of layers in each block.
        in_channels (int): Channel number of input features.
        kernel_size (int): Kernel size of the conv layers in the decoder.
        conv_cfg (dict): Dictionary to construct convolution layer. If it is
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
                 act_cfg=dict(
                     type='LeakyReLU', negative_slope=0.2, inplace=True),
                 with_spectral_norm=False,
                 late_downsample=False):
        super(ResGCADecoder,
              self).__init__(block, layers, in_channels, kernel_size, conv_cfg,
                             norm_cfg, act_cfg, with_spectral_norm,
                             late_downsample)
        self.gca = GCAModule(128, 128)

    def forward(self, inputs):
        """Forward function of resnet shortcut decoder.

        Args:
            inputs (dict): Output dictionary of the ResGCAEncoder containing:

              - out (Tensor): Output of the ResGCAEncoder.
              - feat1 (Tensor): Shortcut connection from input image.
              - feat2 (Tensor): Shortcut connection from conv2 of \
                    ResGCAEncoder.
              - feat3 (Tensor): Shortcut connection from layer1 of \
                    ResGCAEncoder.
              - feat4 (Tensor): Shortcut connection from layer2 of \
                    ResGCAEncoder.
              - feat5 (Tensor): Shortcut connection from layer3 of \
                    ResGCAEncoder.
              - img_feat (Tensor): Image feature extracted by guidance head.
              - unknown (Tensor): Unknown tensor generated by trimap.

        Returns:
            Tensor: Output tensor.
        """
        img_feat = inputs['img_feat']
        unknown = inputs['unknown']
        feat1 = inputs['feat1']
        feat2 = inputs['feat2']
        feat3 = inputs['feat3']
        feat4 = inputs['feat4']
        feat5 = inputs['feat5']
        x = inputs['out']

        x = self.layer1(x) + feat5
        x = self.layer2(x) + feat4
        x = self.gca(img_feat, x, unknown)
        x = self.layer3(x) + feat3
        x = self.layer4(x) + feat2
        x = self.conv1(x) + feat1
        x = self.conv2(x)

        return x

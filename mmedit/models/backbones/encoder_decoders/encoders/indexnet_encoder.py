# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, constant_init, xavier_init
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import SyncBatchNorm

from mmedit.models.common import ASPP, DepthwiseSeparableConvModule
from mmedit.models.registry import COMPONENTS
from mmedit.utils import get_root_logger


def build_index_block(in_channels,
                      out_channels,
                      kernel_size,
                      stride=2,
                      padding=0,
                      groups=1,
                      norm_cfg=dict(type='BN'),
                      use_nonlinear=False,
                      expansion=1):
    """Build an conv block for IndexBlock.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        kernel_size (int): The kernel size of the block.
        stride (int, optional): The stride of the block. Defaults to 2.
        padding (int, optional): The padding of the block. Defaults to 0.
        groups (int, optional): The groups of the block. Defaults to 1.
        norm_cfg (dict, optional): The norm config of the block.
            Defaults to dict(type='BN').
        use_nonlinear (bool, optional): Whether use nonlinearty in the block.
            If true, a ConvModule with kernel size 1 will be appended and an
            ``ReLU6`` nonlinearty will be added to the origin ConvModule.
            Defaults to False.
        expansion (int, optional): Expandsion ratio of the middle channels.
            Effective when ``use_nonlinear`` is true. Defaults to 1.

    Returns:
        nn.Module: The built conv block.
    """
    if use_nonlinear:
        return nn.Sequential(
            ConvModule(
                in_channels,
                in_channels * expansion,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU6')),
            ConvModule(
                in_channels * expansion,
                out_channels,
                1,
                stride=1,
                padding=0,
                groups=groups,
                bias=False,
                norm_cfg=None,
                act_cfg=None))

    return ConvModule(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        norm_cfg=None,
        act_cfg=None)


class HolisticIndexBlock(nn.Module):
    """Holistic Index Block.

    From https://arxiv.org/abs/1908.00672.

    Args:
        in_channels (int): Input channels of the holistic index block.
        kernel_size (int): Kernel size of the conv layers. Default: 2.
        padding (int): Padding number of the conv layers. Default: 0.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        use_nonlinear (bool): Whether add a non-linear conv layer in the index
            block. Default: False.
    """

    def __init__(self,
                 in_channels,
                 norm_cfg=dict(type='BN'),
                 use_context=False,
                 use_nonlinear=False):
        super().__init__()

        if use_context:
            kernel_size, padding = 4, 1
        else:
            kernel_size, padding = 2, 0

        self.index_block = build_index_block(
            in_channels,
            4,
            kernel_size,
            stride=2,
            padding=padding,
            groups=1,
            norm_cfg=norm_cfg,
            use_nonlinear=use_nonlinear,
            expansion=2)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape (N, C, H, W).

        Returns:
            tuple(Tensor): Encoder index feature and decoder index feature.
        """
        x = self.index_block(x)

        # normalization
        y = self.sigmoid(x)
        z = self.softmax(y)
        # pixel shuffling
        enc_idx_feat = self.pixel_shuffle(z)
        dec_idx_feat = self.pixel_shuffle(y)

        return enc_idx_feat, dec_idx_feat


class DepthwiseIndexBlock(nn.Module):
    """Depthwise index block.

    From https://arxiv.org/abs/1908.00672.

    Args:
        in_channels (int): Input channels of the holistic index block.
        kernel_size (int): Kernel size of the conv layers. Default: 2.
        padding (int): Padding number of the conv layers. Default: 0.
        mode (str): Mode of index block. Should be 'o2o' or 'm2o'. In 'o2o'
            mode, the group of the conv layers is 1; In 'm2o' mode, the group
            of the conv layer is `in_channels`.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        use_nonlinear (bool): Whether add a non-linear conv layer in the index
            blocks. Default: False.
    """

    def __init__(self,
                 in_channels,
                 norm_cfg=dict(type='BN'),
                 use_context=False,
                 use_nonlinear=False,
                 mode='o2o'):
        super().__init__()

        groups = in_channels if mode == 'o2o' else 1

        if use_context:
            kernel_size, padding = 4, 1
        else:
            kernel_size, padding = 2, 0

        self.index_blocks = nn.ModuleList()
        for _ in range(4):
            self.index_blocks.append(
                build_index_block(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride=2,
                    padding=padding,
                    groups=groups,
                    norm_cfg=norm_cfg,
                    use_nonlinear=use_nonlinear))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape (N, C, H, W).

        Returns:
            tuple(Tensor): Encoder index feature and decoder index feature.
        """
        n, c, h, w = x.shape

        feature_list = [
            _index_block(x).unsqueeze(2) for _index_block in self.index_blocks
        ]
        x = torch.cat(feature_list, dim=2)

        # normalization
        y = self.sigmoid(x)
        z = self.softmax(y)
        # pixel shuffling
        y = y.view(n, c * 4, h // 2, w // 2)
        z = z.view(n, c * 4, h // 2, w // 2)
        enc_idx_feat = self.pixel_shuffle(z)
        dec_idx_feat = self.pixel_shuffle(y)

        return enc_idx_feat, dec_idx_feat


class InvertedResidual(nn.Module):
    """Inverted residual layer for indexnet encoder.

    It basically is a depthwise separable conv module. If `expand_ratio` is not
    one, then a conv module of kernel_size 1 will be inserted to change the
    input channels to `in_channels * expand_ratio`.

    Args:
        in_channels (int): Input channels of the layer.
        out_channels (int): Output channels of the layer.
        stride (int): Stride of the depthwise separable conv module.
        dilation (int): Dilation of the depthwise separable conv module.
        expand_ratio (float): Expand ratio of the input channels of the
            depthwise separable conv module.
        norm_cfg (dict | None): Config dict for normalization layer.
        use_res_connect (bool, optional): Whether use shortcut connection.
            Defaults to False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilation,
                 expand_ratio,
                 norm_cfg,
                 use_res_connect=False):
        super().__init__()
        assert stride in [1, 2], 'stride must 1 or 2'

        self.use_res_connect = use_res_connect
        self.kernel_size = 3
        self.dilation = dilation

        if expand_ratio == 1:
            self.conv = DepthwiseSeparableConvModule(
                in_channels,
                out_channels,
                3,
                stride=stride,
                dilation=dilation,
                norm_cfg=norm_cfg,
                dw_act_cfg=dict(type='ReLU6'),
                pw_act_cfg=None)
        else:
            hidden_dim = round(in_channels * expand_ratio)
            self.conv = nn.Sequential(
                ConvModule(
                    in_channels,
                    hidden_dim,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                DepthwiseSeparableConvModule(
                    hidden_dim,
                    out_channels,
                    3,
                    stride=stride,
                    dilation=dilation,
                    norm_cfg=norm_cfg,
                    dw_act_cfg=dict(type='ReLU6'),
                    pw_act_cfg=None))

    def pad(self, inputs, kernel_size, dilation):
        effective_ksize = kernel_size + (kernel_size - 1) * (dilation - 1)
        left = (effective_ksize - 1) // 2
        right = effective_ksize // 2
        return F.pad(inputs, (left, right, left, right))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape (N, C, H, W).

        Returns:
            Tensor: Output feature map.
        """
        out = self.conv(self.pad(x, self.kernel_size, self.dilation))

        if self.use_res_connect:
            out = out + x

        return out


@COMPONENTS.register_module()
class IndexNetEncoder(nn.Module):
    """Encoder for IndexNet.

    Please refer to https://arxiv.org/abs/1908.00672.

    Args:
        in_channels (int, optional): Input channels of the encoder.
        out_stride (int, optional): Output stride of the encoder. For
            example, if `out_stride` is 32, the input feature map or image
            will be downsample to the 1/32 of original size.
            Defaults to 32.
        width_mult (int, optional): Width multiplication factor of channel
            dimension in MobileNetV2. Defaults to 1.
        index_mode (str, optional): Index mode of the index network. It
            must be one of {'holistic', 'o2o', 'm2o'}. If it is set to
            'holistic', then Holistic index network will be used as the
            index network. If it is set to 'o2o' (or 'm2o'), when O2O
            (or M2O) Depthwise index network will be used as the index
            network. Defaults to 'm2o'.
        aspp (bool, optional): Whether use ASPP module to augment output
            feature. Defaults to True.
        norm_cfg (None | dict, optional): Config dict for normalization
            layer. Defaults to dict(type='BN').
        freeze_bn (bool, optional): Whether freeze batch norm layer.
            Defaults to False.
        use_nonlinear (bool, optional): Whether use nonlinearty in index
            network. Refer to the paper for more information.
            Defaults to True.
        use_context (bool, optional): Whether use larger kernel size in
            index network. Refer to the paper for more information.
            Defaults to True.

    Raises:
        ValueError: out_stride must 16 or 32.
        NameError: Supported index_mode are {'holistic', 'o2o', 'm2o'}.
    """

    def __init__(self,
                 in_channels,
                 out_stride=32,
                 width_mult=1,
                 index_mode='m2o',
                 aspp=True,
                 norm_cfg=dict(type='BN'),
                 freeze_bn=False,
                 use_nonlinear=True,
                 use_context=True):
        super().__init__()
        if out_stride not in [16, 32]:
            raise ValueError(f'out_stride must 16 or 32, got {out_stride}')

        self.out_stride = out_stride
        self.width_mult = width_mult

        # we name the index network in the paper index_block
        if index_mode == 'holistic':
            index_block = HolisticIndexBlock
        elif index_mode in ('o2o', 'm2o'):
            index_block = partial(DepthwiseIndexBlock, mode=index_mode)
        else:
            raise NameError('Unknown index block mode {}'.format(index_mode))

        # default setting
        initial_channels = 32
        inverted_residual_setting = [
            # expand_ratio, input_chn, output_chn, num_blocks, stride, dilation
            [1, initial_channels, 16, 1, 1, 1],
            [6, 16, 24, 2, 2, 1],
            [6, 24, 32, 3, 2, 1],
            [6, 32, 64, 4, 2, 1],
            [6, 64, 96, 3, 1, 1],
            [6, 96, 160, 3, 2, 1],
            [6, 160, 320, 1, 1, 1],
        ]

        # update layer setting according to width_mult
        initial_channels = int(initial_channels * width_mult)
        for layer_setting in inverted_residual_setting:
            # update in_channels and out_channels
            layer_setting[1] = int(layer_setting[1] * self.width_mult)
            layer_setting[2] = int(layer_setting[2] * self.width_mult)

        if out_stride == 32:
            # It should be noted that layers 0 is not an InvertedResidual layer
            # but a ConvModule. Thus, the index of InvertedResidual layer in
            # downsampled_layers starts at 1.
            self.downsampled_layers = [0, 2, 3, 4, 6]
        else:  # out_stride is 16
            self.downsampled_layers = [0, 2, 3, 4]
            # if out_stride is 16, then increase the dilation of the last two
            # InvertedResidual layer to increase the receptive field
            inverted_residual_setting[5][5] = 2
            inverted_residual_setting[6][5] = 2

        # build the first layer
        self.layers = nn.ModuleList([
            ConvModule(
                in_channels,
                initial_channels,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU6'))
        ])
        # build bottleneck layers
        for layer_setting in inverted_residual_setting:
            self.layers.append(self._make_layer(layer_setting, norm_cfg))

        # freeze encoder batch norm layers
        if freeze_bn:
            self.freeze_bn()

        # build index blocks
        self.index_layers = nn.ModuleList()
        for layer in self.downsampled_layers:
            # inverted_residual_setting begins at layer1, the in_channels
            # of layer1 is the out_channels of layer0
            self.index_layers.append(
                index_block(inverted_residual_setting[layer][1], norm_cfg,
                            use_context, use_nonlinear))
        self.avg_pool = nn.AvgPool2d(2, stride=2)

        if aspp:
            dilation = (2, 4, 8) if out_stride == 32 else (6, 12, 18)
            self.dconv = ASPP(
                320 * self.width_mult,
                160,
                mid_channels=int(256 * self.width_mult),
                dilations=dilation,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU6'),
                separable_conv=True)
        else:
            self.dconv = ConvModule(
                320 * self.width_mult,
                160,
                1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU6'))

        self.out_channels = 160

    def _make_layer(self, layer_setting, norm_cfg):
        # expand_ratio, in_channels, out_channels, num_blocks, stride, dilation
        (expand_ratio, in_channels, out_channels, num_blocks, stride,
         dilation) = layer_setting

        # downsample is now implemented by index block. In those layers that
        # have downsampling originally, use stride of 1 in the first block and
        # decrease the dilation accordingly.
        dilation0 = max(dilation // 2, 1) if stride == 2 else dilation
        layers = [
            InvertedResidual(in_channels, out_channels, 1, dilation0,
                             expand_ratio, norm_cfg)
        ]

        in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(
                InvertedResidual(
                    in_channels,
                    out_channels,
                    1,
                    dilation,
                    expand_ratio,
                    norm_cfg,
                    use_res_connect=True))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        """Set BatchNorm modules in the model to evaluation mode."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, SyncBatchNorm)):
                m.eval()

    def init_weights(self, pretrained=None):
        """Init weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape (N, C, H, W).

        Returns:
            dict: Output tensor, shortcut feature and decoder index feature.
        """
        dec_idx_feat_list = list()
        shortcuts = list()
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.downsampled_layers:
                enc_idx_feat, dec_idx_feat = self.index_layers[
                    self.downsampled_layers.index(i)](
                        x)
                x = enc_idx_feat * x
                shortcuts.append(x)
                dec_idx_feat_list.append(dec_idx_feat)
                x = 4 * self.avg_pool(x)
            elif i != 7:
                shortcuts.append(x)
                dec_idx_feat_list.append(None)

        x = self.dconv(x)

        return {
            'out': x,
            'shortcuts': shortcuts,
            'dec_idx_feat_list': dec_idx_feat_list
        }

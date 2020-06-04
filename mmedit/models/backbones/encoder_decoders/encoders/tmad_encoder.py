from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmedit.models.registry import COMPONENTS


class PGDownsampleBlock(nn.Module):
    """PGGAN-style downsample block.

    In this block, we follow the downsample block used in the discriminator of
    PGGAN. The architecture of this block is [conv, conv, downsample].

    Details can be found in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Args:
        in_channels (int): Channels of input feature or image.
        out_channels (int): Channels of output feature.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        scale_factor (int | float | tuple[float]): When `interpolation` is in
            ['nearest', 'bilinear'], this args is the same as
            `torch.nn.functional.interpolate`. If `interpolation` is
            'avgpool2d', `scale_factor` will be used as `kernel_size` and
            `stride` in the pooling layers. In this case, only interger has
            been supported.
        size (tuple[int]):  When `interpolation` in ['nearest', 'bilinear'],
            this args is the same as `torch.nn.functional.interpolate`.
        interpolation (str | None): Method used for downsampling. Currently, we
            support ['nearest', 'bilinear', 'avgpool2d']. If given `None`, the
            interpolation will be removed.
        kwargs (keyword arguments): Keyword arguments for `ConvModule`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                 scale_factor=0.5,
                 size=None,
                 interpolation='nearest',
                 **kwargs):
        super(PGDownsampleBlock, self).__init__()
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            act_cfg=act_cfg,
            **kwargs)
        self.conv2 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            act_cfg=act_cfg,
            **kwargs)

        if interpolation in ['nearest', 'bilinear']:
            self.downsample = partial(
                F.interpolate,
                size=size,
                scale_factor=scale_factor,
                mode=interpolation)
        elif interpolation == 'avgpool2d':
            self.downsample = partial(
                F.avg_pool2d, kernel_size=scale_factor, stride=scale_factor)
        elif interpolation is None:
            self.downsample = nn.Identity()
        else:
            raise NotImplementedError('Currently, we do not support the '
                                      f'interpolation mode {interpolation}')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.downsample(x)
        return x


@COMPONENTS.register_module
class TMADEncoder(nn.Module):
    """TMAD Encoder.

    In this ecoder, we build up the model with `PGDownsampleBlock`. Details can
    be found in:
    Texture Memory Augmented Deep Image Inpainting.

    Args:
        in_channels (int): Channels of input feature or image.
        channel_factor (int, optional): The output channels of the input conv
            module and the width of the encoder is computed by multiplying a
            constant with channel_factor. Defaults to 16.
        num_blocks (int, optional): The number of downsampling blocks used in
            the encoder. Defaults to 3.
        conv_cfg (None | dict, optional): Config dict for convolution layer.
            Defaults to None.
        norm_cfg (None | dict, optional): Config dict for normalization layer.
            Defaults to None.
        act_cfg (None | dict, optional): Config dict for activation layer.
            Defaults to dict(type='LeakyReLU', negative_slope=0.2).
        scale_factor (int | float | tuple[float]): When `interpolation` in
            ['nearest', 'bilinear'], this args is the same as
            `torch.nn.functional.interpolate`. If `interpolation` is
            'avgpool2d', `scale_factor` will be used as `kernel_size` and
            `stride` in the pooling layers. In this case, only interger has
            been supported.
        size (tuple[int]):  When `interpolation` in ['nearest', 'bilinear'],
            this args is the same as `torch.nn.functional.interpolate`.
        interpolation (str, optional): Method used for downsampling. Currently,
            we support ['nearest', 'bilinear', 'avgpool2d'].
            Defaults to 'bilinear'.
        kwargs (keyword arguments): Keyword arguments for `ConvModule`.
    """

    def __init__(self,
                 in_channels,
                 channel_factor=16,
                 num_blocks=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                 scale_factor=0.5,
                 size=None,
                 interpolation='bilinear',
                 **kwargs):

        super(TMADEncoder, self).__init__()
        self.num_blocks = num_blocks

        self.input_conv = ConvModule(
            in_channels,
            channel_factor,
            kernel_size=3,
            padding=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            **kwargs)

        encoder_blocks_ = []

        in_channels = channel_factor
        for i in range(num_blocks):
            encoder_blocks_.append(
                PGDownsampleBlock(
                    in_channels,
                    in_channels * 2,
                    interpolation=interpolation,
                    scale_factor=scale_factor,
                    size=size,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **kwargs))
            in_channels *= 2

        self.encoder_blocks = nn.ModuleList(encoder_blocks_)

    def forward(self, x):
        output_dict = dict()
        x = self.input_conv(x)
        for i in range(self.num_blocks):
            x = self.encoder_blocks[i](x)
            output_dict[f'dsblock{i}'] = x
        output_dict['out'] = x
        return output_dict

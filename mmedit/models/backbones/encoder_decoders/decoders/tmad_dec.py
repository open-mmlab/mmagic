import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer
from mmedit.models.registry import COMPONENTS


class PGUpsampleBlock(nn.Module):
    """PGGAN-style upsample block.

    In this block, we follow the downsample block used in the generator of
    PGGAN. The architecture of this block is [upsample, conv, conv].

    Details can be found in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Args:
        in_channels (int): Channels of input feature or image.
        out_channels (int): Channels of output feature.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        scale_factor (float | tuple[float]): When `interpolation` in
            ['nearest', 'bilinear'], this args is the same as
            `torch.nn.functional.interpolate`.
        size (tuple[int]):  When `interpolation` in ['nearest', 'bilinear'],
            this args is the same as `torch.nn.functional.interpolate`.
        interpolation (str | None): Method used for downsampling. Currently, we
            support ['nearest', 'bilinear', 'carafe']. If given `None`, the
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
                 scale_factor=2.,
                 size=None,
                 interpolation='nearest',
                 **kwargs):
        super(PGUpsampleBlock, self).__init__()
        self.interpolation = interpolation

        self.conv1 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act_cfg=act_cfg,
            **kwargs)
        self.conv2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act_cfg=act_cfg,
            **kwargs)

        if interpolation in ['nearest', 'bilinear']:
            self.upsample = partial(
                F.interpolate,
                size=size,
                scale_factor=scale_factor,
                mode=interpolation)
        elif interpolation == 'carafe':
            from mmedit.ops.carafe_upsample import CARAFEPack
            self.upsample = CARAFEPack(in_channels, scale_factor=scale_factor)
        elif interpolation is None:
            self.upsample = nn.Identity()
        else:
            raise NotImplementedError(
                'Currently, we do not suppport the '
                f'interpolation mode {interpolation} for upsampling')

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return x


@COMPONENTS.register_module()
class TMADDecoder(nn.Module):
    """TMAD Decoder.

    In this decoder, we build up the model with `PGUpsampleBlock`. Details can
    be found in:
    Texture Memory Augmented Deep Image Inpainting.

    Args:
        in_channels (int): Channels of input feature or image.
        out_channels (int): Channels of output feature.
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
            `torch.nn.functional.interpolate`.
        size (tuple[int]):  When `interpolation` in ['nearest', 'bilinear'],
            this args is the same as `torch.nn.functional.interpolate`.
        interpolation (str, optional): Method used for downsampling. Currently,
            we support ['nearest', 'bilinear', 'avgpool2d'].
            Defaults to 'bilinear'.
        out_act_cfg (dict): Config dict for output activation layer. Here, we
            provide commonly used `clip` operation.
        kwargs (keyword arguments): Keyword arguments for `ConvModule`.
    """

    def __init__(self,
                 in_channels,
                 out_channels=3,
                 num_blocks=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                 scale_factor=2.,
                 size=None,
                 interpolation='bilinear',
                 out_act_cfg=dict(type='clip', min=-1., max=1.),
                 **kwargs):
        super(TMADDecoder, self).__init__()
        self.with_out_activation = out_act_cfg is not None

        decoder_blocks_ = []
        for i in range(num_blocks):
            decoder_blocks_.append(
                PGUpsampleBlock(
                    in_channels,
                    in_channels // 2,
                    scale_factor=scale_factor,
                    size=size,
                    interpolation=interpolation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **kwargs))
            in_channels = in_channels // 2

        self.decoder_blocks = nn.Sequential(*decoder_blocks_)

        self.output_conv1 = ConvModule(
            in_channels,
            in_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        self.output_conv2 = ConvModule(
            in_channels // 2,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
            **kwargs)

        if self.with_out_activation:
            act_type = out_act_cfg['type']
            if act_type == 'clip':
                act_cfg_ = copy.deepcopy(out_act_cfg)
                act_cfg_.pop('type')
                self.out_act = partial(torch.clamp, **act_cfg_)
            else:
                self.out_act = build_activation_layer(out_act_cfg)

    def forward(self, x):
        x = self.decoder_blocks(x)
        x = self.output_conv1(x)
        x = self.output_conv2(x)

        if self.with_out_activation:
            x = self.out_act(x)

        return x

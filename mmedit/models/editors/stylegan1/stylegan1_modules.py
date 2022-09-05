# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from functools import partial

import mmengine
import torch
import torch.nn as nn
from mmcv.ops.fused_bias_leakyrelu import fused_bias_leakyrelu
from mmcv.ops.upfirdn2d import upfirdn2d

from mmedit.registry import MODELS
from ..pggan import (EqualizedLRConvModule, EqualizedLRConvUpModule,
                     EqualizedLRLinearModule)


class EqualLinearActModule(nn.Module):
    """Equalized LR Linear Module with Activation Layer.

    This module is modified from ``EqualizedLRLinearModule`` defined in PGGAN.
    The major features updated in this module is adding support for activation
    layers used in StyleGAN2.

    Args:
        equalized_lr_cfg (dict | None, optional): Config for equalized lr.
            Defaults to dict(gain=1., lr_mul=1.).
        bias (bool, optional): Whether to use bias item. Defaults to True.
        bias_init (float, optional): The value for bias initialization.
            Defaults to ``0.``.
        act_cfg (dict | None, optional): Config for activation layer.
            Defaults to None.
    """

    def __init__(self,
                 *args,
                 equalized_lr_cfg=dict(gain=1., lr_mul=1.),
                 bias=True,
                 bias_init=0.,
                 act_cfg=None,
                 **kwargs):
        super().__init__()
        self.with_activation = act_cfg is not None
        # w/o bias in linear layer
        self.linear = EqualizedLRLinearModule(
            *args, bias=False, equalized_lr_cfg=equalized_lr_cfg, **kwargs)

        if equalized_lr_cfg is not None:
            self.lr_mul = equalized_lr_cfg.get('lr_mul', 1.)
        else:
            self.lr_mul = 1.

        # define bias outside linear layer
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.linear.out_features).fill_(bias_init))
        else:
            self.bias = None

        if self.with_activation:
            act_cfg = deepcopy(act_cfg)
            if act_cfg['type'] == 'fused_bias':
                self.act_type = act_cfg.pop('type')
                assert self.bias is not None
                self.activate = partial(fused_bias_leakyrelu, **act_cfg)
            else:
                self.act_type = 'normal'
                self.activate = MODELS.build(act_cfg)
        else:
            self.act_type = None

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, ...).

        Returns:
            Tensor: Output feature map.
        """
        if x.ndim >= 3:
            x = x.reshape(x.size(0), -1)
        x = self.linear(x)

        if self.with_activation and self.act_type == 'fused_bias':
            x = self.activate(x, self.bias * self.lr_mul)
        elif self.bias is not None and self.with_activation:
            x = self.activate(x + self.bias * self.lr_mul)
        elif self.bias is not None:
            x = x + self.bias * self.lr_mul
        elif self.with_activation:
            x = self.activate(x)

        return x


class NoiseInjection(nn.Module):
    """Noise Injection Module.

    In StyleGAN2, they adopt this module to inject spatial random noise map in
    the generators.

    Args:
        noise_weight_init (float, optional): Initialization weight for noise
            injection. Defaults to ``0.``.
    """

    def __init__(self, noise_weight_init=0.):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1).fill_(noise_weight_init))

    def forward(self, image, noise=None, return_noise=False):
        """Forward Function.

        Args:
            image (Tensor): Spatial features with a shape of (N, C, H, W).
            noise (Tensor, optional): Noises from the outside.
                Defaults to None.
            return_noise (bool, optional): Whether to return noise tensor.
                Defaults to False.

        Returns:
            Tensor: Output features.
        """
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        noise = noise.to(image.dtype)
        if return_noise:
            return image + self.weight.to(image.dtype) * noise, noise

        return image + self.weight.to(image.dtype) * noise


class ConstantInput(nn.Module):
    """Constant Input.

    In StyleGAN2, they substitute the original head noise input with such a
    constant input module.

    Args:
        channel (int): Channels for the constant input tensor.
        size (int, optional): Spatial size for the constant input.
            Defaults to 4.
    """

    def __init__(self, channel, size=4):
        super().__init__()
        if isinstance(size, int):
            size = [size, size]
        elif mmengine.is_seq_of(size, int):
            assert len(
                size
            ) == 2, f'The length of size should be 2 but got {len(size)}'
        else:
            raise ValueError(f'Got invalid value in size, {size}')

        self.input = nn.Parameter(torch.randn(1, channel, *size))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, ...).

        Returns:
            Tensor: Output feature map.
        """
        batch = x.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Blur(nn.Module):
    """Blur module.

    This module is adopted rightly after upsampling operation in StyleGAN2.

    Args:
        kernel (Array): Blur kernel/filter used in UpFIRDn.
        pad (list[int]): Padding for features.
        upsample_factor (int, optional): Upsampling factor. Defaults to 1.
    """

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map.
        """

        # In Tero's implementation, he uses fp32
        return upfirdn2d(x, self.kernel.to(x.dtype), pad=self.pad)


class AdaptiveInstanceNorm(nn.Module):
    r"""Adaptive Instance Normalization Module.

    Ref: https://github.com/rosinality/style-based-gan-pytorch/blob/master/model.py  # noqa

    Args:
        in_channel (int): The number of input's channel.
        style_dim (int): Style latent dimension.
    """

    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.affine = EqualizedLRLinearModule(style_dim, in_channel * 2)

        self.affine.bias.data[:in_channel] = 1
        self.affine.bias.data[in_channel:] = 0

    def forward(self, input, style):
        """Forward function.

        Args:
            input (Tensor): Input tensor with shape (n, c, h, w).
            style (Tensor): Input style tensor with shape (n, c).

        Returns:
            Tensor: Forward results.
        """
        style = self.affine(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class StyleConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 style_channels,
                 padding=1,
                 initial=False,
                 blur_kernel=[1, 2, 1],
                 upsample=False,
                 fused=False):
        """Convolutional style blocks composing of noise injector, AdaIN module
        and convolution layers.

        Args:
            in_channels (int): The channel number of the input tensor.
            out_channels (itn): The channel number of the output tensor.
            kernel_size (int): The kernel size of convolution layers.
            style_channels (int): The number of channels for style code.
            padding (int, optional): Padding of convolution layers.
                Defaults to 1.
            initial (bool, optional): Whether this is the first StyleConv of
                StyleGAN's generator. Defaults to False.
            blur_kernel (list, optional): The blurry kernel.
                Defaults to [1, 2, 1].
            upsample (bool, optional): Whether perform upsampling.
                Defaults to False.
            fused (bool, optional): Whether use fused upconv.
                Defaults to False.
        """
        super().__init__()

        if initial:
            self.conv1 = ConstantInput(in_channels)
        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        EqualizedLRConvUpModule(
                            in_channels,
                            out_channels,
                            kernel_size,
                            padding=padding,
                            act_cfg=dict(type='LeakyReLU',
                                         negative_slope=0.2)),
                        Blur(blur_kernel, pad=(1, 1)),
                    )
                else:
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualizedLRConvModule(
                            in_channels,
                            out_channels,
                            kernel_size,
                            padding=padding,
                            act_cfg=None), Blur(blur_kernel, pad=(1, 1)))
            else:
                self.conv1 = EqualizedLRConvModule(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    act_cfg=None)

        self.noise_injector1 = NoiseInjection()
        self.activate1 = nn.LeakyReLU(0.2)
        self.adain1 = AdaptiveInstanceNorm(out_channels, style_channels)

        self.conv2 = EqualizedLRConvModule(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            act_cfg=None)
        self.noise_injector2 = NoiseInjection()
        self.activate2 = nn.LeakyReLU(0.2)
        self.adain2 = AdaptiveInstanceNorm(out_channels, style_channels)

    def forward(self,
                x,
                style1,
                style2,
                noise1=None,
                noise2=None,
                return_noise=False):
        """Forward function.

        Args:
            x (Tensor): Input tensor.
            style1 (Tensor): Input style tensor with shape (n, c).
            style2 (Tensor): Input style tensor with shape (n, c).
            noise1 (Tensor, optional): Noise tensor with shape (n, c, h, w).
                Defaults to None.
            noise2 (Tensor, optional): Noise tensor with shape (n, c, h, w).
                Defaults to None.
            return_noise (bool, optional): If True, ``noise1`` and ``noise2``
            will be returned with ``out``. Defaults to False.

        Returns:
            Tensor | tuple[Tensor]: Forward results.
        """
        out = self.conv1(x)
        if return_noise:
            out, noise1 = self.noise_injector1(
                out, noise=noise1, return_noise=return_noise)
        else:
            out = self.noise_injector1(
                out, noise=noise1, return_noise=return_noise)
        out = self.activate1(out)
        out = self.adain1(out, style1)

        out = self.conv2(out)
        if return_noise:
            out, noise2 = self.noise_injector2(
                out, noise=noise2, return_noise=return_noise)
        else:
            out = self.noise_injector2(
                out, noise=noise2, return_noise=return_noise)
        out = self.activate2(out)
        out = self.adain2(out, style2)

        if return_noise:
            return out, noise1, noise2

        return out

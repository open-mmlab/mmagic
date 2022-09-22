# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...base_archs import conv2d, conv_transpose2d
from ..pggan import equalized_lr
from ..stylegan1 import Blur, EqualLinearActModule, NoiseInjection
from ..stylegan2.stylegan2_modules import _FusedBiasLeakyReLU


class ModulatedPEConv2d(nn.Module):
    r"""Modulated Conv2d in StyleGANv2 with Positional Encoding (PE).

    This module is modified from the ``ModulatedConv2d`` in StyleGAN2 to
    support the experiments in: Positional Encoding as Spatial Inductive Bias
    in GANs, CVPR'2021.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size, same as :obj:`nn.Con2d`.
        style_channels (int): Channels for the style codes.
        demodulate (bool, optional): Whether to adopt demodulation.
            Defaults to True.
        upsample (bool, optional): Whether to adopt upsampling in features.
            Defaults to False.
        downsample (bool, optional): Whether to adopt downsampling in features.
            Defaults to False.
        blur_kernel (list[int], optional): Blurry kernel.
            Defaults to [1, 3, 3, 1].
        equalized_lr_cfg (dict | None, optional): Configs for equalized lr.
            Defaults to dict(mode='fan_in', lr_mul=1., gain=1.).
        style_mod_cfg (dict, optional): Configs for style modulation module.
            Defaults to dict(bias_init=1.).
        style_bias (float, optional): Bias value for style code.
            Defaults to 0..
        eps (float, optional): Epsilon value to avoid computation error.
            Defaults to 1e-8.
        no_pad (bool, optional): Whether to removing the padding in
            convolution. Defaults to False.
        deconv2conv (bool, optional): Whether to substitute the transposed conv
            with (conv2d, upsampling). Defaults to False.
        interp_pad (int | None, optional): The padding number of interpolation
            pad. Defaults to None.
        up_config (dict, optional): Upsampling config.
            Defaults to dict(scale_factor=2, mode='nearest').
        up_after_conv (bool, optional): Whether to adopt upsampling after
            convolution. Defaults to False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 style_channels,
                 demodulate=True,
                 upsample=False,
                 downsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 equalized_lr_cfg=dict(mode='fan_in', lr_mul=1., gain=1.),
                 style_mod_cfg=dict(bias_init=1.),
                 style_bias=0.,
                 eps=1e-8,
                 no_pad=False,
                 deconv2conv=False,
                 interp_pad=None,
                 up_config=dict(scale_factor=2, mode='nearest'),
                 up_after_conv=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.style_channels = style_channels
        self.demodulate = demodulate
        # sanity check for kernel size
        assert isinstance(self.kernel_size,
                          int) and (self.kernel_size >= 1
                                    and self.kernel_size % 2 == 1)
        self.upsample = upsample
        self.downsample = downsample
        self.style_bias = style_bias
        self.eps = eps
        self.no_pad = no_pad
        self.deconv2conv = deconv2conv
        self.interp_pad = interp_pad
        self.with_interp_pad = interp_pad is not None
        self.up_config = deepcopy(up_config)
        self.up_after_conv = up_after_conv

        # build style modulation module
        style_mod_cfg = dict() if style_mod_cfg is None else style_mod_cfg

        self.style_modulation = EqualLinearActModule(style_channels,
                                                     in_channels,
                                                     **style_mod_cfg)
        # set lr_mul for conv weight
        lr_mul_ = 1.
        if equalized_lr_cfg is not None:
            lr_mul_ = equalized_lr_cfg.get('lr_mul', 1.)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size,
                        kernel_size).div_(lr_mul_))

        # build blurry layer for upsampling
        if upsample and not self.deconv2conv:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, (pad0, pad1), upsample_factor=factor)

        # build blurry layer for downsampling
        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        # add equalized_lr hook for conv weight
        if equalized_lr_cfg is not None:
            equalized_lr(self, **equalized_lr_cfg)

        # if `no_pad`, remove all of the padding in conv
        self.padding = kernel_size // 2 if not no_pad else 0

    def forward(self, x, style):
        """Forward function.

        Args:
            x ([Tensor): Input features with shape of (N, C, H, W).
            style (Tensor): Style latent with shape of (N, C).

        Returns:
            Tensor: Output feature with shape of (N, C, H, W).
        """
        n, c, h, w = x.shape
        # process style code
        style = self.style_modulation(style).view(n, 1, c, 1,
                                                  1) + self.style_bias

        # combine weight and style
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(n, self.out_channels, 1, 1, 1)

        weight = weight.view(n * self.out_channels, c, self.kernel_size,
                             self.kernel_size)

        if self.upsample and not self.deconv2conv:
            x = x.reshape(1, n * c, h, w)
            weight = weight.view(n, self.out_channels, c, self.kernel_size,
                                 self.kernel_size)
            weight = weight.transpose(1, 2).reshape(n * c, self.out_channels,
                                                    self.kernel_size,
                                                    self.kernel_size)
            x = conv_transpose2d(x, weight, padding=0, stride=2, groups=n)
            x = x.reshape(n, self.out_channels, *x.shape[-2:])
            x = self.blur(x)
        elif self.upsample and self.deconv2conv:
            if self.up_after_conv:
                x = x.reshape(1, n * c, h, w)
                x = conv2d(x, weight, padding=self.padding, groups=n)
                x = x.view(n, self.out_channels, *x.shape[2:4])

            if self.with_interp_pad:
                h_, w_ = x.shape[-2:]
                up_cfg_ = deepcopy(self.up_config)
                up_scale = up_cfg_.pop('scale_factor')
                size_ = (h_ * up_scale + self.interp_pad,
                         w_ * up_scale + self.interp_pad)
                x = F.interpolate(x, size=size_, **up_cfg_)
            else:
                x = F.interpolate(x, **self.up_config)

            if not self.up_after_conv:
                h_, w_ = x.shape[-2:]
                x = x.view(1, n * c, h_, w_)
                x = conv2d(x, weight, padding=self.padding, groups=n)
                x = x.view(n, self.out_channels, *x.shape[2:4])

        elif self.downsample:
            x = self.blur(x)
            x = x.view(1, n * self.in_channels, *x.shape[-2:])
            x = conv2d(x, weight, stride=2, padding=0, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])
        else:
            x = x.view(1, n * c, h, w)
            x = conv2d(x, weight, stride=1, padding=self.padding, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])

        return x


class ModulatedPEStyleConv(nn.Module):
    """Modulated Style Convolution with Positional Encoding.

    This module is modified from the ``ModulatedStyleConv`` in StyleGAN2 to
    support the experiments in: Positional Encoding as Spatial Inductive Bias
    in GANs, CVPR'2021.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size, same as :obj:`nn.Con2d`.
        style_channels (int): Channels for the style codes.
        demodulate (bool, optional): Whether to adopt demodulation.
            Defaults to True.
        upsample (bool, optional): Whether to adopt upsampling in features.
            Defaults to False.
        downsample (bool, optional): Whether to adopt downsampling in features.
            Defaults to False.
        blur_kernel (list[int], optional): Blurry kernel.
            Defaults to [1, 3, 3, 1].
        equalized_lr_cfg (dict | None, optional): Configs for equalized lr.
            Defaults to dict(mode='fan_in', lr_mul=1., gain=1.).
        style_mod_cfg (dict, optional): Configs for style modulation module.
            Defaults to dict(bias_init=1.).
        style_bias (float, optional): Bias value for style code.
            Defaults to 0..
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 style_channels,
                 upsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 demodulate=True,
                 style_mod_cfg=dict(bias_init=1.),
                 style_bias=0.,
                 **kwargs):
        super().__init__()

        self.conv = ModulatedPEConv2d(
            in_channels,
            out_channels,
            kernel_size,
            style_channels,
            demodulate=demodulate,
            upsample=upsample,
            blur_kernel=blur_kernel,
            style_mod_cfg=style_mod_cfg,
            style_bias=style_bias,
            **kwargs)

        self.noise_injector = NoiseInjection()
        self.activate = _FusedBiasLeakyReLU(out_channels)

    def forward(self, x, style, noise=None, return_noise=False):
        """Forward Function.

        Args:
            x ([Tensor): Input features with shape of (N, C, H, W).
            style (Tensor): Style latent with shape of (N, C).
            noise (Tensor, optional): Noise for injection. Defaults to None.
            return_noise (bool, optional): Whether to return noise tensors.
                Defaults to False.

        Returns:
            Tensor: Output features with shape of (N, C, H, W)
        """
        out = self.conv(x, style)
        if return_noise:
            out, noise = self.noise_injector(
                out, noise=noise, return_noise=return_noise)
        else:
            out = self.noise_injector(
                out, noise=noise, return_noise=return_noise)

        out = self.activate(out)

        if return_noise:
            return out, noise

        return out

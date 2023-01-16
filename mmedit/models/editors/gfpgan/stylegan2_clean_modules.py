# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.runner.amp import autocast

from ..stylegan1 import Blur, EqualLinearActModule, NoiseInjection
from ..stylegan2.stylegan2_modules import UpsampleUpFIRDn


class ModulatedConv2d(nn.Module):
    r"""Modulated Conv2d in StyleGANv2.

    This module implements the modulated convolution layers proposed in
    StyleGAN2. Details can be found in Analyzing and Improving the Image
    Quality of StyleGAN, CVPR2020.

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
    """

    def __init__(
            self,
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
            padding=None,  # self define padding
            eps=1e-8,
            fp16_enabled=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.style_channels = style_channels
        self.demodulate = demodulate
        self.fp16_enabled = fp16_enabled
        # sanity check for kernel size
        assert isinstance(self.kernel_size,
                          int) and (self.kernel_size >= 1
                                    and self.kernel_size % 2 == 1)
        self.upsample = upsample
        self.downsample = downsample
        self.style_bias = style_bias
        self.eps = eps

        # build style modulation module
        style_mod_cfg = dict() if style_mod_cfg is None else style_mod_cfg

        self.style_modulation = EqualLinearActModule(
            style_channels,
            in_channels,
            equalized_lr_cfg=None,
            **style_mod_cfg)
        # set lr_mul for conv weight
        lr_mul_ = 1.
        if equalized_lr_cfg is not None:
            lr_mul_ = equalized_lr_cfg.get('lr_mul', 1.)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size,
                        kernel_size).div_(lr_mul_))

        # build blurry layer for upsampling
        if upsample:
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
        # if equalized_lr_cfg is not None:
        #     equalized_lr(self, **equalized_lr_cfg)

        self.padding = padding if padding else (kernel_size // 2)

    def forward(self, x, style, input_gain=None):
        n, c, h, w = x.shape

        weight = self.weight
        # Pre-normalize inputs to avoid FP16 overflow.
        # if x.dtype == torch.float16 and self.demodulate:
        if self.fp16_enabled and self.demodulate:
            weight = weight * (
                1 / np.sqrt(
                    self.in_channels * self.kernel_size * self.kernel_size) /
                weight.norm(float('inf'), dim=[1, 2, 3], keepdim=True)
            )  # max_Ikk
            style = style / style.norm(
                float('inf'), dim=1, keepdim=True)  # max_I

        with autocast(enabled=self.fp16_enabled):
            # process style code
            style = self.style_modulation(style).view(n, 1, c, 1,
                                                      1) + self.style_bias
            # combine weight and style
            weight = weight * style
            if self.demodulate:
                demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
                weight = weight * demod.view(n, self.out_channels, 1, 1, 1)

            if input_gain is not None:
                # input_gain shape [batch, in_ch]
                input_gain = input_gain.expand(n, self.in_channels)
                # weight shape [batch, out_ch, in_ch, kernel_size, kernel_size]
                weight = weight * input_gain.unsqueeze(1).unsqueeze(
                    3).unsqueeze(4)

            weight = weight.view(n * self.out_channels, c, self.kernel_size,
                                 self.kernel_size)

            if self.fp16_enabled:
                weight = weight.to(torch.float16)
                x = x.to(torch.float16)

            if self.upsample:
                x = F.interpolate(
                    x, scale_factor=2, mode='bilinear', align_corners=False)
            elif self.downsample:
                x = F.interpolate(
                    x, scale_factor=0.5, mode='bilinear', align_corners=False)

            b, c, h, w = x.shape
            x = x.view(1, b * c, h, w)
            # weight: (b*c_out, c_in, k, k), groups=b
            out = F.conv2d(x, weight, padding=self.padding, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])
            x = out
        return x


class ModulatedStyleConv(nn.Module):
    """Modulated Style Convolution.

    In this module, we integrate the modulated conv2d, noise injector and
    activation layers into together.

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
            Defaults to ``0.``.
        fp16_enabled (bool, optional): Whether to use fp16 training in this
            module. Defaults to False.
        conv_clamp (float, optional): Clamp the convolutional layer results to
            avoid gradient overflow. Defaults to `256.0`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 style_channels,
                 upsample=False,
                 demodulate=True,
                 style_mod_cfg=dict(bias_init=1.),
                 style_bias=0.,
                 fp16_enabled=False,
                 conv_clamp=256):
        super().__init__()

        # add support for fp16
        self.fp16_enabled = fp16_enabled
        self.conv_clamp = float(conv_clamp)

        self.conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            style_channels,
            demodulate=demodulate,
            equalized_lr_cfg=None,
            upsample=upsample,
            style_mod_cfg=style_mod_cfg,
            style_bias=style_bias,
            fp16_enabled=fp16_enabled)

        self.noise_injector = NoiseInjection()
        self.activate = FusedBiasLeakyReLU(out_channels)

    def forward(self,
                x,
                style,
                noise=None,
                add_noise=True,
                return_noise=False):
        """Forward Function.

        Args:
            x ([Tensor): Input features with shape of (N, C, H, W).
            style (Tensor): Style latent with shape of (N, C).
            noise (Tensor, optional): Noise for injection. Defaults to None.
            add_noise (bool, optional): Whether apply noise injection to
                feature. Defaults to True.
            return_noise (bool, optional): Whether to return noise tensors.
                Defaults to False.

        Returns:
            Tensor: Output features with shape of (N, C, H, W)
        """
        with autocast(enabled=self.fp16_enabled):
            # TODO
            out = self.conv(x, style) * 2**0.5

            if add_noise:
                if return_noise:
                    out, noise = self.noise_injector(
                        out, noise=noise, return_noise=return_noise)
                else:
                    out = self.noise_injector(
                        out, noise=noise, return_noise=return_noise)

            # TODO: FP16 in activate layers
            # TODO
            # out = self.activate(out)
            bias = self.activate.bias.unsqueeze(0)
            bias = bias.unsqueeze(2)
            bias = bias.unsqueeze(3)
            bias = bias.repeat(1, 1, out.shape[2], out.shape[3])
            out = F.leaky_relu(out + bias, negative_slope=0.2)

            if self.fp16_enabled:
                out = torch.clamp(
                    out, min=-self.conv_clamp, max=self.conv_clamp)

        if return_noise:
            return out, noise

        return out


# TODO need to check the unconsistency with mmedit FusedBiasLeakyReLU
class FusedBiasLeakyReLU(nn.Module):
    r"""Fused bias leaky ReLU.

    This function is introduced in the StyleGAN2:
    `Analyzing and Improving the Image Quality of StyleGAN
    <http://arxiv.org/abs/1912.04958>`_

    The bias term comes from the convolution operation. In addition, to keep
    the variance of the feature map or gradients unchanged, they also adopt a
    scale similarly with Kaiming initialization. However, since the
    :math:`1+{alpha}^2` is too small, we can just ignore it. Therefore, the
    final scale is just :math:`\sqrt{2}`. Of course, you may change it with
    your own scale.

    TODO: Implement the CPU version.

    Args:
        num_channels (int): The channel number of the feature map.
        negative_slope (float, optional): Same as nn.LeakyRelu.
            Defaults to 0.2.
        scale (float, optional): A scalar to adjust the variance of the feature
            map. Defaults to 2**0.5.
    """

    def __init__(self,
                 num_channels: int,
                 negative_slope: float = 0.2,
                 scale: float = 2**0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        bias = bias.repeat(1, 1, input.shape[2], input.shape[3])
        input = F.leaky_relu(input + bias, negative_slope=0.2)
        return input


class ModulatedToRGB(nn.Module):
    """To RGB layer.

    This module is designed to output image tensor in StyleGAN2.

    Args:
        in_channels (int): Input channels.
        style_channels (int): Channels for the style codes.
        out_channels (int, optional): Output channels. Defaults to 3.
        upsample (bool, optional): Whether to adopt upsampling in features.
            Defaults to False.
        blur_kernel (list[int], optional): Blurry kernel.
            Defaults to [1, 3, 3, 1].
        style_mod_cfg (dict, optional): Configs for style modulation module.
            Defaults to dict(bias_init=1.).
        style_bias (float, optional): Bias value for style code.
            Defaults to 0..
        fp16_enabled (bool, optional): Whether to use fp16 training in this
            module. Defaults to False.
        conv_clamp (float, optional): Clamp the convolutional layer results to
            avoid gradient overflow. Defaults to `256.0`.
        out_fp32 (bool, optional): Whether to convert the output feature map to
            `torch.float32`. Defaults to `True`.
    """

    def __init__(self,
                 in_channels,
                 style_channels,
                 out_channels=3,
                 upsample=True,
                 blur_kernel=[1, 3, 3, 1],
                 style_mod_cfg=dict(bias_init=1.),
                 style_bias=0.,
                 fp16_enabled=False,
                 conv_clamp=256,
                 out_fp32=True):
        super().__init__()

        if upsample:
            self.upsample = UpsampleUpFIRDn(blur_kernel)

        # add support for fp16
        self.fp16_enabled = fp16_enabled
        self.conv_clamp = float(conv_clamp)

        self.conv = ModulatedConv2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=1,
            style_channels=style_channels,
            equalized_lr_cfg=None,
            demodulate=False,
            style_mod_cfg=style_mod_cfg,
            style_bias=style_bias,
            fp16_enabled=fp16_enabled)

        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        # enforece the output to be fp32 (follow Tero's implementation)
        self.out_fp32 = out_fp32

    # @auto_fp16(apply_to=('x', 'style'))
    def forward(self, x, style, skip=None):
        """Forward Function.

        Args:
            x ([Tensor): Input features with shape of (N, C, H, W).
            style (Tensor): Style latent with shape of (N, C).
            skip (Tensor, optional): Tensor for skip link. Defaults to None.

        Returns:
            Tensor: Output features with shape of (N, C, H, W)
        """
        with autocast(enabled=self.fp16_enabled):
            out = self.conv(x, style)
            out = out + self.bias.to(x.dtype)

            if self.fp16_enabled:
                out = torch.clamp(
                    out, min=-self.conv_clamp, max=self.conv_clamp)

            # Here, Tero adopts FP16 at `skip`.
            if skip is not None:
                if hasattr(self, 'upsample'):
                    skip = self.upsample(skip)
                out = out + skip
        if self.out_fp32:
            out = out.to(torch.float32)
        return out

# Copyright (c) OpenMMLab. All rights reserved.
import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmcv.ops.fused_bias_leakyrelu import (FusedBiasLeakyReLU,
                                           fused_bias_leakyrelu)
from mmcv.ops.upfirdn2d import upfirdn2d
from mmengine.dist import get_dist_info
from mmengine.runner.amp import autocast

from mmedit.models.base_archs import AllGatherLayer
from ...base_archs import conv2d, conv_transpose2d
from ..pggan import EqualizedLRConvModule, equalized_lr
from ..stylegan1 import Blur, EqualLinearActModule, NoiseInjection, make_kernel


class _FusedBiasLeakyReLU(FusedBiasLeakyReLU):
    """Wrap FusedBiasLeakyReLU to support FP16 training."""

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, ...).

        Returns:
            Tensor: Output feature map.
        """
        return fused_bias_leakyrelu(x, self.bias.to(x.dtype),
                                    self.negative_slope, self.scale)


class UpsampleUpFIRDn(nn.Module):
    """UpFIRDn for Upsampling.

    This module is used in the ``to_rgb`` layers in StyleGAN2 for upsampling
    the images.

    Args:
        kernel (Array): Blur kernel/filter used in UpFIRDn.
        factor (int, optional): Upsampling factor. Defaults to 2.
    """

    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor**2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map.
        """
        out = upfirdn2d(
            x, self.kernel.to(x.dtype), up=self.factor, down=1, pad=self.pad)

        return out


class DownsampleUpFIRDn(nn.Module):
    """UpFIRDn for Downsampling.

    This module is mentioned in StyleGAN2 for dowampling the feature maps.

    Args:
        kernel (Array): Blur kernel/filter used in UpFIRDn.
        factor (int, optional): Downsampling factor. Defaults to 2.
    """

    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        """Forward function.

        Args:
            input (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map.
        """
        out = upfirdn2d(
            input,
            self.kernel.to(input.dtype),
            up=1,
            down=self.factor,
            pad=self.pad)

        return out


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
        if equalized_lr_cfg is not None:
            equalized_lr(self, **equalized_lr_cfg)

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
                x = x.reshape(1, n * c, h, w)
                weight = weight.view(n, self.out_channels, c, self.kernel_size,
                                     self.kernel_size)
                weight = weight.transpose(1,
                                          2).reshape(n * c, self.out_channels,
                                                     self.kernel_size,
                                                     self.kernel_size)
                x = conv_transpose2d(x, weight, padding=0, stride=2, groups=n)
                x = x.reshape(n, self.out_channels, *x.shape[-2:])
                x = self.blur(x)

            elif self.downsample:
                x = self.blur(x)
                x = x.view(1, n * self.in_channels, *x.shape[-2:])
                x = conv2d(x, weight, stride=2, padding=0, groups=n)
                x = x.view(n, self.out_channels, *x.shape[-2:])
            else:
                x = x.reshape(1, n * c, h, w)
                x = conv2d(x, weight, stride=1, padding=self.padding, groups=n)
                x = x.view(n, self.out_channels, *x.shape[-2:])
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
                 blur_kernel=[1, 3, 3, 1],
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
            upsample=upsample,
            blur_kernel=blur_kernel,
            style_mod_cfg=style_mod_cfg,
            style_bias=style_bias,
            fp16_enabled=fp16_enabled)

        self.noise_injector = NoiseInjection()
        self.activate = _FusedBiasLeakyReLU(out_channels)

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
            out = self.conv(x, style)

            if add_noise:
                if return_noise:
                    out, noise = self.noise_injector(
                        out, noise=noise, return_noise=return_noise)
                else:
                    out = self.noise_injector(
                        out, noise=noise, return_noise=return_noise)

            # TODO: FP16 in activate layers
            out = self.activate(out)

            if self.fp16_enabled:
                out = torch.clamp(
                    out, min=-self.conv_clamp, max=self.conv_clamp)

        if return_noise:
            return out, noise

        return out


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


class ConvDownLayer(nn.Sequential):
    """Convolution and Downsampling layer.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size, same as :obj:`nn.Con2d`.
        downsample (bool, optional): Whether to adopt downsampling in features.
            Defaults to False.
        blur_kernel (list[int], optional): Blurry kernel.
            Defaults to [1, 3, 3, 1].
        bias (bool, optional): Whether to use bias parameter. Defaults to True.
        act_cfg (dict, optional): Activation configs.
            Defaults to dict(type='fused_bias').
        fp16_enabled (bool, optional): Whether to use fp16 training in this
            module. Defaults to False.
        conv_clamp (float, optional): Clamp the convolutional layer results to
            avoid gradient overflow. Defaults to `256.0`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 downsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 bias=True,
                 act_cfg=dict(type='fused_bias'),
                 fp16_enabled=False,
                 conv_clamp=256.):

        self.fp16_enabled = fp16_enabled
        self.conv_clamp = float(conv_clamp)
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2

        self.with_fused_bias = act_cfg is not None and act_cfg.get(
            'type') == 'fused_bias'
        if self.with_fused_bias:
            conv_act_cfg = None
        else:
            conv_act_cfg = act_cfg
        layers.append(
            EqualizedLRConvModule(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not self.with_fused_bias,
                norm_cfg=None,
                act_cfg=conv_act_cfg,
                equalized_lr_cfg=dict(mode='fan_in', gain=1.)))
        if self.with_fused_bias:
            layers.append(_FusedBiasLeakyReLU(out_channels))

        super(ConvDownLayer, self).__init__(*layers)

    # @auto_fp16(apply_to=('x', ))
    def forward(self, x):
        with autocast(enabled=self.fp16_enabled):
            x = super().forward(x)
            if self.fp16_enabled:
                x = torch.clamp(x, min=-self.conv_clamp, max=self.conv_clamp)
        return x


class ResBlock(nn.Module):
    """Residual block used in the discriminator of StyleGAN2.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size, same as :obj:`nn.Con2d`.
        fp16_enabled (bool, optional): Whether to use fp16 training in this
            module. Defaults to False.
        convert_input_fp32 (bool, optional): Whether to convert input type to
            fp32 if not `fp16_enabled`. This argument is designed to deal with
            the cases where some modules are run in FP16 and others in FP32.
            Defaults to True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 blur_kernel=[1, 3, 3, 1],
                 fp16_enabled=False,
                 convert_input_fp32=True):
        super().__init__()

        self.fp16_enabled = fp16_enabled
        self.convert_input_fp32 = convert_input_fp32

        self.conv1 = ConvDownLayer(
            in_channels,
            in_channels,
            3,
            fp16_enabled=fp16_enabled,
            blur_kernel=blur_kernel)
        self.conv2 = ConvDownLayer(
            in_channels,
            out_channels,
            3,
            downsample=True,
            fp16_enabled=fp16_enabled,
            blur_kernel=blur_kernel)

        self.skip = ConvDownLayer(
            in_channels,
            out_channels,
            1,
            downsample=True,
            act_cfg=None,
            bias=False,
            fp16_enabled=fp16_enabled,
            blur_kernel=blur_kernel)

    def forward(self, input):
        """Forward function.

        Args:
            input (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map.
        """
        # TODO: study whether this explicit datatype transfer will harm the
        # apex training speed
        if not self.fp16_enabled and self.convert_input_fp32:
            input = input.to(torch.float32)

        with autocast(enabled=self.fp16_enabled):
            out = self.conv1(input)
            out = self.conv2(out)

            skip = self.skip(input)
            out = (out + skip) / np.sqrt(2)

        return out


class ModMBStddevLayer(nn.Module):
    """Modified MiniBatch Stddev Layer.

    This layer is modified from ``MiniBatchStddevLayer`` used in PGGAN. In
    StyleGAN2, the authors add a new feature, `channel_groups`, into this
    layer.

    Note that to accelerate the training procedure, we also add a new feature
    of ``sync_std`` to achieve multi-nodes/machine training. This feature is
    still in beta version and we have tested it on 256 scales.

    Args:
        group_size (int, optional): The size of groups in batch dimension.
            Defaults to 4.
        channel_groups (int, optional): The size of groups in channel
            dimension. Defaults to 1.
        sync_std (bool, optional): Whether to use synchronized std feature.
            Defaults to False.
        sync_groups (int | None, optional): The size of groups in node
            dimension. Defaults to None.
        eps (float, optional): Epsilon value to avoid computation error.
            Defaults to 1e-8.
    """

    def __init__(self,
                 group_size=4,
                 channel_groups=1,
                 sync_std=False,
                 sync_groups=None,
                 eps=1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = eps
        self.channel_groups = channel_groups
        self.sync_std = sync_std
        self.sync_groups = group_size if sync_groups is None else sync_groups

        if self.sync_std:
            assert torch.distributed.is_initialized(
            ), 'Only in distributed training can the sync_std be activated.'
            mmengine.print_log('Adopt synced minibatch stddev layer', 'mmgen')

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map with shape of (N, C+1, H, W).
        """

        if self.sync_std:
            # concatenate all features
            all_features = torch.cat(AllGatherLayer.apply(x), dim=0)
            # get the exact features we need in calculating std-dev
            rank, ws = get_dist_info()
            local_bs = all_features.shape[0] // ws
            start_idx = local_bs * rank
            # avoid the case where start idx near the tail of features
            if start_idx + self.sync_groups > all_features.shape[0]:
                start_idx = all_features.shape[0] - self.sync_groups
            end_idx = min(local_bs * rank + self.sync_groups,
                          all_features.shape[0])

            x = all_features[start_idx:end_idx]

        # batch size should be smaller than or equal to group size. Otherwise,
        # batch size should be divisible by the group size.
        assert x.shape[
            0] <= self.group_size or x.shape[0] % self.group_size == 0, (
                'Batch size be smaller than or equal '
                'to group size. Otherwise,'
                ' batch size should be divisible by the group size.'
                f'But got batch size {x.shape[0]},'
                f' group size {self.group_size}')
        assert x.shape[1] % self.channel_groups == 0, (
            '"channel_groups" must be divided by the feature channels. '
            f'channel_groups: {self.channel_groups}, '
            f'feature channels: {x.shape[1]}')

        n, c, h, w = x.shape
        group_size = min(n, self.group_size)
        # [G, M, Gc, C', H, W]
        y = torch.reshape(x, (group_size, -1, self.channel_groups,
                              c // self.channel_groups, h, w))
        y = torch.var(y, dim=0, unbiased=False)
        y = torch.sqrt(y + self.eps)
        # [M, 1, 1, 1]
        y = y.mean(dim=(2, 3, 4), keepdim=True).squeeze(2)
        y = y.repeat(group_size, 1, h, w)
        return torch.cat([x, y], dim=1)

# Copyright (c) OpenMMLab. All rights reserved.
import math
from copy import deepcopy
from functools import partial

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.cnn.utils import normal_init
from mmcv.ops.fused_bias_leakyrelu import (FusedBiasLeakyReLU,
                                           fused_bias_leakyrelu)
from mmcv.ops.upfirdn2d import upfirdn2d
from packaging import version
from torch.nn.init import _calculate_correct_fan


def pixel_norm(x, eps=1e-6):
    """Pixel Normalization.

    This normalization is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Args:
        x (torch.Tensor): Tensor to be normalized.
        eps (float, optional): Epsilon to avoid dividing zero.
            Defaults to 1e-6.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    if version.parse(torch.__version__) >= version.parse('1.7.0'):
        norm = torch.linalg.norm(x, ord=2, dim=1, keepdim=True)
    # support older pytorch version
    else:
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
    norm = norm / torch.sqrt(torch.tensor(x.shape[1]).to(x))

    return x / (norm + eps)


class PixelNorm(nn.Module):
    """Pixel Normalization.

    This module is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Args:
        eps (float, optional): Epsilon value. Defaults to 1e-6.
    """

    _abbr_ = 'pn'

    def __init__(self, in_channels=None, eps=1e-6):
        super(PixelNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        return pixel_norm(x, self.eps)


class EqualizedLR:
    r"""Equalized Learning Rate.

    This trick is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    The general idea is to dynamically rescale the weight in training instead
    of in initializing so that the variance of the responses in each layer is
    guaranteed with some statistical properties.

    Note that this function is always combined with a convolution module which
    is initialized with :math:`\mathcal{N}(0, 1)`.

    Args:
        name (str | optional): The name of weights. Defaults to 'weight'.
        mode (str, optional): The mode of computing ``fan`` which is the
            same as ``kaiming_init`` in pytorch. You can choose one from
            ['fan_in', 'fan_out']. Defaults to 'fan_in'.
    """

    def __init__(self, name='weight', gain=2**0.5, mode='fan_in', lr_mul=1.0):
        self.name = name
        self.mode = mode
        self.gain = gain
        self.lr_mul = lr_mul

    def compute_weight(self, module):
        """Compute weight with equalized learning rate.

        Args:
            module (nn.Module): A module that is wrapped with equalized lr.

        Returns:
            torch.Tensor: Updated weight.
        """
        weight = getattr(module, self.name + '_orig')
        if weight.ndim == 5:
            # weight in shape of [b, out, in, k, k]
            fan = _calculate_correct_fan(weight[0], self.mode)
        else:
            assert weight.ndim <= 4
            fan = _calculate_correct_fan(weight, self.mode)
        weight = weight * torch.tensor(
            self.gain, device=weight.device) * torch.sqrt(
                torch.tensor(1. / fan, device=weight.device)) * self.lr_mul

        return weight

    def __call__(self, module, inputs):
        """Standard interface for forward pre hooks."""
        setattr(module, self.name, self.compute_weight(module))

    @staticmethod
    def apply(module, name, gain=2**0.5, mode='fan_in', lr_mul=1.):
        """Apply function.

        This function is to register an equalized learning rate hook in an
        ``nn.Module``.

        Args:
            module (nn.Module): Module to be wrapped.
            name (str | optional): The name of weights. Defaults to 'weight'.
            mode (str, optional): The mode of computing ``fan`` which is the
                same as ``kaiming_init`` in pytorch. You can choose one from
                ['fan_in', 'fan_out']. Defaults to 'fan_in'.

        Returns:
            nn.Module: Module that is registered with equalized lr hook.
        """
        # sanity check for duplicated hooks.
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, EqualizedLR):
                raise RuntimeError(
                    'Cannot register two equalized_lr hooks on the same '
                    f'parameter {name} in {module} module.')

        fn = EqualizedLR(name, gain=gain, mode=mode, lr_mul=lr_mul)
        weight = module._parameters[name]

        delattr(module, name)
        module.register_parameter(name + '_orig', weight)

        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a
        # plain attribute.

        setattr(module, name, weight.data)
        module.register_forward_pre_hook(fn)

        # TODO: register load state dict hook

        return fn


def equalized_lr(module, name='weight', gain=2**0.5, mode='fan_in', lr_mul=1.):
    r"""Equalized Learning Rate.

    This trick is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    The general idea is to dynamically rescale the weight in training instead
    of in initializing so that the variance of the responses in each layer is
    guaranteed with some statistical properties.

    Note that this function is always combined with a convolution module which
    is initialized with :math:`\mathcal{N}(0, 1)`.

    Args:
        module (nn.Module): Module to be wrapped.
        name (str | optional): The name of weights. Defaults to 'weight'.
        mode (str, optional): The mode of computing ``fan`` which is the
            same as ``kaiming_init`` in pytorch. You can choose one from
            ['fan_in', 'fan_out']. Defaults to 'fan_in'.

    Returns:
        nn.Module: Module that is registered with equalized lr hook.
    """
    EqualizedLR.apply(module, name, gain=gain, mode=mode, lr_mul=lr_mul)

    return module


class EqualizedLRConvModule(ConvModule):
    r"""Equalized LR ConvModule.

    In this module, we inherit default ``mmcv.cnn.ConvModule`` and adopt
    equalized lr in convolution. The equalized learning rate is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Note that, the initialization of ``self.conv`` will be overwritten as
    :math:`\mathcal{N}(0, 1)`.

    Args:
        equalized_lr_cfg (dict | None, optional): Config for ``EqualizedLR``.
            If ``None``, equalized learning rate is ignored. Defaults to
            dict(mode='fan_in').
    """

    def __init__(self, *args, equalized_lr_cfg=dict(mode='fan_in'), **kwargs):
        super(EqualizedLRConvModule, self).__init__(*args, **kwargs)
        self.with_equlized_lr = equalized_lr_cfg is not None
        if self.with_equlized_lr:
            self.conv = equalized_lr(self.conv, **equalized_lr_cfg)
            # initialize the conv weight with standard Gaussian noise.
            self._init_conv_weights()

    def _init_conv_weights(self):
        """Initialize conv weights as described in PGGAN."""
        normal_init(self.conv)


class EqualizedLRLinearModule(nn.Linear):
    r"""Equalized LR LinearModule.

    In this module, we adopt equalized lr in ``nn.Linear``. The equalized
    learning rate is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Note that, the initialization of ``self.weight`` will be overwritten as
    :math:`\mathcal{N}(0, 1)`.

    Args:
        equalized_lr_cfg (dict | None, optional): Config for ``EqualizedLR``.
            If ``None``, equalized learning rate is ignored. Defaults to
            dict(mode='fan_in').
    """

    def __init__(self, *args, equalized_lr_cfg=dict(mode='fan_in'), **kwargs):
        super(EqualizedLRLinearModule, self).__init__(*args, **kwargs)
        self.with_equlized_lr = equalized_lr_cfg is not None
        if self.with_equlized_lr:
            self.lr_mul = equalized_lr_cfg.get('lr_mul', 1.)
        else:
            # In fact, lr_mul will only be used in EqualizedLR for
            # initialization
            self.lr_mul = 1.
        if self.with_equlized_lr:
            equalized_lr(self, **equalized_lr_cfg)
            self._init_linear_weights()

    def _init_linear_weights(self):
        """Initialize linear weights as described in PGGAN."""
        nn.init.normal_(self.weight, 0, 1. / self.lr_mul)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)


class EqualLinearActModule(nn.Module):
    """Equalized LR Linear Module with Activation Layer.

    Args:
        nn ([type]): [description]
    """

    def __init__(self,
                 *args,
                 equalized_lr_cfg=dict(gain=1., lr_mul=1.),
                 bias=True,
                 bias_init=0.,
                 act_cfg=None,
                 **kwargs):
        super(EqualLinearActModule, self).__init__()
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
                self.activate = build_activation_layer(act_cfg)
        else:
            self.act_type = None

    def forward(self, x):
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


def _make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class UpsampleUpFIRDn(nn.Module):

    def __init__(self, kernel, factor=2):
        super(UpsampleUpFIRDn, self).__init__()

        self.factor = factor
        kernel = _make_kernel(kernel) * (factor**2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, x):
        out = upfirdn2d(x, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class DonwsampleUpFIRDn(nn.Module):

    def __init__(self, kernel, factor=2):
        super(DonwsampleUpFIRDn, self).__init__()

        self.factor = factor
        kernel = _make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(
            input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):

    def __init__(self, kernel, pad, upsample_factor=1):
        super(Blur, self).__init__()
        kernel = _make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, x):
        return upfirdn2d(x, self.kernel, pad=self.pad)


class ModulatedConv2d(nn.Module):
    r"""Modulated Conv2d in StyleGANv2.

    Attention:

    #. ``style_bias`` is provided to check the difference between official TF
       implementation and other PyTorch implementation.
       In TF, Tero explicitly add the ``1.`` after style code, while unofficial
       implementation adopts bias initialization with ``1.``.
       Details can be found in:
       https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py#L214
       https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py#L99
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
                 eps=1e-8):
        super(ModulatedConv2d, self).__init__()
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

        self.padding = kernel_size // 2

    def forward(self, x, style):
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

        if self.upsample:
            x = x.reshape(1, n * c, h, w)
            weight = weight.view(n, self.out_channels, c, self.kernel_size,
                                 self.kernel_size)
            weight = weight.transpose(1, 2).reshape(n * c, self.out_channels,
                                                    self.kernel_size,
                                                    self.kernel_size)
            x = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=n)
            x = x.reshape(n, self.out_channels, *x.shape[-2:])
            x = self.blur(x)

        elif self.downsample:
            x = self.blur(x)
            x = x.view(1, n * self.in_channels, *x.shape[-2:])
            x = F.conv2d(x, weight, stride=2, padding=0, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])
        else:
            x = x.view(1, n * c, h, w)
            x = F.conv2d(x, weight, stride=1, padding=self.padding, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])

        return x


class NoiseInjection(nn.Module):

    def __init__(self, noise_weight_init=0.):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1).fill_(noise_weight_init))

    def forward(self, image, noise=None, return_noise=False):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        if return_noise:
            return image + self.weight * noise, noise

        return image + self.weight * noise


class ConstantInput(nn.Module):

    def __init__(self, channel, size=4):
        super().__init__()
        if isinstance(size, int):
            size = [size, size]
        elif mmcv.is_seq_of(size, int):
            assert len(
                size
            ) == 2, f'The length of size should be 2 but got {len(size)}'
        else:
            raise ValueError(f'Got invalid value in size, {size}')

        self.input = nn.Parameter(torch.randn(1, channel, *size))

    def forward(self, x):
        batch = x.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class ModulatedPEConv2d(nn.Module):
    r"""Modulated Conv2d in StyleGANv2.

    Attention:

    #. ``style_bias`` is provided to check the difference between official TF
       implementation and other PyTorch implementation.
       In TF, Tero explicitly add the ``1.`` after style code, while unofficial
       implementation adopts bias initialization with ``1.``.
       Details can be found in:
       https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py#L214
       https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py#L99
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
        super(ModulatedPEConv2d, self).__init__()
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
            x = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=n)
            x = x.reshape(n, self.out_channels, *x.shape[-2:])
            x = self.blur(x)
        elif self.upsample and self.deconv2conv:
            if self.up_after_conv:
                x = x.reshape(1, n * c, h, w)
                x = F.conv2d(x, weight, padding=self.padding, groups=n)
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
                x = F.conv2d(x, weight, padding=self.padding, groups=n)
                x = x.view(n, self.out_channels, *x.shape[2:4])

        elif self.downsample:
            x = self.blur(x)
            x = x.view(1, n * self.in_channels, *x.shape[-2:])
            x = F.conv2d(x, weight, stride=2, padding=0, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])
        else:
            x = x.view(1, n * c, h, w)
            x = F.conv2d(x, weight, stride=1, padding=self.padding, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])

        return x


class ModulatedStyleConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 style_channels,
                 upsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 demodulate=True,
                 style_mod_cfg=dict(bias_init=1.),
                 style_bias=0.):
        super(ModulatedStyleConv, self).__init__()

        self.conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            style_channels,
            demodulate=demodulate,
            upsample=upsample,
            blur_kernel=blur_kernel,
            style_mod_cfg=style_mod_cfg,
            style_bias=style_bias)

        self.noise_injector = NoiseInjection()
        self.activate = FusedBiasLeakyReLU(out_channels)

    def forward(self, x, style, noise=None, return_noise=False):
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
        else:
            return out


class ModulatedPEStyleConv(nn.Module):

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
        super(ModulatedPEStyleConv, self).__init__()

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
        self.activate = FusedBiasLeakyReLU(out_channels)

    def forward(self, x, style, noise=None, return_noise=False):
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
        else:
            return out


class ModulatedToRGB(nn.Module):

    def __init__(self,
                 in_channels,
                 style_channels,
                 out_channels=3,
                 upsample=True,
                 blur_kernel=[1, 3, 3, 1],
                 style_mod_cfg=dict(bias_init=1.),
                 style_bias=0.):
        super(ModulatedToRGB, self).__init__()

        if upsample:
            self.upsample = UpsampleUpFIRDn(blur_kernel)

        self.conv = ModulatedConv2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=1,
            style_channels=style_channels,
            demodulate=False,
            style_mod_cfg=style_mod_cfg,
            style_bias=style_bias)

        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        out = self.conv(x, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


class ConvDownLayer(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 downsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 bias=True,
                 act_cfg=dict(type='fused_bias')):

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
            layers.append(FusedBiasLeakyReLU(out_channels))

        super(ConvDownLayer, self).__init__(*layers)


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, blur_kernel=[1, 3, 3, 1]):
        super(ResBlock, self).__init__()

        self.conv1 = ConvDownLayer(
            in_channels, in_channels, 3, blur_kernel=blur_kernel)
        self.conv2 = ConvDownLayer(
            in_channels,
            out_channels,
            3,
            downsample=True,
            blur_kernel=blur_kernel)

        self.skip = ConvDownLayer(
            in_channels,
            out_channels,
            1,
            downsample=True,
            act_cfg=None,
            bias=False,
            blur_kernel=blur_kernel)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class ModMBStddevLayer(nn.Module):
    """Modified MiniBatch Stddev Layer.

    This layer is modified from ``MiniBatchStddevLayer`` used in PGGAN. In
    StyleGAN2, the authors add a new feature, `channel_groups`, into this
    layer.
    """

    def __init__(self,
                 group_size=4,
                 channel_groups=1,
                 sync_groups=None,
                 eps=1e-8):
        super(ModMBStddevLayer, self).__init__()
        self.group_size = group_size
        self.eps = eps
        self.channel_groups = channel_groups
        self.sync_groups = group_size if sync_groups is None else sync_groups

    def forward(self, x):

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

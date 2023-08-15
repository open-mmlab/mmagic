# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import ConvModule, build_norm_layer
from mmengine.model import BaseModule, normal_init
from torch.nn.init import _calculate_correct_fan

from mmagic.models.archs import AllGatherLayer
from mmagic.registry import MODELS


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
    if torch.__version__ >= '1.7.0':
        norm = torch.linalg.norm(x, ord=2, dim=1, keepdim=True)
    # support older pytorch version
    else:
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
    norm = norm / torch.sqrt(torch.tensor(x.shape[1]).to(x))

    return x / (norm + eps)


@MODELS.register_module()
class PixelNorm(BaseModule):
    """Pixel Normalization.

    This module is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Args:
        eps (float, optional): Epsilon value. Defaults to 1e-6.
    """

    _abbr_ = 'pn'

    def __init__(self, in_channels=None, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return pixel_norm(x, self.eps)


@MODELS.register_module()
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
        super().__init__(*args, **kwargs)
        self.with_equalized_lr = equalized_lr_cfg is not None
        if self.with_equalized_lr:
            self.conv = equalized_lr(self.conv, **equalized_lr_cfg)
            # initialize the conv weight with standard Gaussian noise.
            self._init_conv_weights()

    def _init_conv_weights(self):
        """Initialize conv weights as described in PGGAN."""
        normal_init(self.conv)


@MODELS.register_module()
class EqualizedLRConvUpModule(EqualizedLRConvModule):
    r"""Equalized LR (Upsample + Conv) Module.

    In this module, we inherit ``EqualizedLRConvModule`` and adopt
    upsampling before convolution. As for upsampling, in addition to the
    sampling layer in MMCV, we also offer the "fused_nn" type. "fused_nn"
    denotes fusing upsampling and convolution. The fusion is modified from
    the official Tensorflow implementation in:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L86

    Args:
        upsample (dict | None, optional): Config for upsampling operation. If
        ``None``, upsampling is ignored. If you need a faster fused version as
        the official PGGAN in Tensorflow, you should set it as
        ``dict(type='fused_nn')``. Defaults to
        ``dict(type='nearest', scale_factor=2)``.
    """

    def __init__(self,
                 *args,
                 upsample=dict(type='nearest', scale_factor=2),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.with_upsample = upsample is not None
        if self.with_upsample:
            if upsample.get('type') == 'fused_nn':
                assert isinstance(self.conv, nn.ConvTranspose2d)
                self.conv.register_forward_pre_hook(
                    EqualizedLRConvUpModule.fused_nn_hook)
            else:
                self.upsample_layer = MODELS.build(upsample)

    def forward(self, x, **kwargs):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if hasattr(self, 'upsample_layer'):
            x = self.upsample_layer(x)
        return super().forward(x, **kwargs)

    @staticmethod
    def fused_nn_hook(module, inputs):
        """Standard interface for forward pre hooks."""
        weight = module.weight
        # pad the last two dimensions
        weight = F.pad(weight, (1, 1, 1, 1))
        weight = weight[..., 1:, 1:] + weight[..., 1:, :-1] + weight[
            ..., :-1, 1:] + weight[..., :-1, :-1]
        module.weight = weight


@MODELS.register_module()
class EqualizedLRConvDownModule(EqualizedLRConvModule):
    r"""Equalized LR (Conv + Downsample)  Module.

    In this module, we inherit ``EqualizedLRConvModule`` and adopt
    downsampling after convolution. As for downsampling, we provide two modes
    of "avgpool" and "fused_pool". "avgpool" denotes the commonly used average
    pooling operation, while "fused_pool" represents fusing downsampling and
    convolution. The fusion is modified from the official Tensorflow
    implementation in:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L109


    Args:
        downsample (dict | None, optional): Config for downsampling operation.
            If ``None``, downsampling is ignored. Currently, we support the
            types of ["avgpool", "fused_pool"]. Defaults to
            dict(type='fused_pool').
    """

    def __init__(self, *args, downsample=dict(type='fused_pool'), **kwargs):
        super().__init__(*args, **kwargs)
        downsample_cfg = deepcopy(downsample)
        self.with_downsample = downsample is not None
        if self.with_downsample:
            type_ = downsample_cfg.pop('type')
            if type_ == 'avgpool':
                self.downsample = nn.AvgPool2d(2, 2)
            elif type_ == 'fused_pool':
                self.conv.register_forward_pre_hook(
                    EqualizedLRConvDownModule.fused_avgpool_hook)
            elif callable(downsample):
                self.downsample = downsample
            else:
                raise NotImplementedError(
                    'Currently, we only support ["avgpool", "fused_pool"] as '
                    f'the type of downsample, but got {type_} instead.')

    def forward(self, x, **kwargs):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            torch.Tensor: Normalized tensor.
        """
        x = super().forward(x, **kwargs)
        if hasattr(self, 'downsample'):
            x = self.downsample(x)
        return x

    @staticmethod
    def fused_avgpool_hook(module, inputs):
        """Standard interface for forward pre hooks."""
        weight = module.weight
        # pad the last two dimensions
        weight = F.pad(weight, (1, 1, 1, 1))
        weight = (weight[..., 1:, 1:] + weight[..., 1:, :-1] +
                  weight[..., :-1, 1:] + weight[..., :-1, :-1]) * 0.25
        module.weight = weight


@MODELS.register_module()
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
        super().__init__(*args, **kwargs)
        self.with_equalized_lr = equalized_lr_cfg is not None
        if self.with_equalized_lr:
            self.lr_mul = equalized_lr_cfg.get('lr_mul', 1.)
        else:
            # In fact, lr_mul will only be used in EqualizedLR for
            # initialization
            self.lr_mul = 1.
        if self.with_equalized_lr:
            equalized_lr(self, **equalized_lr_cfg)
            self._init_linear_weights()

    def _init_linear_weights(self):
        """Initialize linear weights as described in PGGAN."""
        nn.init.normal_(self.weight, 0, 1. / self.lr_mul)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)


@MODELS.register_module()
class PGGANNoiseTo2DFeat(BaseModule):

    def __init__(self,
                 noise_size,
                 out_channels,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                 norm_cfg=dict(type='PixelNorm'),
                 normalize_latent=True,
                 order=('linear', 'act', 'norm')):
        super().__init__()
        self.noise_size = noise_size
        self.out_channels = out_channels
        self.normalize_latent = normalize_latent
        self.with_activation = act_cfg is not None
        self.with_norm = norm_cfg is not None
        self.order = order
        assert len(order) == 3 and set(order) == set(['linear', 'act', 'norm'])

        # w/o bias, because the bias is added after reshaping the tensor to
        # 2D feature
        self.linear = EqualizedLRLinearModule(
            noise_size,
            out_channels * 16,
            equalized_lr_cfg=dict(gain=np.sqrt(2) / 4),
            bias=False)

        if self.with_activation:
            self.activation = MODELS.build(act_cfg)

        # add bias for reshaped 2D feature.
        self.register_parameter(
            'bias', nn.Parameter(torch.zeros(1, out_channels, 1, 1)))

        if self.with_norm:
            _, self.norm = build_norm_layer(norm_cfg, out_channels)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input noise tensor with shape (n, c).

        Returns:
            Tensor: Forward results with shape (n, c, 4, 4).
        """
        assert x.ndim == 2
        if self.normalize_latent:
            x = pixel_norm(x)
        for order in self.order:
            if order == 'linear':
                x = self.linear(x)
                # [n, c, 4, 4]
                x = torch.reshape(x, (-1, self.out_channels, 4, 4))
                x = x + self.bias
            elif order == 'act' and self.with_activation:
                x = self.activation(x)
            elif order == 'norm' and self.with_norm:
                x = self.norm(x)

        return x


class PGGANDecisionHead(BaseModule):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 bias=True,
                 equalized_lr_cfg=dict(gain=1),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                 out_act=None):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.with_activation = act_cfg is not None
        self.with_out_activation = out_act is not None

        # setup linear layers
        # dirty code for supporting default mode in PGGAN
        if equalized_lr_cfg:
            equalized_lr_cfg_ = dict(gain=2**0.5)
        else:
            equalized_lr_cfg_ = None
        self.linear0 = EqualizedLRLinearModule(
            self.in_channels,
            self.mid_channels,
            bias=bias,
            equalized_lr_cfg=equalized_lr_cfg_)
        self.linear1 = EqualizedLRLinearModule(
            self.mid_channels,
            self.out_channels,
            bias=bias,
            equalized_lr_cfg=equalized_lr_cfg)

        # setup activation layers
        if self.with_activation:
            self.activation = MODELS.build(act_cfg)

        if self.with_out_activation:
            self.out_activation = MODELS.build(out_act)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if x.ndim > 2:
            x = torch.reshape(x, (x.shape[0], -1))

        x = self.linear0(x)
        if self.with_activation:
            x = self.activation(x)

        x = self.linear1(x)
        if self.with_out_activation:
            x = self.out_activation(x)

        return x


@MODELS.register_module()
class MiniBatchStddevLayer(BaseModule):
    """Minibatch standard deviation.

    Args:
        group_size (int, optional): The size of groups in batch dimension.
            Defaults to 4.
        eps (float, optional):  Epsilon value to avoid computation error.
            Defaults to 1e-8.
        gather_all_batch (bool, optional): Whether gather batch from all GPUs.
            Defaults to False.
    """

    def __init__(self, group_size=4, eps=1e-8, gather_all_batch=False):
        super().__init__()
        self.group_size = group_size
        self.eps = eps
        self.gather_all_batch = gather_all_batch
        if self.gather_all_batch:
            assert torch.distributed.is_initialized(
            ), 'Only in distributed training can the tensors be all gathered.'

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.gather_all_batch:
            x = torch.cat(AllGatherLayer.apply(x), dim=0)

        # batch size should be smaller than or equal to group size. Otherwise,
        # batch size should be divisible by the group size.
        assert x.shape[
            0] <= self.group_size or x.shape[0] % self.group_size == 0, (
                'Batch size be smaller than or equal '
                'to group size. Otherwise,'
                ' batch size should be divisible by the group size.'
                f'But got batch size {x.shape[0]},'
                f' group size {self.group_size}')
        n, c, h, w = x.shape
        group_size = min(n, self.group_size)
        # [G, M, C, H, W]
        y = torch.reshape(x, (group_size, -1, c, h, w))
        # [G, M, C, H, W]
        y = y - y.mean(dim=0, keepdim=True)
        # In pt>=1.7, you can just use `.square()` function.
        # [M, C, H, W]
        y = y.pow(2).mean(dim=0, keepdim=False)
        y = torch.sqrt(y + self.eps)
        # [M, 1, 1, 1]
        y = y.mean(dim=(1, 2, 3), keepdim=True)
        y = y.repeat(group_size, 1, h, w)
        return torch.cat([x, y], dim=1)

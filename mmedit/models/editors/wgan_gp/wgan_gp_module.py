# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import constant_init

from mmedit.registry import MODELS, MODULES


@MODULES.register_module()
class WGANNoiseTo2DFeat(nn.Module):
    """Module used in WGAN-GP to transform 1D noise tensor in order [N, C] to
    2D shape feature tensor in order [N, C, H, W].

    Args:
        noise_size (int): Size of the input noise vector.
        out_channels (int): The channel number of the output feature.
        act_cfg (dict, optional): Config for the activation layer. Defaults to
            dict(type='ReLU').
        norm_cfg (dict, optional): Config dict to build norm layer. Defaults to
            dict(type='BN').
        order (tuple, optional): The order of conv/norm/activation layers. It
            is a sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm"). Defaults to
            ('linear', 'act', 'norm').
    """

    def __init__(self,
                 noise_size,
                 out_channels,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN'),
                 order=('linear', 'act', 'norm')):
        super().__init__()
        self.noise_size = noise_size
        self.out_channels = out_channels
        self.with_activation = act_cfg is not None
        self.with_norm = norm_cfg is not None
        self.order = order
        assert len(order) == 3 and set(order) == set(['linear', 'act', 'norm'])

        # w/o bias, because the bias is added after reshaping the tensor to
        # 2D feature
        self.linear = nn.Linear(noise_size, out_channels * 16, bias=False)

        if self.with_activation:
            self.activation = MODELS.build(act_cfg)

        # add bias for reshaped 2D feature.
        self.register_parameter(
            'bias', nn.Parameter(torch.zeros(1, out_channels, 1, 1)))

        if self.with_norm:
            _, self.norm = build_norm_layer(norm_cfg, out_channels)
        self._init_weight()

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input noise tensor with shape (n, c).

        Returns:
            Tensor: Forward results with shape (n, c, 4, 4).
        """
        assert x.ndim == 2
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

    def _init_weight(self):
        """Initialize weights for the model."""
        nn.init.normal_(self.linear.weight, 0., 1.)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)


class WGANDecisionHead(nn.Module):
    """Module used in WGAN-GP to get the final prediction result with 4x4
    resolution input tensor in the bottom of the discriminator.

    Args:
        in_channels (int): Number of channels in input feature map.
        mid_channels (int): Number of channels in feature map after
            convolution.
        out_channels (int): The channel number of the final output layer.
        bias (bool, optional): Whether to use bias parameter. Defaults to True.
        act_cfg (dict, optional): Config for the activation layer. Defaults to
            dict(type='ReLU').
        out_act (dict, optional): Config for the activation layer of output
            layer. Defaults to None.
        norm_cfg (dict, optional): Config dict to build norm layer. Defaults to
            dict(type='LN2d').
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 bias=True,
                 act_cfg=dict(type='ReLU'),
                 out_act=None,
                 norm_cfg=dict(type='LN2d')):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.with_out_activation = out_act is not None

        # setup conv layer
        self.conv = ConvLNModule(
            in_channels,
            feature_shape=(mid_channels, 1, 1),
            kernel_size=4,
            out_channels=mid_channels,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            order=('conv', 'norm', 'act'))
        # setup linear layer
        self.linear = nn.Linear(
            self.mid_channels, self.out_channels, bias=bias)

        if self.with_out_activation:
            self.out_activation = MODELS.build(out_act)

        self._init_weight()

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.conv(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.linear(x)
        if self.with_out_activation:
            x = self.out_activation(x)
        return x

    def _init_weight(self):
        """Initialize weights for the model."""
        nn.init.normal_(self.linear.weight, 0., 1.)
        nn.init.constant_(self.linear.bias, 0.)


@MODELS.register_module()
class ConvLNModule(ConvModule):
    r"""ConvModule with Layer Normalization.

    In this module, we inherit default ``mmcv.cnn.ConvModule`` and deal with
    the situation that 'norm_cfg' is 'LN2d' or 'GN'. We adopt 'GN' as a
    replacement for layer normalization referring to:
    https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch/blob/master/module.py # noqa

    Args:
        feature_shape (tuple): The shape of feature map that will be.
    """

    def __init__(self, *args, feature_shape=None, **kwargs):
        if 'norm_cfg' in kwargs and kwargs['norm_cfg'] is not None and kwargs[
                'norm_cfg']['type'] in ['LN2d', 'GN']:
            nkwargs = deepcopy(kwargs)
            nkwargs['norm_cfg'] = None
            super().__init__(*args, **nkwargs)
            self.with_norm = True
            self.norm_name = kwargs['norm_cfg']['type']
            if self.norm_name == 'LN2d':
                norm = nn.LayerNorm(feature_shape)
                self.add_module(self.norm_name, norm)
            else:
                norm = nn.GroupNorm(1, feature_shape[0])
                self.add_module(self.norm_name, norm)
        else:
            super().__init__(*args, **kwargs)

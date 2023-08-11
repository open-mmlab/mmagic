# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.dist import is_distributed
from mmengine.model import BaseModule, constant_init, xavier_init
from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn.utils import spectral_norm

from mmagic.models.editors.biggan.biggan_modules import SNConvModule
from mmagic.models.editors.biggan.biggan_snmodule import SNEmbedding
from mmagic.registry import MODELS


@MODELS.register_module()
class SNGANGenResBlock(BaseModule):
    """ResBlock used in Generator of SNGAN / Proj-GAN.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        hidden_channels (int, optional): Input channels of the second Conv
            layer of the block. If ``None`` is given, would be set as
            ``out_channels``. Default to None.
        num_classes (int, optional): Number of classes would like to generate.
            This argument would pass to norm layers and influence the structure
            and behavior of the normalization process. Default to 0.
        use_cbn (bool, optional): Whether use conditional normalization. This
            argument would pass to norm layers. Default to True.
        use_norm_affine (bool, optional): Whether use learnable affine
            parameters in norm operation when cbn is off. Default False.
        act_cfg (dict, optional): Config for activate function. Default
            to ``dict(type='ReLU')``.
        upsample_cfg (dict, optional): Config for the upsample method.
            Default to ``dict(type='nearest', scale_factor=2)``.
        upsample (bool, optional): Whether apply upsample operation in this
            module. Default to True.
        auto_sync_bn (bool, optional): Whether convert Batch Norm to
            Synchronized ones when Distributed training is on. Default to True.
        conv_cfg (dict | None): Config for conv blocks of this module. If pass
            ``None``, would use ``_default_conv_cfg``. Default to ``None``.
        with_spectral_norm (bool, optional): Whether use spectral norm for
            conv blocks and norm layers. Default to True.
        with_embedding_spectral_norm (bool, optional): Whether use spectral
            norm for embedding layers in normalization blocks or not. If not
            specified (set as ``None``), ``with_embedding_spectral_norm`` would
            be set as the same value as ``with_spectral_norm``.
            Default to None.
        sn_style (str, optional): The style of spectral normalization.
            If set to `ajbrock`, implementation by
            ajbrock(https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py)
            will be adopted.
            If set to `torch`, implementation by `PyTorch` will be adopted.
            Defaults to `torch`.
        norm_eps (float, optional): eps for Normalization layers (both
            conditional and non-conditional ones). Default to `1e-4`.
        sn_eps (float, optional): eps for spectral normalization operation.
            Default to `1e-12`.
        init_cfg (dict, optional): Config for weight initialization.
            Default to ``dict(type='BigGAN')``.
    """

    _default_conv_cfg = dict(kernel_size=3, stride=1, padding=1, act_cfg=None)

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 num_classes=0,
                 use_cbn=True,
                 use_norm_affine=False,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN'),
                 upsample_cfg=dict(type='nearest', scale_factor=2),
                 upsample=True,
                 auto_sync_bn=True,
                 conv_cfg=None,
                 with_spectral_norm=False,
                 with_embedding_spectral_norm=None,
                 sn_style='torch',
                 norm_eps=1e-4,
                 sn_eps=1e-12,
                 init_cfg=dict(type='BigGAN')):

        super().__init__()
        self.learnable_sc = in_channels != out_channels or upsample
        self.with_upsample = upsample
        self.init_type = init_cfg.get('type', None)

        self.activate = MODELS.build(act_cfg)
        hidden_channels = out_channels if hidden_channels is None \
            else hidden_channels

        if self.with_upsample:
            self.upsample = MODELS.build(upsample_cfg)

        self.conv_cfg = deepcopy(self._default_conv_cfg)
        if conv_cfg is not None:
            self.conv_cfg.update(conv_cfg)

        # set `norm_spectral_norm` as `with_spectral_norm` if not defined
        with_embedding_spectral_norm = with_embedding_spectral_norm \
            if with_embedding_spectral_norm is not None else with_spectral_norm

        sn_cfg = dict(eps=sn_eps, sn_style=sn_style)
        self.conv_1 = SNConvModule(
            in_channels,
            hidden_channels,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=sn_cfg,
            **self.conv_cfg)
        self.conv_2 = SNConvModule(
            hidden_channels,
            out_channels,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=sn_cfg,
            **self.conv_cfg)

        self.norm_1 = SNConditionNorm(in_channels, num_classes, use_cbn,
                                      norm_cfg, use_norm_affine, auto_sync_bn,
                                      with_embedding_spectral_norm, sn_style,
                                      norm_eps, sn_eps, init_cfg)
        self.norm_2 = SNConditionNorm(hidden_channels, num_classes, use_cbn,
                                      norm_cfg, use_norm_affine, auto_sync_bn,
                                      with_embedding_spectral_norm, sn_style,
                                      norm_eps, sn_eps, init_cfg)

        if self.learnable_sc:
            # use hyperparameters-fixed shortcut here
            self.shortcut = SNConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                act_cfg=None,
                with_spectral_norm=with_spectral_norm,
                spectral_norm_cfg=sn_cfg)
        self.init_weights()

    def forward(self, x, y=None):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            y (Tensor): Input label with shape (n, ).
                Default None.

        Returns:
            Tensor: Forward results.
        """

        out = self.norm_1(x, y)
        out = self.activate(out)
        if self.with_upsample:
            out = self.upsample(out)
        out = self.conv_1(out)

        out = self.norm_2(out, y)
        out = self.activate(out)
        out = self.conv_2(out)

        shortcut = self.forward_shortcut(x)
        return out + shortcut

    def forward_shortcut(self, x):
        """Forward the shortcut branch.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = x
        if self.learnable_sc:
            if self.with_upsample:
                out = self.upsample(out)
            out = self.shortcut(out)
        return out

    def init_weights(self):
        """Initialize weights for the model."""
        if self.init_type.upper() == 'STUDIO':
            nn.init.orthogonal_(self.conv_1.conv.weight)
            nn.init.orthogonal_(self.conv_2.conv.weight)
            self.conv_1.conv.bias.data.fill_(0.)
            self.conv_2.conv.bias.data.fill_(0.)
            if self.learnable_sc:
                nn.init.orthogonal_(self.shortcut.conv.weight)
                self.shortcut.conv.bias.data.fill_(0.)
        elif self.init_type.upper() == 'BIGGAN':
            xavier_uniform_(self.conv_1.conv.weight, gain=1)
            xavier_uniform_(self.conv_2.conv.weight, gain=1)
            if self.learnable_sc:
                xavier_uniform_(self.shortcut.conv.weight, gain=1)
        elif self.init_type.upper() == 'SAGAN':
            xavier_init(self.conv_1, gain=1, distribution='uniform')
            xavier_init(self.conv_2, gain=1, distribution='uniform')
            if self.learnable_sc:
                xavier_init(self.shortcut, gain=1, distribution='uniform')
        elif self.init_type.upper() in ['SNGAN', 'SNGAN-PROJ', 'GAN-PROJ']:
            xavier_init(self.conv_1, gain=np.sqrt(2), distribution='uniform')
            xavier_init(self.conv_2, gain=np.sqrt(2), distribution='uniform')
            if self.learnable_sc:
                xavier_init(self.shortcut, gain=1, distribution='uniform')
        else:
            raise NotImplementedError('Unknown initialization method: '
                                      f'\'{self.init_type}\'')


@MODELS.register_module()
class SNGANDiscResBlock(BaseModule):
    """resblock used in discriminator of sngan / proj-gan.

    args:
        in_channels (int): input channels.
        out_channels (int): output channels.
        hidden_channels (int, optional): input channels of the second conv
            layer of the block. if ``none`` is given, would be set as
            ``out_channels``. Defaults to none.
        downsample (bool, optional): whether apply downsample operation in this
            module.  Defaults to false.
        act_cfg (dict, optional): config for activate function. default
            to ``dict(type='relu')``.
        conv_cfg (dict | none): config for conv blocks of this module. if pass
            ``none``, would use ``_default_conv_cfg``. default to ``none``.
        with_spectral_norm (bool, optional): whether use spectral norm for
            conv blocks and norm layers. Defaults to true.
        sn_eps (float, optional): eps for spectral normalization operation.
            Default to `1e-12`.
        sn_style (str, optional): The style of spectral normalization.
            If set to `ajbrock`, implementation by
            ajbrock(https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py)
            will be adopted.
            If set to `torch`, implementation by `PyTorch` will be adopted.
            Defaults to `torch`.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to ``dict(type='BigGAN')``.
    """

    _default_conv_cfg = dict(kernel_size=3, stride=1, padding=1, act_cfg=None)

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 downsample=False,
                 act_cfg=dict(type='ReLU'),
                 conv_cfg=None,
                 with_spectral_norm=True,
                 sn_style='torch',
                 sn_eps=1e-12,
                 init_cfg=dict(type='BigGAN')):

        super().__init__()
        hidden_channels = out_channels if hidden_channels is None \
            else hidden_channels
        self.with_downsample = downsample
        self.init_type = init_cfg.get('type', None)

        self.conv_cfg = deepcopy(self._default_conv_cfg)
        if conv_cfg is not None:
            self.conv_cfg.update(conv_cfg)

        self.activate = MODELS.build(act_cfg)

        sn_cfg = dict(eps=sn_eps, sn_style=sn_style)
        self.conv_1 = SNConvModule(
            in_channels,
            hidden_channels,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=sn_cfg,
            **self.conv_cfg)
        self.conv_2 = SNConvModule(
            hidden_channels,
            out_channels,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=sn_cfg,
            **self.conv_cfg)

        if self.with_downsample:
            self.downsample = nn.AvgPool2d(2, 2)

        self.learnable_sc = in_channels != out_channels or downsample
        if self.learnable_sc:
            # use hyperparameters-fixed shortcut here
            self.shortcut = SNConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                act_cfg=None,
                with_spectral_norm=with_spectral_norm,
                spectral_norm_cfg=sn_cfg)
        self.init_weights()

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        out = self.activate(x)
        out = self.conv_1(out)
        out = self.activate(out)
        out = self.conv_2(out)
        if self.with_downsample:
            out = self.downsample(out)

        shortcut = self.forward_shortcut(x)
        return out + shortcut

    def forward_shortcut(self, x: Tensor) -> Tensor:
        """Forward the shortcut branch.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = x
        if self.learnable_sc:
            out = self.shortcut(out)
            if self.with_downsample:
                out = self.downsample(out)
        return out

    def init_weights(self):
        """Initialize weights."""
        if self.init_type.upper() == 'STUDIO':
            nn.init.orthogonal_(self.conv_1.conv.weight)
            nn.init.orthogonal_(self.conv_2.conv.weight)
            self.conv_1.conv.bias.data.fill_(0.)
            self.conv_2.conv.bias.data.fill_(0.)
            if self.learnable_sc:
                nn.init.orthogonal_(self.shortcut.conv.weight)
                self.shortcut.conv.bias.data.fill_(0.)
        elif self.init_type.upper() == 'BIGGAN':
            xavier_uniform_(self.conv_1.conv.weight, gain=1)
            xavier_uniform_(self.conv_2.conv.weight, gain=1)
            if self.learnable_sc:
                xavier_uniform_(self.shortcut.conv.weight, gain=1)
        elif self.init_type.upper() == 'SAGAN':
            xavier_init(self.conv_1, gain=1, distribution='uniform')
            xavier_init(self.conv_2, gain=1, distribution='uniform')
            if self.learnable_sc:
                xavier_init(self.shortcut, gain=1, distribution='uniform')
        elif self.init_type.upper() in ['SNGAN', 'SNGAN-PROJ', 'GAN-PROJ']:
            xavier_init(self.conv_1, gain=np.sqrt(2), distribution='uniform')
            xavier_init(self.conv_2, gain=np.sqrt(2), distribution='uniform')
            if self.learnable_sc:
                xavier_init(self.shortcut, gain=1, distribution='uniform')
        else:
            raise NotImplementedError('Unknown initialization method: '
                                      f'\'{self.init_type}\'')


@MODELS.register_module()
class SNGANDiscHeadResBlock(BaseModule):
    """The first ResBlock used in discriminator of sngan / proj-gan. Compared
    to ``SNGANDisResBlock``, this module has a different forward order.

    args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        downsample (bool, optional): whether apply downsample operation in this
            module.  default to false.
        conv_cfg (dict | none): config for conv blocks of this module. if pass
            ``none``, would use ``_default_conv_cfg``. default to ``none``.
        act_cfg (dict, optional): config for activate function. default
            to ``dict(type='relu')``.
        with_spectral_norm (bool, optional): whether use spectral norm for
            conv blocks and norm layers. default to true.
        sn_style (str, optional): The style of spectral normalization.
            If set to `ajbrock`, implementation by
            ajbrock(https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py)
            will be adopted.
            If set to `torch`, implementation by `PyTorch` will be adopted.
            Defaults to `torch`.
        sn_eps (float, optional): eps for spectral normalization operation.
            Default to `1e-12`.
        init_cfg (dict, optional): Config for weight initialization.
            Default to ``dict(type='BigGAN')``.
    """

    _default_conv_cfg = dict(kernel_size=3, stride=1, padding=1, act_cfg=None)

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 with_spectral_norm=True,
                 sn_eps=1e-12,
                 sn_style='torch',
                 init_cfg=dict(type='BigGAN')):

        super().__init__()

        self.init_type = init_cfg.get('type', None)
        self.conv_cfg = deepcopy(self._default_conv_cfg)
        if conv_cfg is not None:
            self.conv_cfg.update(conv_cfg)

        self.activate = MODELS.build(act_cfg)

        sn_cfg = dict(eps=sn_eps, sn_style=sn_style)
        self.conv_1 = SNConvModule(
            in_channels,
            out_channels,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=sn_cfg,
            **self.conv_cfg)
        self.conv_2 = SNConvModule(
            out_channels,
            out_channels,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=sn_cfg,
            **self.conv_cfg)

        self.downsample = nn.AvgPool2d(2, 2)

        # use hyperparameters-fixed shortcut here
        self.shortcut = SNConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=sn_cfg)
        self.init_weights()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = self.conv_1(x)
        out = self.activate(out)
        out = self.conv_2(out)
        out = self.downsample(out)

        shortcut = self.forward_shortcut(x)
        return out + shortcut

    def forward_shortcut(self, x: Tensor) -> Tensor:
        """Forward the shortcut branch.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = self.downsample(x)
        out = self.shortcut(out)
        return out

    def init_weights(self):
        """Initialize weights."""
        if self.init_type.upper() == 'STUDIO':
            for m in [self.conv_1, self.conv_2, self.shortcut]:
                nn.init.orthogonal_(m.conv.weight)
                m.conv.bias.data.fill_(0.)
        elif self.init_type.upper() == 'BIGGAN':
            xavier_uniform_(self.conv_1.conv.weight, gain=1)
            xavier_uniform_(self.conv_2.conv.weight, gain=1)
            xavier_uniform_(self.shortcut.conv.weight, gain=1)
        elif self.init_type.upper() == 'SAGAN':
            xavier_init(self.conv_1, gain=1, distribution='uniform')
            xavier_init(self.conv_2, gain=1, distribution='uniform')
            xavier_init(self.shortcut, gain=1, distribution='uniform')
        elif self.init_type.upper() in ['SNGAN', 'SNGAN-PROJ', 'GAN-PROJ']:
            xavier_init(self.conv_1, gain=np.sqrt(2), distribution='uniform')
            xavier_init(self.conv_2, gain=np.sqrt(2), distribution='uniform')
            xavier_init(self.shortcut, gain=1, distribution='uniform')
        else:
            raise NotImplementedError('Unknown initialization method: '
                                      f'\'{self.init_type}\'')


@MODELS.register_module()
class SNConditionNorm(BaseModule):
    """Conditional Normalization for SNGAN / Proj-GAN. The implementation
    refers to.

    https://github.com/pfnet-research/sngan_projection/blob/master/source/links/conditional_batch_normalization.py  # noda

    and

    https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/utils/model_ops.py  # noqa

    Args:
        in_channels (int): Number of the channels of the input feature map.
        num_classes (int): Number of the classes in the dataset. If ``use_cbn``
            is True, ``num_classes`` must larger than 0.
        use_cbn (bool, optional): Whether use conditional normalization. If
            ``use_cbn`` is True, two embedding layers would be used to mapping
            label to weight and bias used in normalization process.
        norm_cfg (dict, optional): Config for normalization method. Defaults
            to ``dict(type='BN')``.
        cbn_norm_affine (bool):  Whether set ``affine=True`` when use conditional batch norm.
            This argument only work when ``use_cbn`` is True. Defaults to False.
        auto_sync_bn (bool, optional): Whether convert Batch Norm to
            Synchronized ones when Distributed training is on. Defaults to True.
        with_spectral_norm (bool, optional): whether use spectral norm for
            conv blocks and norm layers. Defaults to true.
        norm_eps (float, optional): eps for Normalization layers (both
            conditional and non-conditional ones). Defaults to `1e-4`.
        sn_style (str, optional): The style of spectral normalization.
            If set to `ajbrock`, implementation by
            ajbrock(https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py)
            will be adopted.
            If set to `torch`, implementation by `PyTorch` will be adopted.
            Defaults to `torch`.
        sn_eps (float, optional): eps for spectral normalization operation.
            Defaults to `1e-12`.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to ``dict(type='BigGAN')``.
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 use_cbn=True,
                 norm_cfg=dict(type='BN'),
                 cbn_norm_affine=False,
                 auto_sync_bn=True,
                 with_spectral_norm=False,
                 sn_style='torch',
                 norm_eps=1e-4,
                 sn_eps=1e-12,
                 init_cfg=dict(type='BigGAN')):
        super().__init__()
        self.use_cbn = use_cbn
        self.init_type = init_cfg.get('type', None)

        norm_cfg = deepcopy(norm_cfg)
        norm_type = norm_cfg['type']

        if norm_type not in ['IN', 'BN', 'SyncBN']:
            raise ValueError('Only support `IN` (InstanceNorm), '
                             '`BN` (BatcnNorm) and `SyncBN` for '
                             'Class-conditional bn. '
                             f'Receive norm_type: {norm_type}')

        if self.use_cbn:
            norm_cfg.setdefault('affine', cbn_norm_affine)
        norm_cfg.setdefault('eps', norm_eps)
        if is_distributed() and auto_sync_bn and norm_type == 'BN':
            norm_cfg['type'] = 'SyncBN'

        _, self.norm = build_norm_layer(norm_cfg, in_channels)

        if self.use_cbn:
            if num_classes <= 0:
                raise ValueError('`num_classes` must be larger '
                                 'than 0 with `use_cbn=True`')
            self.reweight_embedding = (
                self.init_type.upper() == 'BIGGAN'
                or self.init_type.upper() == 'STUDIO')
            if with_spectral_norm:
                if sn_style == 'torch':
                    self.weight_embedding = spectral_norm(
                        nn.Embedding(num_classes, in_channels), eps=sn_eps)
                    self.bias_embedding = spectral_norm(
                        nn.Embedding(num_classes, in_channels), eps=sn_eps)
                elif sn_style == 'ajbrock':
                    self.weight_embedding = SNEmbedding(
                        num_classes, in_channels, eps=sn_eps)
                    self.bias_embedding = SNEmbedding(
                        num_classes, in_channels, eps=sn_eps)
                else:
                    raise NotImplementedError(
                        f'{sn_style} style spectral Norm is not '
                        'supported yet')
            else:
                self.weight_embedding = nn.Embedding(num_classes, in_channels)
                self.bias_embedding = nn.Embedding(num_classes, in_channels)

        self.init_weights()

    def forward(self, x, y=None):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            y (Tensor, optional): Input label with shape (n, ).
                Default None.

        Returns:
            Tensor: Forward results.
        """

        out = self.norm(x)
        if self.use_cbn:
            weight = self.weight_embedding(y)[:, :, None, None]
            bias = self.bias_embedding(y)[:, :, None, None]
            if self.reweight_embedding:
                # print('reweight_called --> correct')
                weight = weight + 1.
            out = out * weight + bias
        return out

    def init_weights(self):
        """Initialize weights."""
        if self.use_cbn:
            if self.init_type.upper() == 'STUDIO':
                nn.init.orthogonal_(self.weight_embedding.weight)
                nn.init.orthogonal_(self.bias_embedding.weight)
            elif self.init_type.upper() == 'BIGGAN':
                xavier_uniform_(self.weight_embedding.weight, gain=1)
                xavier_uniform_(self.bias_embedding.weight, gain=1)
            elif self.init_type.upper() in [
                    'SNGAN', 'SNGAN-PROJ', 'GAN-PROJ', 'SAGAN'
            ]:
                constant_init(self.weight_embedding, 1)
                constant_init(self.bias_embedding, 0)
            else:
                raise NotImplementedError('Unknown initialization method: '
                                          f'\'{self.init_type}\'')

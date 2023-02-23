# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch
import torch.nn as nn
from mmengine.model import (BaseModule, normal_init, update_init_info,
                            xavier_init)
from torch.nn.utils import spectral_norm

from mmagic.registry import MODELS
from .biggan_modules import SelfAttentionBlock
from .biggan_snmodule import SNEmbedding, SNLinear


@MODELS.register_module()
class BigGANDiscriminator(BaseModule):
    """BigGAN Discriminator. The implementation refers to
    https://github.com/ajbrock/BigGAN-PyTorch/blob/master/BigGAN.py # noqa.

    In BigGAN, we use a SAGAN-based architecture composing of an self-attention
    block and number of convolutional residual blocks with spectral
    normalization.

    More details can be found in: Large Scale GAN Training for High Fidelity
    Natural Image Synthesis (ICLR2019).

    The design of the model structure is highly corresponding to the output
    resolution. For the original BigGAN's generator, you can set ``output_scale``
    as you need and use the default value of ``arch_cfg`` and ``blocks_cfg``.
    If you want to customize the model, you can set the arguments in this way:

    ``arch_cfg``: Config for the architecture of this generator. You can refer
    the ``_default_arch_cfgs`` in the ``_get_default_arch_cfg`` function to see
    the format of the ``arch_cfg``. Basically, you need to provide information
    of each block such as the numbers of input and output channels, whether to
    perform upsampling, etc.

    ``blocks_cfg``: Config for the convolution block. You can replace the block
    type to your registered customized block and adjust block params here.
    However, you should notice that some params are shared among these blocks
    like ``act_cfg``, ``with_spectral_norm``, ``sn_eps``, etc.

    Args:
        input_scale (int): The scale of the input image.
        num_classes (int, optional): The number of conditional classes.
            Defaults to 0.
        in_channels (int, optional): The channel number of the input image.
            Defaults to 3.
        out_channels (int, optional): The channel number of the final output.
            Defaults to 1.
        base_channels (int, optional): The basic channel number of the
            discriminator. The other layers contains channels based on this
            number. Defaults to 96.
        sn_eps (float, optional): Epsilon value for spectral normalization.
            Defaults to 1e-6.
        sn_style (str, optional): The style of spectral normalization.
            If set to `ajbrock`, implementation by
            ajbrock(https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py)
            will be adopted.
            If set to `torch`, implementation by `PyTorch` will be adopted.
            Defaults to `ajbrock`.
        act_cfg (dict, optional): Config for the activation layer.
            Defaults to dict(type='ReLU').
        with_spectral_norm (bool, optional): Whether to use spectral
            normalization. Defaults to True.
        blocks_cfg (dict, optional): Config for the convolution block.
            Defaults to dict(type='BigGANDiscResBlock').
        arch_cfg (dict, optional): Config for the architecture of this
            discriminator. Defaults to None.
        init_cfg (dict, optional): Initialization config dict. If type is
            `Pretrained`, the pretrain model will be loaded. Otherwise, type
            will be parsed as the name of initialization method. Support values
            are 'ortho', 'N02', 'xavier'. Defaults to dict(type='ortho').
    """

    def __init__(self,
                 input_scale,
                 num_classes=0,
                 in_channels=3,
                 out_channels=1,
                 base_channels=96,
                 sn_eps=1e-6,
                 sn_style='ajbrock',
                 act_cfg=dict(type='ReLU'),
                 with_spectral_norm=True,
                 blocks_cfg=dict(type='BigGANDiscResBlock'),
                 arch_cfg=None,
                 init_cfg=dict(type='ortho')):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.input_scale = input_scale
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.arch = arch_cfg if arch_cfg else self._get_default_arch_cfg(
            self.input_scale, self.in_channels, self.base_channels)
        self.blocks_cfg = deepcopy(blocks_cfg)
        self.blocks_cfg.update(
            dict(
                act_cfg=act_cfg,
                sn_eps=sn_eps,
                sn_style=sn_style,
                with_spectral_norm=with_spectral_norm))
        self.sn_style = sn_style

        self.conv_blocks = nn.ModuleList()
        for index, out_ch in enumerate(self.arch['out_channels']):
            # change args to adapt to current block
            self.blocks_cfg.update(
                dict(
                    in_channels=self.arch['in_channels'][index],
                    out_channels=out_ch,
                    with_downsample=self.arch['downsample'][index],
                    is_head_block=(index == 0)))
            self.conv_blocks.append(MODELS.build(self.blocks_cfg))
            if self.arch['attention'][index]:
                self.conv_blocks.append(
                    SelfAttentionBlock(
                        out_ch,
                        with_spectral_norm=with_spectral_norm,
                        sn_eps=sn_eps,
                        sn_style=sn_style))

        self.activate = MODELS.build(act_cfg)

        self.decision = nn.Linear(self.arch['out_channels'][-1], out_channels)
        if with_spectral_norm:
            if sn_style == 'torch':
                self.decision = spectral_norm(self.decision, eps=sn_eps)
            elif sn_style == 'ajbrock':
                self.decision = SNLinear(
                    self.arch['out_channels'][-1], out_channels, eps=sn_eps)
            else:
                raise NotImplementedError('sn style')

        if self.num_classes > 0:
            self.proj_y = nn.Embedding(self.num_classes,
                                       self.arch['out_channels'][-1])
            if with_spectral_norm:
                if sn_style == 'torch':
                    self.proj_y = spectral_norm(self.proj_y, eps=sn_eps)
                elif sn_style == 'ajbrock':
                    self.proj_y = SNEmbedding(
                        self.num_classes,
                        self.arch['out_channels'][-1],
                        eps=sn_eps)
                else:
                    raise NotImplementedError('sn style')

    def _get_default_arch_cfg(self, input_scale, in_channels, base_channels):
        assert input_scale in [32, 64, 128, 256, 512]
        _default_arch_cfgs = {
            '32': {
                'in_channels':
                [in_channels] + [base_channels * item for item in [4, 4, 4]],
                'out_channels':
                [base_channels * item for item in [4, 4, 4, 4]],
                'downsample': [True, True, False, False],
                'resolution': [16, 8, 8, 8],
                'attention': [False, False, False, False]
            },
            '64': {
                'in_channels': [in_channels] +
                [base_channels * item for item in [1, 2, 4, 8]],
                'out_channels':
                [base_channels * item for item in [1, 2, 4, 8, 16]],
                'downsample': [True] * 4 + [False],
                'resolution': [32, 16, 8, 4, 4],
                'attention': [False, False, False, False, False]
            },
            '128': {
                'in_channels': [in_channels] +
                [base_channels * item for item in [1, 2, 4, 8, 16]],
                'out_channels':
                [base_channels * item for item in [1, 2, 4, 8, 16, 16]],
                'downsample': [True] * 5 + [False],
                'resolution': [64, 32, 16, 8, 4, 4],
                'attention': [True, False, False, False, False, False]
            },
            '256': {
                'in_channels': [in_channels] +
                [base_channels * item for item in [1, 2, 4, 8, 8, 16]],
                'out_channels':
                [base_channels * item for item in [1, 2, 4, 8, 8, 16, 16]],
                'downsample': [True] * 6 + [False],
                'resolution': [128, 64, 32, 16, 8, 4, 4],
                'attention': [False, True, False, False, False, False]
            },
            '512': {
                'in_channels': [in_channels] +
                [base_channels * item for item in [1, 1, 2, 4, 8, 8, 16]],
                'out_channels':
                [base_channels * item for item in [1, 1, 2, 4, 8, 8, 16, 16]],
                'downsample': [True] * 7 + [False],
                'resolution': [256, 128, 64, 32, 16, 8, 4, 4],
                'attention': [False, False, False, True, False, False, False]
            }
        }

        return _default_arch_cfgs[str(input_scale)]

    def forward(self, x, label=None):
        """Forward function.

        Args:
            x (torch.Tensor): Fake or real image tensor.
            label (torch.Tensor | None): Label Tensor. Defaults to None.

        Returns:
            torch.Tensor: Prediction for the reality of the input image with
                given label.
        """
        x0 = x
        for conv_block in self.conv_blocks:
            x0 = conv_block(x0)
        x0 = self.activate(x0)
        x0 = torch.sum(x0, dim=[2, 3])
        out = self.decision(x0)

        if self.num_classes > 0:
            w_y = self.proj_y(label)
            out = out + torch.sum(w_y * x0, dim=1, keepdim=True)
        return out

    def init_weights(self):
        """Init weights for models."""
        if self.init_cfg is not None and self.init_cfg['type'] == 'Pretrained':
            super().init_weights()
            return
        for m in self.modules():
            init_type = self.init_cfg['type']
            module_name = m.__class__.__name__
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                if init_type == 'ortho':
                    nn.init.orthogonal_(m.weight)
                elif init_type == 'N02':
                    normal_init(m, 0.0, 0.02)
                elif init_type == 'xavier':
                    xavier_init(m)
                else:
                    raise NotImplementedError(
                        f'{init_type} initialization not supported now.')
                # save init info
                init_info = (f'{module_name} belongs to (nn.Conv2d, '
                             'nn.Linear, nn.Embedding), initialize by '
                             f'\'init_type\' {init_type}')
                if hasattr(m, '_params_init_info'):
                    update_init_info(m, init_info)
        self._is_init = True

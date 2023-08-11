# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, xavier_init
from mmengine.runner import load_checkpoint
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix
from torch.nn.init import xavier_uniform_
from torch.nn.utils import spectral_norm

from mmagic.registry import MODELS


@MODELS.register_module('SAGANDiscriminator')
@MODELS.register_module()
class ProjDiscriminator(BaseModule):
    r"""Discriminator for SNGAN / Proj-GAN. The implementation is refer to
    https://github.com/pfnet-research/sngan_projection/tree/master/dis_models

    The overall structure of the projection discriminator can be split into a
    ``from_rgb`` layer, a group of ResBlocks, a linear decision layer, and a
    projection layer. To support defining custom layers, we introduce
    ``from_rgb_cfg`` and ``blocks_cfg``.

    The design of the model structure is highly corresponding to the output
    resolution. Therefore, we provide `channels_cfg` and `downsample_cfg` to
    control the input channels and the downsample behavior of the intermediate
    blocks.

    ``downsample_cfg``: In default config of SNGAN / Proj-GAN, whether to apply
        downsample in each intermediate blocks is quite flexible and
        corresponding to the resolution of the output image. Therefore, we
        support user to define the ``downsample_cfg`` by themselves, and to
        control the structure of the discriminator.

    ``channels_cfg``: In default config of SNGAN / Proj-GAN, the number of
        ResBlocks and the channels of those blocks are corresponding to the
        resolution of the output image. Therefore, we allow user to define
        `channels_cfg` for try their own models.  We also provide a default
        config to allow users to build the model only from the output
        resolution.

    Args:
        input_scale (int): The scale of the input image.
        num_classes (int, optional): The number classes you would like to
            generate. If num_classes=0, no label projection would be used.
            Default to 0.
        base_channels (int, optional): The basic channel number of the
            discriminator. The other layers contains channels based on this
            number.  Defaults to 128.
        input_channels (int, optional): Channels of the input image.
            Defaults to 3.
        attention_cfg (dict, optional): Config for the self-attention block.
            Default to ``dict(type='SelfAttentionBlock')``.
        attention_after_nth_block (int | list[int], optional): Self-attention
            block would be added after which *ConvBlock* (including the head
            block). If ``int`` is passed, only one attention block would be
            added. If ``list`` is passed, self-attention blocks would be added
            after multiple ConvBlocks. To be noted that if the input is
            smaller than ``1``, self-attention corresponding to this index
            would be ignored. Default to 0.
        channels_cfg (list | dict[list], optional): Config for input channels
            of the intermediate blocks. If list is passed, each element of the
            list means the input channels of current block is how many times
            compared to the ``base_channels``. For block ``i``, the input and
            output channels should be ``channels_cfg[i]`` and
            ``channels_cfg[i+1]`` If dict is provided, the key of the dict
            should be the output scale and corresponding value should be a list
            to define channels.  Default: Please refer to
            ``_defualt_channels_cfg``.
        downsample_cfg (list[bool] | dict[list], optional): Config for
            downsample behavior of the intermediate layers. If a list is
            passed, ``downsample_cfg[idx] == True`` means apply downsample in
            idx-th block, and vice versa. If dict is provided, the key dict
            should be the input scale of the image and corresponding value
            should be a list ti define the downsample behavior. Default: Please
            refer to ``_default_downsample_cfg``.
        from_rgb_cfg (dict, optional): Config for the first layer to convert
            rgb image to feature map. Defaults to
            ``dict(type='SNGANDiscHeadResBlock')``.
        blocks_cfg (dict, optional): Config for the intermediate blocks.
            Defaults to ``dict(type='SNGANDiscResBlock')``
        act_cfg (dict, optional): Activation config for the final output
            layer. Defaults to ``dict(type='ReLU')``.
        with_spectral_norm (bool, optional): Whether use spectral norm for
            all conv blocks or not. Default to True.
        sn_style (str, optional): The style of spectral normalization.
            If set to `ajbrock`, implementation by
            ajbrock(https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py)
            will be adopted.
            If set to `torch`, implementation by `PyTorch` will be adopted.
            Defaults to `torch`.
        sn_eps (float, optional): eps for spectral normalization operation.
            Defaults to `1e-12`.
        init_cfg (dict, optional): Config for weight initialization.
            Default to ``dict(type='BigGAN')``.
        pretrained (str | dict , optional): Path for the pretrained model or
            dict containing information for pretrained models whose necessary
            key is 'ckpt_path'. Besides, you can also provide 'prefix' to load
            the generator part from the whole state dict.  Defaults to None.
    """

    # default channel factors
    _defualt_channels_cfg = {
        32: [1, 1, 1],
        64: [2, 4, 8, 16],
        128: [2, 4, 8, 16, 16],
    }

    # default downsample behavior
    _defualt_downsample_cfg = {
        32: [True, False, False],
        64: [True, True, True, True],
        128: [True, True, True, True, False]
    }

    def __init__(self,
                 input_scale,
                 num_classes=0,
                 base_channels=128,
                 input_channels=3,
                 attention_cfg=dict(type='SelfAttentionBlock'),
                 attention_after_nth_block=-1,
                 channels_cfg=None,
                 downsample_cfg=None,
                 from_rgb_cfg=dict(type='SNGANDiscHeadResBlock'),
                 blocks_cfg=dict(type='SNGANDiscResBlock'),
                 act_cfg=dict(type='ReLU'),
                 with_spectral_norm=True,
                 sn_style='torch',
                 sn_eps=1e-12,
                 init_cfg=dict(type='BigGAN'),
                 pretrained=None):

        super().__init__()

        self.init_type = init_cfg.get('type', None)

        # add SN options and activation function options to cfg
        self.from_rgb_cfg = deepcopy(from_rgb_cfg)
        self.from_rgb_cfg.setdefault('act_cfg', act_cfg)
        self.from_rgb_cfg.setdefault('with_spectral_norm', with_spectral_norm)
        self.from_rgb_cfg.setdefault('sn_style', sn_style)
        self.from_rgb_cfg.setdefault('init_cfg', init_cfg)

        # add SN options and activation function options to cfg
        self.blocks_cfg = deepcopy(blocks_cfg)
        self.blocks_cfg.setdefault('act_cfg', act_cfg)
        self.blocks_cfg.setdefault('with_spectral_norm', with_spectral_norm)
        self.blocks_cfg.setdefault('sn_style', sn_style)
        self.blocks_cfg.setdefault('sn_eps', sn_eps)
        self.blocks_cfg.setdefault('init_cfg', init_cfg)

        channels_cfg = deepcopy(self._defualt_channels_cfg) \
            if channels_cfg is None else deepcopy(channels_cfg)
        if isinstance(channels_cfg, dict):
            if input_scale not in channels_cfg:
                raise KeyError(f'`input_scale={input_scale} is not found in '
                               '`channel_cfg`, only support configs for '
                               f'{[chn for chn in channels_cfg.keys()]}')
            self.channel_factor_list = channels_cfg[input_scale]
        elif isinstance(channels_cfg, list):
            self.channel_factor_list = channels_cfg
        else:
            raise ValueError('Only support list or dict for `channel_cfg`, '
                             f'receive {type(channels_cfg)}')

        downsample_cfg = deepcopy(self._defualt_downsample_cfg) \
            if downsample_cfg is None else deepcopy(downsample_cfg)
        if isinstance(downsample_cfg, dict):
            if input_scale not in downsample_cfg:
                raise KeyError(f'`output_scale={input_scale} is not found in '
                               '`downsample_cfg`, only support configs for '
                               f'{[chn for chn in downsample_cfg.keys()]}')
            self.downsample_list = downsample_cfg[input_scale]
        elif isinstance(downsample_cfg, list):
            self.downsample_list = downsample_cfg
        else:
            raise ValueError('Only support list or dict for `channel_cfg`, '
                             f'receive {type(downsample_cfg)}')

        if len(self.downsample_list) != len(self.channel_factor_list):
            raise ValueError('`downsample_cfg` should have same length with '
                             '`channels_cfg`, but receive '
                             f'{len(self.downsample_list)} and '
                             f'{len(self.channel_factor_list)}.')

        # check `attention_after_nth_block`
        if not isinstance(attention_after_nth_block, list):
            attention_after_nth_block = [attention_after_nth_block]
        if not all([isinstance(idx, int)
                    for idx in attention_after_nth_block]):
            raise ValueError('`attention_after_nth_block` only support int or '
                             'a list of int. Please check your input type.')

        self.from_rgb = MODELS.build(
            self.from_rgb_cfg,
            default_args=dict(
                in_channels=input_channels, out_channels=base_channels))

        self.conv_blocks = nn.ModuleList()
        # add self-attention block after the first block
        if 1 in attention_after_nth_block:
            attn_cfg_ = deepcopy(attention_cfg)
            attn_cfg_['in_channels'] = base_channels
            attn_cfg_['sn_style'] = sn_style
            self.conv_blocks.append(MODELS.build(attn_cfg_))

        for idx in range(len(self.downsample_list)):
            factor_input = 1 if idx == 0 else self.channel_factor_list[idx - 1]
            factor_output = self.channel_factor_list[idx]

            # get block-specific config
            block_cfg_ = deepcopy(self.blocks_cfg)
            block_cfg_['downsample'] = self.downsample_list[idx]
            block_cfg_['in_channels'] = factor_input * base_channels
            block_cfg_['out_channels'] = factor_output * base_channels
            self.conv_blocks.append(MODELS.build(block_cfg_))

            # build self-attention block
            # the first ConvBlock is `from_rgb` block,
            # add 2 to get the index of the ConvBlocks
            if idx + 2 in attention_after_nth_block:
                attn_cfg_ = deepcopy(attention_cfg)
                attn_cfg_['in_channels'] = factor_output * base_channels
                self.conv_blocks.append(MODELS.build(attn_cfg_))

        self.decision = nn.Linear(factor_output * base_channels, 1)

        if with_spectral_norm:
            self.decision = spectral_norm(self.decision)

        self.num_classes = num_classes

        # In this case, discriminator is designed for conditional synthesis.
        if num_classes > 0:
            self.proj_y = nn.Embedding(num_classes,
                                       factor_output * base_channels)
            if with_spectral_norm:
                self.proj_y = spectral_norm(self.proj_y)

        self.activate = MODELS.build(act_cfg)
        self.init_weights(pretrained)

    def forward(self, x, label=None):
        """Forward function. If `self.num_classes` is larger than 0, label
        projection would be used.

        Args:
            x (torch.Tensor): Fake or real image tensor.
            label (torch.Tensor, options): Label correspond to the input image.
                Noted that, if `self.num_classed` is larger than 0,
                `label` should not be None.  Default to None.

        Returns:
            torch.Tensor: Prediction for the reality of the input image.
        """
        h = self.from_rgb(x)
        for conv_block in self.conv_blocks:
            h = conv_block(h)
        h = self.activate(h)
        h = torch.sum(h, dim=[2, 3])
        out = self.decision(h)

        if self.num_classes > 0:
            w_y = self.proj_y(label)
            out = out + torch.sum(w_y * h, dim=1, keepdim=True)
        return out.view(out.size(0), -1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for SNGAN-Proj and SAGAN. If ``pretrained=None`` and
        weight initialization would follow the ``INIT_TYPE`` in
        ``init_cfg=dict(type=INIT_TYPE)``.

        For SNGAN-Proj
        (``INIT_TYPE.upper() in ['SNGAN', 'SNGAN-PROJ', 'GAN-PROJ']``),
        we follow the initialization method in the official Chainer's
        implementation (https://github.com/pfnet-research/sngan_projection).

        For SAGAN (``INIT_TYPE.upper() == 'SAGAN'``), we follow the
        initialization method in official tensorflow's implementation
        (https://github.com/brain-research/self-attention-gan).

        Besides the reimplementation of the official code's initialization, we
        provide BigGAN's and Pytorch-StudioGAN's style initialization
        (``INIT_TYPE.upper() == BIGGAN`` and ``INIT_TYPE.upper() == STUDIO``).
        Please refer to https://github.com/ajbrock/BigGAN-PyTorch and
        https://github.com/POSTECH-CVLab/PyTorch-StudioGAN.

        Args:
            pretrained (str | dict, optional): Path for the pretrained model or
                dict containing information for pretrained models whose
                necessary key is 'ckpt_path'. Besides, you can also provide
                'prefix' to load the generator part from the whole state dict.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif isinstance(pretrained, dict):
            ckpt_path = pretrained.get('ckpt_path', None)
            assert ckpt_path is not None
            prefix = pretrained.get('prefix', '')
            map_location = pretrained.get('map_location', 'cpu')
            strict = pretrained.get('strict', True)
            state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path,
                                                      map_location)
            self.load_state_dict(state_dict, strict=strict)
        elif pretrained is None:
            if self.init_type.upper() == 'STUDIO':
                # initialization method from Pytorch-StudioGAN
                #   * weight: orthogonal_init gain=1
                #   * bias  : 0
                for m in self.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                        nn.init.orthogonal_(m.weight, gain=1)
                        if hasattr(m, 'bias') and m.bias is not None:
                            m.bias.data.fill_(0.)
            elif self.init_type.upper() == 'BIGGAN':
                # initialization method from BigGAN-pytorch
                #   * weight: xavier_init gain=1
                #   * bias  : default
                for m in self.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                        xavier_uniform_(m.weight, gain=1)
            elif self.init_type.upper() == 'SAGAN':
                # initialization method from official tensorflow code
                #   * weight: xavier_init gain=1
                #   * bias  : 0
                for m in self.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                        xavier_init(m, gain=1, distribution='uniform')
            elif self.init_type.upper() in ['SNGAN', 'SNGAN-PROJ', 'GAN-PROJ']:
                # initialization method from the official chainer code
                #   * embedding.weight: xavier_init gain=1
                #   * conv.weight     : xavier_init gain=sqrt(2)
                #   * shortcut.weight : xavier_init gain=1
                #   * bias            : 0
                for n, m in self.named_modules():
                    if isinstance(m, nn.Conv2d):
                        if 'shortcut' in n:
                            xavier_init(m, gain=1, distribution='uniform')
                        else:
                            xavier_init(
                                m, gain=np.sqrt(2), distribution='uniform')
                    if isinstance(m, (nn.Linear, nn.Embedding)):
                        xavier_init(m, gain=1, distribution='uniform')
            else:
                raise NotImplementedError('Unknown initialization method: '
                                          f'\'{self.init_type}\'')
        else:
            raise TypeError("'pretrained' must by a str or None. "
                            f'But receive {type(pretrained)}.')

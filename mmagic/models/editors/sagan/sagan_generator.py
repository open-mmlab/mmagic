# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine import is_list_of
from mmengine.dist import is_distributed
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, constant_init, xavier_init
from mmengine.runner import load_checkpoint
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix
from torch.nn.init import xavier_uniform_
from torch.nn.utils import spectral_norm

from mmagic.models.utils import get_module_device
from mmagic.registry import MODELS


@MODELS.register_module('SAGANGenerator')
@MODELS.register_module()
class SNGANGenerator(BaseModule):
    r"""Generator for SNGAN / Proj-GAN. The implementation refers to
    https://github.com/pfnet-research/sngan_projection/tree/master/gen_models

    In our implementation, we have two notable design. Namely,
    ``channels_cfg`` and ``blocks_cfg``.

    ``channels_cfg``: In default config of SNGAN / Proj-GAN, the number of
        ResBlocks and the channels of those blocks are corresponding to the
        resolution of the output image. Therefore, we allow user to define
        ``channels_cfg`` to try their own models. We also provide a default
        config to allow users to build the model only from the output
        resolution.

    ``block_cfg``: In reference code, the generator consists of a group of
        ResBlock. However, in our implementation, to make this model more
        generalize, we support defining ``blocks_cfg`` by users and loading
        the blocks by calling the build_module method.

    Args:
        output_scale (int): Output scale for the generated image.
        num_classes (int, optional): The number classes you would like to
            generate. This arguments would influence the structure of the
            intermediate blocks and label sampling operation in ``forward``
            (e.g. If num_classes=0, ConditionalNormalization layers would
            degrade to unconditional ones.). This arguments would be passed
            to intermediate blocks by overwrite their config. Defaults to 0.
        base_channels (int, optional): The basic channel number of the
            generator. The other layers contains channels based on this number.
            Default to 64.
        out_channels (int, optional): Channels of the output images.
            Default to 3.
        input_scale (int, optional): Input scale for the features.
            Defaults to 4.
        noise_size (int, optional): Size of the input noise vector.
            Default to 128.
        attention_cfg (dict, optional): Config for the self-attention block.
            Default to ``dict(type='SelfAttentionBlock')``.
        attention_after_nth_block (int | list[int], optional): Self attention
            block would be added after which *ConvBlock*. If ``int`` is passed,
            only one attention block would be added. If ``list`` is passed,
            self-attention blocks would be added after multiple ConvBlocks.
            To be noted that if the input is smaller than ``1``,
            self-attention corresponding to this index would be ignored.
            Default to 0.
        channels_cfg (list | dict[list], optional): Config for input channels
            of the intermediate blocks. If list is passed, each element of the
            list means the input channels of current block is how many times
            compared to the ``base_channels``. For block ``i``, the input and
            output channels should be ``channels_cfg[i]`` and
            ``channels_cfg[i+1]`` If dict is provided, the key of the dict
            should be the output scale and corresponding value should be a list
            to define channels.  Default: Please refer to
            ``_defualt_channels_cfg``.
        blocks_cfg (dict, optional): Config for the intermediate blocks.
            Defaults to ``dict(type='SNGANGenResBlock')``
        act_cfg (dict, optional): Activation config for the final output
            layer. Defaults to ``dict(type='ReLU')``.
        use_cbn (bool, optional): Whether use conditional normalization. This
            argument would pass to norm layers. Defaults to True.
        auto_sync_bn (bool, optional): Whether convert Batch Norm to
            Synchronized ones when Distributed training is on. Defaults to
            True.
        with_spectral_norm (bool, optional): Whether use spectral norm for
            conv blocks or not. Default to False.
        with_embedding_spectral_norm (bool, optional): Whether use spectral
            norm for embedding layers in normalization blocks or not. If not
            specified (set as ``None``), ``with_embedding_spectral_norm`` would
            be set as the same value as ``with_spectral_norm``.
            Defaults to None.
        sn_style (str, optional): The style of spectral normalization.
            If set to `ajbrock`, implementation by
            ajbrock(https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py)
            will be adopted.
            If set to `torch`, implementation by `PyTorch` will be adopted.
            Defaults to `torch`.
        norm_eps (float, optional): eps for Normalization layers (both
            conditional and non-conditional ones). Default to `1e-4`.
        sn_eps (float, optional): eps for spectral normalization operation.
            Defaults to `1e-12`.
        init_cfg (string, optional): Config for weight initialization.
            Defaults to ``dict(type='BigGAN')``.
        pretrained (str | dict, optional): Path for the pretrained model or
            dict containing information for pretrained models whose necessary
            key is 'ckpt_path'. Besides, you can also provide 'prefix' to load
            the generator part from the whole state dict.  Defaults to None.
        rgb_to_bgr (bool, optional): Whether to reformat the output channels
                with order `bgr`. We provide several pre-trained BigGAN
                weights whose output channels order is `rgb`. You can set
                this argument to True to use the weights.
    """

    # default channel factors
    _default_channels_cfg = {
        32: [1, 1, 1],
        64: [16, 8, 4, 2],
        128: [16, 16, 8, 4, 2]
    }

    def __init__(self,
                 output_scale,
                 num_classes=0,
                 base_channels=64,
                 out_channels=3,
                 input_scale=4,
                 noise_size=128,
                 attention_cfg=dict(type='SelfAttentionBlock'),
                 attention_after_nth_block=0,
                 channels_cfg=None,
                 blocks_cfg=dict(type='SNGANGenResBlock'),
                 act_cfg=dict(type='ReLU'),
                 use_cbn=True,
                 auto_sync_bn=True,
                 with_spectral_norm=False,
                 with_embedding_spectral_norm=None,
                 sn_style='torch',
                 norm_eps=1e-4,
                 sn_eps=1e-12,
                 init_cfg=dict(type='BigGAN'),
                 pretrained=None,
                 rgb_to_bgr=False):

        super().__init__()

        self.input_scale = input_scale
        self.output_scale = output_scale
        self.noise_size = noise_size
        self.num_classes = num_classes
        self.init_type = init_cfg.get('type', None)
        self.rgb_to_bgr = rgb_to_bgr

        self.blocks_cfg = deepcopy(blocks_cfg)

        self.blocks_cfg.setdefault('num_classes', num_classes)
        self.blocks_cfg.setdefault('act_cfg', act_cfg)
        self.blocks_cfg.setdefault('use_cbn', use_cbn)
        self.blocks_cfg.setdefault('auto_sync_bn', auto_sync_bn)
        self.blocks_cfg.setdefault('with_spectral_norm', with_spectral_norm)

        # set `norm_spectral_norm` as `with_spectral_norm` if not defined
        with_embedding_spectral_norm = with_embedding_spectral_norm \
            if with_embedding_spectral_norm is not None else with_spectral_norm
        self.blocks_cfg.setdefault('with_embedding_spectral_norm',
                                   with_embedding_spectral_norm)
        self.blocks_cfg.setdefault('init_cfg', init_cfg)
        self.blocks_cfg.setdefault('sn_style', sn_style)
        self.blocks_cfg.setdefault('norm_eps', norm_eps)
        self.blocks_cfg.setdefault('sn_eps', sn_eps)

        channels_cfg = deepcopy(self._default_channels_cfg) \
            if channels_cfg is None else deepcopy(channels_cfg)
        if isinstance(channels_cfg, dict):
            if output_scale not in channels_cfg:
                raise KeyError(f'`output_scale={output_scale} is not found in '
                               '`channel_cfg`, only support configs for '
                               f'{[chn for chn in channels_cfg.keys()]}')
            self.channel_factor_list = channels_cfg[output_scale]
        elif isinstance(channels_cfg, list):
            self.channel_factor_list = channels_cfg
        else:
            raise ValueError('Only support list or dict for `channel_cfg`, '
                             f'receive {type(channels_cfg)}')

        self.noise2feat = nn.Linear(
            noise_size,
            input_scale**2 * base_channels * self.channel_factor_list[0])
        if with_spectral_norm:
            self.noise2feat = spectral_norm(self.noise2feat)

        # check `attention_after_nth_block`
        if not isinstance(attention_after_nth_block, list):
            attention_after_nth_block = [attention_after_nth_block]
        if not is_list_of(attention_after_nth_block, int):
            raise ValueError('`attention_after_nth_block` only support int or '
                             'a list of int. Please check your input type.')

        self.conv_blocks = nn.ModuleList()
        self.attention_block_idx = []
        for idx in range(len(self.channel_factor_list)):
            factor_input = self.channel_factor_list[idx]
            factor_output = self.channel_factor_list[idx+1] \
                if idx < len(self.channel_factor_list)-1 else 1

            # get block-specific config
            block_cfg_ = deepcopy(self.blocks_cfg)
            block_cfg_['in_channels'] = factor_input * base_channels
            block_cfg_['out_channels'] = factor_output * base_channels
            self.conv_blocks.append(MODELS.build(block_cfg_))

            # build self-attention block
            # `idx` is start from 0, add 1 to get the index
            if idx + 1 in attention_after_nth_block:
                self.attention_block_idx.append(len(self.conv_blocks))
                attn_cfg_ = deepcopy(attention_cfg)
                attn_cfg_['in_channels'] = factor_output * base_channels
                attn_cfg_['sn_style'] = sn_style
                self.conv_blocks.append(MODELS.build(attn_cfg_))

        to_rgb_norm_cfg = dict(type='BN', eps=norm_eps)
        if is_distributed() and auto_sync_bn:
            to_rgb_norm_cfg['type'] = 'SyncBN'

        self.to_rgb = ConvModule(
            factor_output * base_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            norm_cfg=to_rgb_norm_cfg,
            act_cfg=act_cfg,
            order=('norm', 'act', 'conv'),
            with_spectral_norm=with_spectral_norm)
        self.final_act = MODELS.build(dict(type='Tanh'))

        self.init_weights(pretrained)

    def forward(self, noise, num_batches=0, label=None, return_noise=False):
        """Forward function.

        Args:
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            label (torch.Tensor | callable | None): You can directly give a
                batch of label through a ``torch.Tensor`` or offer a callable
                function to sample a batch of label data. Otherwise, the
                ``None`` indicates to use the default label sampler.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.

        Returns:
            torch.Tensor | dict: If not ``return_noise``, only the output
                image will be returned. Otherwise, a dict contains
                ``fake_image``, ``noise_batch`` and ``label_batch``
                would be returned.
        """
        if isinstance(noise, torch.Tensor):
            assert noise.shape[1] == self.noise_size
            assert noise.ndim == 2, ('The noise should be in shape of (n, c), '
                                     f'but got {noise.shape}')
            noise_batch = noise
        # receive a noise generator and sample noise.
        elif callable(noise):
            noise_generator = noise
            assert num_batches > 0
            noise_batch = noise_generator((num_batches, self.noise_size))
        # otherwise, we will adopt default noise sampler.
        else:
            assert num_batches > 0
            noise_batch = torch.randn((num_batches, self.noise_size))

        if isinstance(label, torch.Tensor):
            if label.ndim != 1:
                assert all([s == 1 for s in label.shape[1:]])
                label = label.view(-1)
            assert label.ndim == 1, ('The label should be in shape of (n, )'
                                     f'but got {label.shape}.')
            label_batch = label
        elif callable(label):
            label_generator = label
            assert num_batches > 0
            label_batch = label_generator(num_batches)
        elif self.num_classes == 0:
            label_batch = None
        else:
            assert num_batches > 0
            label_batch = torch.randint(0, self.num_classes, (num_batches, ))

        # dirty code for putting data on the right device
        noise_batch = noise_batch.to(get_module_device(self))
        if label_batch is not None:
            label_batch = label_batch.to(get_module_device(self))

        x = self.noise2feat(noise_batch)
        x = x.reshape(x.size(0), -1, self.input_scale, self.input_scale)

        for idx, conv_block in enumerate(self.conv_blocks):
            if idx in self.attention_block_idx:
                x = conv_block(x)
            else:
                x = conv_block(x, label_batch)

        out_feat = self.to_rgb(x)
        out_img = self.final_act(out_feat)

        if self.rgb_to_bgr:
            out_img = out_img[:, [2, 1, 0], ...]

        if return_noise:
            return dict(
                fake_img=out_img, noise_batch=noise_batch, label=label_batch)
        return out_img

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for SNGAN-Proj and SAGAN. If ``pretrained=None``,
        weight initialization would follow the ``INIT_TYPE`` in
        ``init_cfg=dict(type=INIT_TYPE)``.

        For SNGAN-Proj,
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
            if self.init_type.upper() in 'STUDIO':
                # initialization method from Pytorch-StudioGAN
                #   * weight: orthogonal_init gain=1
                #   * bias  : 0
                for m in self.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                        nn.init.orthogonal_(m.weight)
                        if hasattr(m, 'bias') and m.bias is not None:
                            m.bias.data.fill_(0.)
            elif self.init_type.upper() == 'BIGGAN':
                # initialization method from BigGAN-pytorch
                #   * weight: xavier_init gain=1
                #   * bias  : default
                for n, m in self.named_modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                        xavier_uniform_(m.weight, gain=1)
            elif self.init_type.upper() == 'SAGAN':
                # initialization method from official tensorflow code
                #   * weight          : xavier_init gain=1
                #   * bias            : 0
                #   * weight_embedding: 1
                #   * bias_embedding  : 0
                for n, m in self.named_modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        xavier_init(m, gain=1, distribution='uniform')
                    if isinstance(m, nn.Embedding):
                        # To be noted that here we initialize the embedding
                        # layer in cBN with specific prefix. If you implement
                        # your own cBN and want to use this initialization
                        # method, please make sure the embedding layers in
                        # your implementation have the same prefix as ours.
                        if 'weight' in n:
                            constant_init(m, 1)
                        if 'bias' in n:
                            constant_init(m, 0)
            elif self.init_type.upper() in ['SNGAN', 'SNGAN-PROJ', 'GAN-PROJ']:
                # initialization method from the official chainer code
                #   * conv.weight     : xavier_init gain=sqrt(2)
                #   * shortcut.weight : xavier_init gain=1
                #   * bias            : 0
                #   * weight_embedding: 1
                #   * bias_embedding  : 0
                for n, m in self.named_modules():
                    if isinstance(m, nn.Conv2d):
                        if 'shortcut' in n or 'to_rgb' in n:
                            xavier_init(m, gain=1, distribution='uniform')
                        else:
                            xavier_init(
                                m, gain=np.sqrt(2), distribution='uniform')
                    if isinstance(m, nn.Linear):
                        xavier_init(m, gain=1, distribution='uniform')
                    if isinstance(m, nn.Embedding):
                        # To be noted that here we initialize the embedding
                        # layer in cBN with specific prefix. If you implement
                        # your own cBN and want to use this initialization
                        # method, please make sure the embedding layers in
                        # your implementation have the same prefix as ours.
                        if 'weight' in n:
                            constant_init(m, 1)
                        if 'bias' in n:
                            constant_init(m, 0)
            else:
                raise NotImplementedError('Unknown initialization method: '
                                          f'\'{self.init_type}\'')

        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But receive {type(pretrained)}.')

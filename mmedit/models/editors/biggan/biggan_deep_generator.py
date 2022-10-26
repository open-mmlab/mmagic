# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import mmengine
import torch
import torch.nn as nn
from mmengine.logging import MMLogger
from mmengine.model import normal_init, xavier_init
from mmengine.runner import load_checkpoint
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix
from torch.nn.utils import spectral_norm

from mmedit.registry import MODULES
from ...utils import get_module_device
from .biggan_modules import SelfAttentionBlock, SNConvModule
from .biggan_snmodule import SNLinear


@MODULES.register_module()
class BigGANDeepGenerator(nn.Module):
    """BigGAN-Deep Generator. The implementation refers to
    https://github.com/ajbrock/BigGAN-PyTorch/blob/master/BigGANdeep.py # noqa.

    In BigGAN, we use a SAGAN-based architecture composing of an
    self-attention block and number of convolutional residual blocks
    with spectral normalization. BigGAN-deep follow the same architecture.

    The main difference between BigGAN and BigGAN-deep is that
    BigGAN-deep uses deeper residual blocks to construct the whole
    model.

    More details can be found in: Large Scale GAN Training for High Fidelity
    Natural Image Synthesis (ICLR2019).

    The design of the model structure is highly corresponding to the output
    resolution. For the original BigGAN-Deep's generator, you can set ``output_scale``
    as you need and use the default value of ``arch_cfg`` and ``blocks_cfg``.
    If you want to customize the model, you can set the arguments in this way:

    ``arch_cfg``: Config for the architecture of this generator. You can refer
    the ``_default_arch_cfgs`` in the ``_get_default_arch_cfg`` function to see
    the format of the ``arch_cfg``. Basically, you need to provide information
    of each block such as the numbers of input and output channels, whether to
    perform upsampling, etc.

    ``blocks_cfg``: Config for the convolution block. You can adjust block params
    like ``channel_ratio`` here. You can also replace the block type
    to your registered customized block. However, you should notice that some
    params are shared among these blocks like ``act_cfg``, ``with_spectral_norm``,
    ``sn_eps``, etc.

    Args:
        output_scale (int): Output scale for the generated image.
        noise_size (int, optional): Size of the input noise vector. Defaults
            to 120.
        num_classes (int, optional): The number of conditional classes. If set
            to 0, this model will be degraded to an unconditional model.
            Defaults to 0.
        out_channels (int, optional): Number of channels in output images.
            Defaults to 3.
        base_channels (int, optional): The basic channel number of the
            generator. The other layers contains channels based on this number.
            Defaults to 96.
        block_depth (int, optional): The repeat times of Residual Blocks in
            each level of architecture. Defaults to 2.
        input_scale (int, optional): The scale of the input 2D feature map.
            Defaults to 4.
        with_shared_embedding (bool, optional): Whether to use shared
            embedding. Defaults to True.
        shared_dim (int, optional): The output channels of shared embedding.
            Defaults to 128.
        sn_eps (float, optional): Epsilon value for spectral normalization.
            Defaults to 1e-6.
        sn_style (str, optional): The style of spectral normalization.
            If set to `ajbrock`, implementation by
            ajbrock(https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py)
            will be adopted.
            If set to `torch`, implementation by `PyTorch` will be adopted.
            Defaults to `ajbrock`.
        init_type (str, optional): The name of an initialization method:
            ortho | N02 | xavier. Defaults to 'ortho'.
        concat_noise (bool, optional): Whether to concat input noise vector
            with class vector. Defaults to True.
        act_cfg (dict, optional): Config for the activation layer. Defaults to
            dict(type='ReLU').
        upsample_cfg (dict, optional): Config for the upsampling operation.
            Defaults to dict(type='nearest', scale_factor=2).
        with_spectral_norm (bool, optional): Whether to use spectral
            normalization. Defaults to True.
        auto_sync_bn (bool, optional): Whether to use synchronized batch
            normalization. Defaults to True.
        blocks_cfg (dict, optional): Config for the convolution block. Defaults
            to dict(type='BigGANGenResBlock').
        arch_cfg (dict, optional): Config for the architecture of this
            generator. Defaults to None.
        out_norm_cfg (dict, optional): Config for the norm of output layer.
            Defaults to dict(type='BN').
        pretrained (str | dict, optional): Path for the pretrained model or
            dict containing information for pretained models whose necessary
            key is 'ckpt_path'. Besides, you can also provide 'prefix' to load
            the generator part from the whole state dict. Defaults to None.
        rgb2bgr (bool, optional): Whether to reformat the output channels
                with order `bgr`. We provide several pre-trained BigGAN-Deep
                weights whose output channels order is `rgb`. You can set
                this argument to True to use the weights.
    """

    def __init__(self,
                 output_scale,
                 noise_size=120,
                 num_classes=0,
                 out_channels=3,
                 base_channels=96,
                 block_depth=2,
                 input_scale=4,
                 with_shared_embedding=True,
                 shared_dim=128,
                 sn_eps=1e-6,
                 sn_style='ajbrock',
                 init_type='ortho',
                 concat_noise=True,
                 act_cfg=dict(type='ReLU', inplace=False),
                 upsample_cfg=dict(type='nearest', scale_factor=2),
                 with_spectral_norm=True,
                 auto_sync_bn=True,
                 blocks_cfg=dict(type='BigGANDeepGenResBlock'),
                 arch_cfg=None,
                 out_norm_cfg=dict(type='BN'),
                 pretrained=None,
                 rgb2bgr=False):
        super().__init__()
        self.noise_size = noise_size
        self.num_classes = num_classes
        self.shared_dim = shared_dim
        self.with_shared_embedding = with_shared_embedding
        self.output_scale = output_scale
        self.arch = arch_cfg if arch_cfg else self._get_default_arch_cfg(
            self.output_scale, base_channels)
        self.input_scale = input_scale
        self.concat_noise = concat_noise
        self.blocks_cfg = deepcopy(blocks_cfg)
        self.upsample_cfg = deepcopy(upsample_cfg)
        self.block_depth = block_depth
        self.rgb2bgr = rgb2bgr
        self.sn_style = sn_style

        # Validity Check
        # If 'num_classes' equals to zero, we shall set 'with_shared_embedding'
        # to False.
        if num_classes == 0:
            assert not self.with_shared_embedding
            assert not self.concat_noise
        elif not self.with_shared_embedding:
            # If not `with_shared_embedding`, we will use `nn.Embedding` to
            # replace the original `Linear` layer in conditional BN.
            # Meanwhile, we do not adopt split noises.
            assert not self.concat_noise

        # First linear layer
        if self.concat_noise:
            self.noise2feat = nn.Linear(
                self.noise_size + self.shared_dim,
                self.arch['in_channels'][0] * (self.input_scale**2))
        else:
            self.noise2feat = nn.Linear(
                self.noise_size,
                self.arch['in_channels'][0] * (self.input_scale**2))

        if with_spectral_norm:
            if sn_style == 'torch':
                self.noise2feat = spectral_norm(self.noise2feat, eps=sn_eps)
            elif sn_style == 'ajbrock':
                self.noise2feat = SNLinear(
                    self.noise_size +
                    (self.shared_dim if self.concat_noise else 0),
                    self.arch['in_channels'][0] * (self.input_scale**2),
                    eps=sn_eps)
            else:
                NotImplementedError(f'{sn_style} style SN is not supported')

        # If using 'shared_embedding', we will get an unified embedding of
        # label for all blocks. If not, we just pass the label to each
        # block.
        if with_shared_embedding:
            self.shared_embedding = nn.Embedding(num_classes, shared_dim)
        else:
            self.shared_embedding = nn.Identity()

        if num_classes > 0:
            if self.concat_noise:
                self.dim_after_concat = (
                    self.shared_dim + self.noise_size
                    if self.with_shared_embedding else self.num_classes)
            else:
                self.dim_after_concat = (
                    self.shared_dim
                    if self.with_shared_embedding else self.num_classes)
        else:
            self.dim_after_concat = 0
        self.blocks_cfg.update(
            dict(
                dim_after_concat=self.dim_after_concat,
                act_cfg=act_cfg,
                sn_eps=sn_eps,
                sn_style=sn_style,
                input_is_label=(num_classes > 0)
                and (not with_shared_embedding),
                with_spectral_norm=with_spectral_norm,
                auto_sync_bn=auto_sync_bn))

        self.conv_blocks = nn.ModuleList()
        for index, out_ch in enumerate(self.arch['out_channels']):
            for depth in range(self.block_depth):
                # change args to adapt to current block
                block_cfg_ = deepcopy(self.blocks_cfg)
                block_cfg_.update(
                    dict(
                        in_channels=self.arch['in_channels'][index],
                        out_channels=out_ch if depth == (self.block_depth - 1)
                        else self.arch['in_channels'][index],
                        upsample_cfg=self.upsample_cfg
                        if self.arch['upsample'][index]
                        and depth == (self.block_depth - 1) else None))
                self.conv_blocks.append(MODULES.build(block_cfg_))

            if self.arch['attention'][index]:
                self.conv_blocks.append(
                    SelfAttentionBlock(
                        out_ch,
                        with_spectral_norm=with_spectral_norm,
                        sn_eps=sn_eps,
                        sn_style=sn_style))

        self.output_layer = SNConvModule(
            self.arch['out_channels'][-1],
            out_channels,
            kernel_size=3,
            padding=1,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps, sn_style=sn_style),
            act_cfg=act_cfg,
            norm_cfg=out_norm_cfg,
            bias=True,
            order=('norm', 'act', 'conv'))

        self.init_weights(pretrained=pretrained, init_type=init_type)

    def _get_default_arch_cfg(self, output_scale, base_channels):
        assert output_scale in [32, 64, 128, 256, 512]
        _default_arch_cfgs = {
            '32': {
                'in_channels': [base_channels * item for item in [4, 4, 4]],
                'out_channels': [base_channels * item for item in [4, 4, 4]],
                'upsample': [True] * 3,
                'resolution': [8, 16, 32],
                'attention': [False, False, False]
            },
            '64': {
                'in_channels':
                [base_channels * item for item in [16, 16, 8, 4]],
                'out_channels':
                [base_channels * item for item in [16, 8, 4, 2]],
                'upsample': [True] * 4,
                'resolution': [8, 16, 32, 64],
                'attention': [False, False, False, True]
            },
            '128': {
                'in_channels':
                [base_channels * item for item in [16, 16, 8, 4, 2]],
                'out_channels':
                [base_channels * item for item in [16, 8, 4, 2, 1]],
                'upsample': [True] * 5,
                'resolution': [8, 16, 32, 64, 128],
                'attention': [False, False, False, True, False]
            },
            '256': {
                'in_channels':
                [base_channels * item for item in [16, 16, 8, 8, 4, 2]],
                'out_channels':
                [base_channels * item for item in [16, 8, 8, 4, 2, 1]],
                'upsample': [True] * 6,
                'resolution': [8, 16, 32, 64, 128, 256],
                'attention': [False, False, False, True, False, False]
            },
            '512': {
                'in_channels':
                [base_channels * item for item in [16, 16, 8, 8, 4, 2, 1]],
                'out_channels':
                [base_channels * item for item in [16, 8, 8, 4, 2, 1, 1]],
                'upsample': [True] * 7,
                'resolution': [8, 16, 32, 64, 128, 256, 512],
                'attention': [False, False, False, True, False, False, False]
            }
        }

        return _default_arch_cfgs[str(output_scale)]

    def forward(self,
                noise,
                label=None,
                num_batches=0,
                return_noise=False,
                truncation=-1.0,
                use_outside_embedding=False):
        """Forward function.

        Args:
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            label (torch.Tensor | callable | None): You can directly give a
                batch of label through a ``torch.Tensor`` or offer a callable
                function to sample a batch of label data. Otherwise, the
                ``None`` indicates to use the default label sampler.
                Defaults to None.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            return_noise (bool, optional): If True, ``noise_batch`` and
                ``label`` will be returned in a dict with ``fake_img``.
                Defaults to False.
            truncation (float, optional): Truncation factor. Give value not
                less than 0., the truncation trick will be adopted.
                Otherwise, the truncation trick will not be adopted.
                Defaults to -1..
            use_outside_embedding (bool, optional): Whether to use outside
                embedding or use `shared_embedding`. Set to `True` if
                embedding has already be performed outside this function.
                Default to False.

        Returns:
            torch.Tensor | dict: If not ``return_noise``, only the output image
                will be returned. Otherwise, a dict contains ``fake_img``,
                ``noise_batch`` and ``label`` will be returned.
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
        # perform truncation
        if truncation >= 0.0:
            noise_batch = torch.clamp(noise_batch, -1. * truncation,
                                      1. * truncation)

        if self.num_classes == 0:
            label_batch = None

        elif isinstance(label, torch.Tensor):
            if not use_outside_embedding:
                assert label.ndim == 1, (
                    'The label shoube be in shape of (n, )'
                    f'but got {label.shape}.')
            label_batch = label
        elif callable(label):
            label_generator = label
            assert num_batches > 0
            label_batch = label_generator((num_batches, ))
        else:
            assert num_batches > 0
            label_batch = torch.randint(0, self.num_classes, (num_batches, ))

        # dirty code for putting data on the right device
        noise_batch = noise_batch.to(get_module_device(self))
        if label_batch is not None:
            label_batch = label_batch.to(get_module_device(self))
            if not use_outside_embedding:
                class_vector = self.shared_embedding(label_batch)
            else:
                class_vector = label_batch
        else:
            class_vector = None

        # If 'concat noise', concat class vector and noise batch
        if self.concat_noise:
            if class_vector is not None:
                z = torch.cat([noise_batch, class_vector], dim=1)
                y = z
        elif self.num_classes > 0:
            z = noise_batch
            y = class_vector
        else:
            z = noise_batch
            y = None

        # First linear layer
        x = self.noise2feat(z)
        # Reshape
        # We use this conversion step to allow for loading TF weights
        # TF convention on shape is [batch, height, width, channels]
        # PT convention on shape is [batch, channels, height, width]
        x = x.view(x.size(0), self.input_scale, self.input_scale, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        # Loop over blocks
        for idx, conv_block in enumerate(self.conv_blocks):
            if isinstance(conv_block, SelfAttentionBlock):
                x = conv_block(x)
            else:
                x = conv_block(x, y)
        # Apply batchnorm-relu-conv-tanh at output
        x = self.output_layer(x)
        out_img = torch.tanh(x)

        if self.rgb2bgr:
            out_img = out_img[:, [2, 1, 0], ...]

        if return_noise:
            output = dict(
                fake_img=out_img, noise_batch=noise_batch, label=label_batch)
            return output

        return out_img

    def init_weights(self, pretrained=None, init_type='ortho'):
        """Init weights for models.

        Args:
            pretrained (str | dict, optional): Path for the pretrained model or
                dict containing information for pretained models whose
                necessary key is 'ckpt_path'. Besides, you can also provide
                'prefix' to load the generator part from the whole state dict.
                Defaults to None.
            init_type (str, optional): The name of an initialization method:
                ortho | N02 | xavier. Defaults to 'ortho'.
        """
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif isinstance(pretrained, dict):
            ckpt_path = pretrained.get('ckpt_path', None)
            assert ckpt_path is not None
            prefix = pretrained.get('prefix', '')
            map_location = pretrained.get('map_location', 'cpu')
            strict = pretrained.get('strict', True)
            state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path,
                                                      map_location)
            self.load_state_dict(state_dict, strict=strict)
            mmengine.print_log(f'Load pretrained model from {ckpt_path}')
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                    if init_type == 'ortho':
                        nn.init.orthogonal_(m.weight)
                    elif init_type == 'N02':
                        normal_init(m, 0.0, 0.02)
                    elif init_type == 'xavier':
                        xavier_init(m)
                    else:
                        raise NotImplementedError(
                            f'{init_type} initialization \
                            not supported now.')
        else:
            raise TypeError('pretrained must be a str or None but'
                            f' got {type(pretrained)} instead.')

# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch
import torch.nn as nn
from mmcv.cnn.bricks.conv_module import ConvModule
from mmengine.logging import MMLogger
from mmengine.model import constant_init
from mmengine.runner import load_checkpoint

from mmedit.registry import MODULES
from .modules import (DenoisingResBlock, EmbedSequential, TimeEmbedding,
                      convert_module_to_f16, convert_module_to_f32)


@MODULES.register_module()
class DenoisingUnet(nn.Module):
    """Denoising Unet. This network receives a diffused image ``x_t`` and
    current timestep ``t``, and returns a ``output_dict`` corresponding to the
    passed ``output_cfg``.

    ``output_cfg`` defines the number of channels and the meaning of the
    output. ``output_cfg`` mainly contains keys of ``mean`` and ``var``,
    denoting how the network outputs mean and variance required for the
    denoising process.
    For ``mean``:
    1. ``dict(mean='EPS')``: Model will predict noise added in the
        diffusion process, and the ``output_dict`` will contain a key named
        ``eps_t_pred``.
    2. ``dict(mean='START_X')``: Model will direct predict the mean of the
        original image `x_0`, and the ``output_dict`` will contain a key named
        ``x_0_pred``.
    3. ``dict(mean='X_TM1_PRED')``: Model will predict the mean of diffused
        image at `t-1` timestep, and the ``output_dict`` will contain a key
        named ``x_tm1_pred``.

    For ``var``:
    1. ``dict(var='FIXED_SMALL')`` or ``dict(var='FIXED_LARGE')``: Variance in
        the denoising process is regarded as a fixed value. Therefore only
        'mean' will be predicted, and the output channels will equal to the
        input image (e.g., three channels for RGB image.)
    2. ``dict(var='LEARNED')``: Model will predict `log_variance` in the
        denoising process, and the ``output_dict`` will contain a key named
        ``log_var``.
    3. ``dict(var='LEARNED_RANGE')``: Model will predict an interpolation
        factor and the `log_variance` will be calculated as
        `factor * upper_bound + (1-factor) * lower_bound`. The ``output_dict``
        will contain a key named ``factor``.

    If ``var`` is not ``FIXED_SMALL`` or ``FIXED_LARGE``, the number of output
    channels will be the double of input channels, where the first half part
    contains predicted mean values and the other part is the predicted
    variance values. Otherwise, the number of output channels equals to the
    input channels, only containing the predicted mean values.

    Args:
        image_size (int | list[int]): The size of image to denoise.
        in_channels (int, optional): The input channels of the input image.
            Defaults as ``3``.
        base_channels (int, optional): The basic channel number of the
            generator. The other layers contain channels based on this number.
            Defaults to ``128``.
        resblocks_per_downsample (int, optional): Number of ResBlock used
            between two downsample operations. The number of ResBlock between
            upsample operations will be the same value to keep symmetry.
            Defaults to 3.
        num_timesteps (int, optional): The total timestep of the denoising
            process and the diffusion process. Defaults to ``1000``.
        use_rescale_timesteps (bool, optional): Whether rescale the input
            timesteps in range of [0, 1000].  Defaults to ``True``.
        dropout (float, optional): The probability of dropout operation of
            each ResBlock. Pass ``0`` to do not use dropout. Defaults as 0.
        embedding_channels (int, optional): The output channels of time
            embedding layer and label embedding layer. If not passed (or
            passed ``-1``), output channels of the embedding layers will set
            as four times of ``base_channels``. Defaults to ``-1``.
        num_classes (int, optional): The number of conditional classes. If set
            to 0, this model will be degraded to an unconditional model.
            Defaults to 0.
        channels_cfg (list | dict[list], optional): Config for input channels
            of the intermedia blocks. If list is passed, each element of the
            list indicates the scale factor for the input channels of the
            current block with regard to the ``base_channels``. For block
            ``i``, the input and output channels should be
            ``channels_cfg[i] * base_channels`` and
            ``channels_cfg[i+1] * base_channels`` If dict is provided, the key
            of the dict should be the output scale and corresponding value
            should be a list to define channels. Default: Please refer to
            ``_defualt_channels_cfg``.
        output_cfg (dict, optional): Config for output variables. Defaults to
            ``dict(mean='eps', var='learned_range')``.
        norm_cfg (dict, optional): The config for normalization layers.
            Defaults to ``dict(type='GN', num_groups=32)``.
        act_cfg (dict, optional): The config for activation layers. Defaults
            to ``dict(type='SiLU', inplace=False)``.
        shortcut_kernel_size (int, optional): The kernel size for shortcut
            conv in ResBlocks. The value of this argument will overwrite the
            default value of `resblock_cfg`. Defaults to `3`.
        use_scale_shift_norm (bool, optional): Whether perform scale and shift
            after normalization operation. Defaults to True.
        num_heads (int, optional): The number of attention heads. Defaults to
            4.
        time_embedding_mode (str, optional): Embedding method of
            ``time_embedding``. Defaults to 'sin'.
        time_embedding_cfg (dict, optional): Config for ``time_embedding``.
            Defaults to None.
        resblock_cfg (dict, optional): Config for ResBlock. Defaults to
            ``dict(type='DenoisingResBlock')``.
        attention_cfg (dict, optional): Config for attention operation.
            Defaults to ``dict(type='MultiHeadAttention')``.
        upsample_conv (bool, optional): Whether use conv in upsample block.
            Defaults to ``True``.
        downsample_conv (bool, optional): Whether use conv operation in
            downsample block.  Defaults to ``True``.
        upsample_cfg (dict, optional): Config for upsample blocks.
            Defaults to ``dict(type='DenoisingDownsample')``.
        downsample_cfg (dict, optional): Config for downsample blocks.
            Defaults to ``dict(type='DenoisingUpsample')``.
        attention_res (int | list[int], optional): Resolution of feature maps
            to apply attention operation. Defaults to ``[16, 8]``.
        pretrained (str | dict, optional): Path for the pretrained model or
            dict containing information for pretained models whose necessary
            key is 'ckpt_path'. Besides, you can also provide 'prefix' to load
            the generator part from the whole state dict.  Defaults to None.
    """

    _default_channels_cfg = {
        512: [0.5, 1, 1, 2, 2, 4, 4],
        256: [1, 1, 2, 2, 4, 4],
        128: [1, 1, 2, 3, 4],
        64: [1, 2, 3, 4],
        32: [1, 2, 2, 2]
    }

    def __init__(self,
                 image_size,
                 in_channels=3,
                 base_channels=128,
                 resblocks_per_downsample=3,
                 num_timesteps=1000,
                 use_rescale_timesteps=False,
                 dropout=0,
                 embedding_channels=-1,
                 num_classes=0,
                 use_fp16=False,
                 channels_cfg=None,
                 output_cfg=dict(mean='eps', var='learned_range'),
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='SiLU', inplace=False),
                 shortcut_kernel_size=1,
                 use_scale_shift_norm=False,
                 resblock_updown=False,
                 num_heads=4,
                 time_embedding_mode='sin',
                 time_embedding_cfg=None,
                 resblock_cfg=dict(type='DenoisingResBlock'),
                 attention_cfg=dict(type='MultiHeadAttention'),
                 downsample_conv=True,
                 upsample_conv=True,
                 downsample_cfg=dict(type='DenoisingDownsample'),
                 upsample_cfg=dict(type='DenoisingUpsample'),
                 attention_res=[16, 8],
                 pretrained=None):

        super().__init__()

        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.use_rescale_timesteps = use_rescale_timesteps
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.output_cfg = deepcopy(output_cfg)
        self.mean_mode = self.output_cfg.get('mean', 'eps')
        self.var_mode = self.output_cfg.get('var', 'learned_range')
        self.in_channels = in_channels

        # double output_channels to output mean and var at same time
        out_channels = in_channels if 'FIXED' in self.var_mode.upper() \
            else 2 * in_channels
        self.out_channels = out_channels

        # check type of image_size
        if not isinstance(image_size, int) and not isinstance(
                image_size, list):
            raise TypeError(
                'Only support `int` and `list[int]` for `image_size`.')
        if isinstance(image_size, list):
            assert len(
                image_size) == 2, 'The length of `image_size` should be 2.'
            assert image_size[0] == image_size[
                1], 'Width and height of the image should be same.'
            image_size = image_size[0]
        self.image_size = image_size

        channels_cfg = deepcopy(self._default_channels_cfg) \
            if channels_cfg is None else deepcopy(channels_cfg)
        if isinstance(channels_cfg, dict):
            if image_size not in channels_cfg:
                raise KeyError(f'`image_size={image_size} is not found in '
                               '`channels_cfg`, only support configs for '
                               f'{[chn for chn in channels_cfg.keys()]}')
            self.channel_factor_list = channels_cfg[image_size]
        elif isinstance(channels_cfg, list):
            self.channel_factor_list = channels_cfg
        else:
            raise ValueError('Only support list or dict for `channels_cfg`, '
                             f'receive {type(channels_cfg)}')

        embedding_channels = base_channels * 4 \
            if embedding_channels == -1 else embedding_channels
        self.time_embedding = TimeEmbedding(
            base_channels,
            embedding_channels=embedding_channels,
            embedding_mode=time_embedding_mode,
            embedding_cfg=time_embedding_cfg,
            act_cfg=act_cfg)

        if self.num_classes != 0:
            self.label_embedding = nn.Embedding(self.num_classes,
                                                embedding_channels)

        self.resblock_cfg = deepcopy(resblock_cfg)
        self.resblock_cfg.setdefault('dropout', dropout)
        self.resblock_cfg.setdefault('norm_cfg', norm_cfg)
        self.resblock_cfg.setdefault('act_cfg', act_cfg)
        self.resblock_cfg.setdefault('embedding_channels', embedding_channels)
        self.resblock_cfg.setdefault('use_scale_shift_norm',
                                     use_scale_shift_norm)
        self.resblock_cfg.setdefault('shortcut_kernel_size',
                                     shortcut_kernel_size)

        # get scales of ResBlock to apply attention
        attention_scale = [image_size // int(res) for res in attention_res]
        self.attention_cfg = deepcopy(attention_cfg)
        self.attention_cfg.setdefault('num_heads', num_heads)
        self.attention_cfg.setdefault('norm_cfg', norm_cfg)

        self.downsample_cfg = deepcopy(downsample_cfg)
        self.downsample_cfg.setdefault('with_conv', downsample_conv)
        self.upsample_cfg = deepcopy(upsample_cfg)
        self.upsample_cfg.setdefault('with_conv', upsample_conv)

        # init the channel scale factor
        scale = 1
        ch = int(base_channels * self.channel_factor_list[0])
        self.in_blocks = nn.ModuleList(
            [EmbedSequential(nn.Conv2d(in_channels, ch, 3, 1, padding=1))])
        self.in_channels_list = [ch]

        # construct the encoder part of Unet
        for level, factor in enumerate(self.channel_factor_list):
            in_channels_ = ch if level == 0 \
                else int(base_channels * self.channel_factor_list[level - 1])
            out_channels_ = int(base_channels * factor)

            for _ in range(resblocks_per_downsample):
                layers = [
                    MODULES.build(
                        self.resblock_cfg,
                        default_args={
                            'in_channels': in_channels_,
                            'out_channels': out_channels_
                        })
                ]
                in_channels_ = out_channels_

                if scale in attention_scale:
                    layers.append(
                        MODULES.build(
                            self.attention_cfg,
                            default_args={'in_channels': in_channels_}))

                self.in_channels_list.append(in_channels_)
                self.in_blocks.append(EmbedSequential(*layers))

            if level != len(self.channel_factor_list) - 1:
                self.in_blocks.append(
                    EmbedSequential(
                        DenoisingResBlock(
                            out_channels_,
                            embedding_channels,
                            use_scale_shift_norm,
                            dropout,
                            norm_cfg=norm_cfg,
                            out_channels=out_channels_,
                            down=True) if resblock_updown else MODULES.
                        build(self.
                              downsample_cfg, {'in_channels': in_channels_})))
                self.in_channels_list.append(in_channels_)
                scale *= 2

        # construct the bottom part of Unet
        self.mid_blocks = EmbedSequential(
            MODULES.build(
                self.resblock_cfg, default_args={'in_channels': in_channels_}),
            MODULES.build(
                self.attention_cfg, default_args={'in_channels':
                                                  in_channels_}),
            MODULES.build(
                self.resblock_cfg, default_args={'in_channels': in_channels_}),
        )

        # construct the decoder part of Unet
        in_channels_list = deepcopy(self.in_channels_list)
        self.out_blocks = nn.ModuleList()
        for level, factor in enumerate(self.channel_factor_list[::-1]):
            for idx in range(resblocks_per_downsample + 1):
                layers = [
                    MODULES.build(
                        self.resblock_cfg,
                        default_args={
                            'in_channels':
                            in_channels_ + in_channels_list.pop(),
                            'out_channels': int(base_channels * factor)
                        })
                ]
                in_channels_ = int(base_channels * factor)
                if scale in attention_scale:
                    layers.append(
                        MODULES.build(
                            self.attention_cfg,
                            default_args={'in_channels': in_channels_}))
                if (level != len(self.channel_factor_list) - 1
                        and idx == resblocks_per_downsample):
                    out_channels_ = in_channels_
                    layers.append(
                        DenoisingResBlock(
                            in_channels_,
                            embedding_channels,
                            use_scale_shift_norm,
                            dropout,
                            norm_cfg=norm_cfg,
                            out_channels=out_channels_,
                            up=True) if resblock_updown else MODULES.
                        build(self.upsample_cfg, {'in_channels': in_channels_}
                              ))
                    scale //= 2
                self.out_blocks.append(EmbedSequential(*layers))

        self.out = ConvModule(
            in_channels=in_channels_,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            bias=True,
            order=('norm', 'act', 'conv'))

        self.init_weights(pretrained)

    def forward(self, x_t, t, label=None, return_noise=False):
        """Forward function.
        Args:
            x_t (torch.Tensor): Diffused image at timestep `t` to denoise.
            t (torch.Tensor): Current timestep.
            label (torch.Tensor | callable | None): You can directly give a
                batch of label through a ``torch.Tensor`` or offer a callable
                function to sample a batch of label data. Otherwise, the
                ``None`` indicates to use the default label sampler.
            return_noise (bool, optional): If True, inputted ``x_t`` and ``t``
                will be returned in a dict with output desired by
                ``output_cfg``. Defaults to False.

        Returns:
            torch.Tensor | dict: If not ``return_noise``
        """
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.long, device=x_t.device)
        elif torch.is_tensor(t) and len(t.shape) == 0:
            t = t[None].to(x_t.device)

        embedding = self.time_embedding(t)

        if label is not None:
            assert hasattr(self, 'label_embedding')
            embedding = self.label_embedding(label) + embedding

        h, hs = x_t, []
        h = h.type(self.dtype)
        # forward downsample blocks
        for block in self.in_blocks:
            h = block(h, embedding)
            hs.append(h)

        # forward middle blocks
        h = self.mid_blocks(h, embedding)

        # forward upsample blocks
        for block in self.out_blocks:
            h = block(torch.cat([h, hs.pop()], dim=1), embedding)
        h = h.type(x_t.dtype)
        outputs = self.out(h)

        return {'outputs': outputs}

    def init_weights(self, pretrained=None):
        """Init weights for models.

        We just use the initialization method proposed in the original paper.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            # As Improved-DDPM, we apply zero-initialization to
            #   second conv block in ResBlock (keywords: conv_2)
            #   the output layer of the Unet (keywords: 'out' but
            #     not 'out_blocks')
            #   projection layer in Attention layer (keywords: proj)
            for n, m in self.named_modules():
                if isinstance(m, nn.Conv2d) and ('conv_2' in n or
                                                 ('out' in n
                                                  and 'out_blocks' not in n)):
                    constant_init(m, 0)
                if isinstance(m, nn.Conv1d) and 'proj' in n:
                    constant_init(m, 0)
        else:
            raise TypeError('pretrained must be a str or None but'
                            f' got {type(pretrained)} instead.')

    def convert_to_fp16(self):
        """Convert the precision of the model to float16."""
        self.in_blocks.apply(convert_module_to_f16)
        self.mid_blocks.apply(convert_module_to_f16)
        self.out_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """Convert the precision of the model to float32."""
        self.in_blocks.apply(convert_module_to_f32)
        self.mid_blocks.apply(convert_module_to_f32)
        self.out_blocks.apply(convert_module_to_f32)

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.runner.amp import autocast
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix
from torch import Tensor

from mmagic.registry import MODELS
from ..stylegan1 import EqualLinearActModule
from ..stylegan3.stylegan3_modules import MappingNetwork
from .ada.augment import AugmentPipe
from .ada.misc import constant
from .stylegan2_modules import ConvDownLayer, ModMBStddevLayer, ResBlock


@MODELS.register_module('StyleGANv2Discriminator')
@MODELS.register_module()
class StyleGAN2Discriminator(BaseModule):
    """StyleGAN2 Discriminator.

    The architecture of this discriminator is proposed in StyleGAN2. More
    details can be found in: Analyzing and Improving the Image Quality of
    StyleGAN CVPR2020.

    You can load pretrained model through passing information into
    ``pretrained`` argument. We have already offered official weights as
    follows:

    - stylegan2-ffhq-config-f: https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth  # noqa
    - stylegan2-horse-config-f: https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-horse-config-f-official_20210327_173203-ef3e69ca.pth  # noqa
    - stylegan2-car-config-f: https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-car-config-f-official_20210327_172340-8cfe053c.pth  # noqa
    - stylegan2-cat-config-f: https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-cat-config-f-official_20210327_172444-15bc485b.pth  # noqa
    - stylegan2-church-config-f: https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth  # noqa

    If you want to load the ema model, you can just use following codes:

    .. code-block:: python

        # ckpt_http is one of the valid path from http source
        discriminator = StyleGAN2Discriminator(1024, 512,
                                               pretrained=dict(
                                                   ckpt_path=ckpt_http,
                                                   prefix='discriminator'))

    Of course, you can also download the checkpoint in advance and set
    ``ckpt_path`` with local path.

    Note that our implementation adopts BGR image as input, while the
    original StyleGAN2 provides RGB images to the discriminator. Thus, we
    provide ``bgr2rgb`` argument to convert the image space. If your images
    follow the RGB order, please set it to ``True`` accordingly.

    Args:
        in_size (int): The input size of images.
        img_channels (int): The number of channels of the input image. Defaults to 3.
        channel_multiplier (int, optional): The multiplier factor for the
            channel number. Defaults to 2.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 3, 3, 1].
        mbstd_cfg (dict, optional): Configs for minibatch-stddev layer.
            Defaults to dict(group_size=4, channel_groups=1).
        cond_size (int, optional): The size of conditional input. If None or
            less than 1, no conditional mapping will be applied. Defaults to None.
        cond_mapping_channels (int, optional): The dimension of the output of
            conditional mapping. Only work when :attr:`c_dim` is larger than 0.
            If :attr:`c_dim` is larger than 0 and :attr:`cmap_dim` is None, will.
            Defaults to None.
        cond_mapping_layers (int, optional): The number of mapping layer used to
            map conditional input. Only work when c_dim is larger than 0. If
            :attr:`cmapping_layer` is None and :attr:`c_dim` is larger than 0,
            cmapping_layer will set as 8. Defaults to None.
        num_fp16_scales (int, optional): The number of resolutions to use auto
            fp16 training. Defaults to 0.
        fp16_enabled (bool, optional): Whether to use fp16 training in this
            module. Defaults to False.
        out_fp32 (bool, optional): Whether to convert the output feature map to
            `torch.float32`. Defaults to `True`.
        convert_input_fp32 (bool, optional): Whether to convert input type to
            fp32 if not `fp16_enabled`. This argument is designed to deal with
            the cases where some modules are run in FP16 and others in FP32.
            Defaults to True.
        input_bgr2rgb (bool, optional): Whether to reformat the input channels
            with order `rgb`. Since we provide several converted weights,
            whose input order is `rgb`. You can set this argument to True if
            you want to finetune on converted weights. Defaults to False.
        pretrained (dict | None, optional): Information for pretrained models.
            The necessary key is 'ckpt_path'. Besides, you can also provide
            'prefix' to load the generator part from the whole state dict.
            Defaults to None.
    """

    def __init__(self,
                 in_size,
                 img_channels=3,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 mbstd_cfg=dict(group_size=4, channel_groups=1),
                 cond_size=None,
                 cond_mapping_channels=None,
                 cond_mapping_layers=None,
                 num_fp16_scales=0,
                 fp16_enabled=False,
                 out_fp32=True,
                 convert_input_fp32=True,
                 input_bgr2rgb=False,
                 init_cfg=None,
                 pretrained=None):
        # TODO: pretrained can be deleted later
        super().__init__(init_cfg=init_cfg)
        self.num_fp16_scale = num_fp16_scales
        self.fp16_enabled = fp16_enabled
        self.convert_input_fp32 = convert_input_fp32
        self.out_fp32 = out_fp32

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        log_size = int(np.log2(in_size))

        in_channels = channels[in_size]

        _use_fp16 = num_fp16_scales > 0 or fp16_enabled
        convs = [
            ConvDownLayer(
                img_channels, channels[in_size], 1, fp16_enabled=_use_fp16)
        ]

        for i in range(log_size, 2, -1):
            out_channel = channels[2**(i - 1)]

            # add fp16 training for higher resolutions
            _use_fp16 = (log_size - i) < num_fp16_scales or fp16_enabled

            convs.append(
                ResBlock(
                    in_channels,
                    out_channel,
                    blur_kernel,
                    fp16_enabled=_use_fp16,
                    convert_input_fp32=convert_input_fp32))

            in_channels = out_channel

        if cond_size is not None and cond_size > 0:
            cond_mapping_channels = 512 if cond_mapping_channels is None \
                else cond_mapping_channels
            cond_mapping_layers = 8 if cond_mapping_layers is None \
                else cond_mapping_layers
            self.mapping = MappingNetwork(
                noise_size=0,
                style_channels=cond_mapping_channels,
                cond_size=cond_size,
                num_ws=None,
                num_layers=cond_mapping_layers,
                w_avg_beta=None)

        self.convs = nn.Sequential(*convs)

        self.mbstd_layer = ModMBStddevLayer(**mbstd_cfg)

        self.final_conv = ConvDownLayer(
            in_channels + 1, channels[4], 3, fp16_enabled=fp16_enabled)

        if cond_size is None or cond_size <= 0:
            final_linear_out_channels = 1
        else:
            final_linear_out_channels = cond_mapping_channels
        self.final_linear = nn.Sequential(
            EqualLinearActModule(
                channels[4] * 4 * 4,
                channels[4],
                act_cfg=dict(type='fused_bias')),
            EqualLinearActModule(channels[4], final_linear_out_channels),
        )

        self.input_bgr2rgb = input_bgr2rgb
        if pretrained is not None:
            self._load_pretrained_model(**pretrained)

    def _load_pretrained_model(self,
                               ckpt_path,
                               prefix='',
                               map_location='cpu',
                               strict=True):
        state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path,
                                                  map_location)
        self.load_state_dict(state_dict, strict=strict)
        mmengine.print_log(f'Load pretrained model from {ckpt_path}')

    def forward(self, x: Tensor, label: Optional[Tensor] = None):
        """Forward function.

        Args:
            x (torch.Tensor): Input image tensor.
            label (torch.Tensor, optional): The conditional input feed to
                mapping layer. Defaults to None.

        Returns:
            torch.Tensor: Predict score for the input image.
        """
        # This setting was used to finetune on converted weights
        if self.input_bgr2rgb:
            x = x[:, [2, 1, 0], ...]

        # convs has own fp-16 controller, do not wrap here
        x = self.convs(x)

        x = self.mbstd_layer(x)

        fp16_enabled = (
            self.final_conv.fp16_enabled or not self.convert_input_fp32)
        with autocast(enabled=fp16_enabled):
            if not fp16_enabled:
                x = x.to(torch.float32)
            x = self.final_conv(x)
            x = x.view(x.shape[0], -1)
            x = self.final_linear(x)

            # conditioning
            if label is not None:
                assert self.mapping is not None, (
                    '\'self.mapping\' must not be None when conditional input '
                    'is passed.')
                cmap = self.mapping(None, label)
                x = (x * cmap).sum(
                    dim=1, keepdim=True) * (1 / np.sqrt(cmap.shape[1]))

        return x


@MODELS.register_module()
class ADAStyleGAN2Discriminator(StyleGAN2Discriminator):

    def __init__(self, in_size, *args, data_aug=None, **kwargs):
        """StyleGANv2 Discriminator with adaptive augmentation.

        Args:
            in_size (int): The input size of images.
            data_aug (dict, optional): Config for data
                augmentation. Defaults to None.
        """
        super().__init__(in_size, *args, **kwargs)
        self.with_ada = data_aug is not None and data_aug != dict()
        if self.with_ada:
            self.ada_aug = MODELS.build(data_aug)
            self.ada_aug.requires_grad = False
        self.log_size = int(np.log2(in_size))

    def forward(self, x):
        """Forward function."""
        if self.with_ada:
            x = self.ada_aug.aug_pipeline(x)
        return super().forward(x)


@MODELS.register_module()
class ADAAug(BaseModule):
    """Data Augmentation Module for Adaptive Discriminator augmentation.

    Args:
        aug_pipeline (dict, optional): Config for augmentation pipeline.
            Defaults to None.
        update_interval (int, optional): Interval for updating
            augmentation probability. Defaults to 4.
        augment_initial_p (float, optional): Initial augmentation
            probability. Defaults to 0..
        ada_target (float, optional): ADA target. Defaults to 0.6.
        ada_kimg (int, optional): ADA training duration. Defaults to 500.
    """

    def __init__(self,
                 aug_pipeline=None,
                 update_interval=4,
                 augment_initial_p=0.,
                 ada_target=0.6,
                 ada_kimg=500):
        super().__init__()
        aug_pipeline = dict() if aug_pipeline is None else aug_pipeline
        self.aug_pipeline = AugmentPipe(**aug_pipeline)
        self.update_interval = update_interval
        self.ada_kimg = ada_kimg
        self.ada_target = ada_target

        self.aug_pipeline.p.copy_(torch.tensor(augment_initial_p))

        # this log buffer stores two numbers: num_scalars, sum_scalars.
        self.register_buffer('log_buffer', torch.zeros((2, )))

    def update(self, iteration=0, num_batches=0):
        """Update Augment probability.

        Args:
            iteration (int, optional): Training iteration.
                Defaults to 0.
            num_batches (int, optional): The number of reals batches.
                Defaults to 0.
        """

        if (iteration + 1) % self.update_interval == 0:

            adjust_step = float(num_batches * self.update_interval) / float(
                self.ada_kimg * 1000.)

            # get the mean value as the ada heuristic
            ada_heuristic = self.log_buffer[1] / self.log_buffer[0]
            adjust = np.sign(ada_heuristic.item() -
                             self.ada_target) * adjust_step
            # update the augment p
            # Note that p may be bigger than 1.0
            self.aug_pipeline.p.copy_(
                (self.aug_pipeline.p +
                 adjust).max(constant(0, device=self.log_buffer.device)))

            self.log_buffer = self.log_buffer * 0.

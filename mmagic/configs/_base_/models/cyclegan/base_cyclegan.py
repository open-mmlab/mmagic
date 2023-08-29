# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.archs.patch_disc import PatchDiscriminator
from mmagic.models.data_preprocessors import DataPreprocessor
from mmagic.models.editors.cyclegan.cyclegan import CycleGAN
from mmagic.models.editors.cyclegan.cyclegan_generator import ResnetGenerator

_domain_a = None  # set by user
_domain_b = None  # set by user
model = dict(
    type=CycleGAN,
    data_preprocessor=dict(type=DataPreprocessor),
    generator=dict(
        type=ResnetGenerator,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        norm_cfg=dict(type=torch.nn.modules.instancenorm.InstanceNorm2d),
        use_dropout=False,
        num_blocks=9,
        padding_mode='reflect',
        init_cfg=dict(type='normal', gain=0.02)),
    discriminator=dict(
        type=PatchDiscriminator,
        in_channels=3,
        base_channels=64,
        num_conv=3,
        norm_cfg=dict(type=torch.nn.modules.instancenorm.InstanceNorm2d),
        init_cfg=dict(type='normal', gain=0.02)),
    default_domain=None,  # set by user
    reachable_domains=None,  # set by user
    related_domains=None  # set by user
)

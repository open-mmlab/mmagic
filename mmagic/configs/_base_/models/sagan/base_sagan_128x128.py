# Copyright (c) OpenMMLab. All rights reserved.
from mmagic.models import DataPreprocessor
from mmagic.models.editors import SAGAN
from mmagic.models.editors.biggan import SelfAttentionBlock
from mmagic.models.editors.sagan import ProjDiscriminator, SNGANGenerator

model = dict(
    type=SAGAN,
    num_classes=1000,
    data_preprocessor=dict(type=DataPreprocessor),
    generator=dict(
        type=SNGANGenerator,
        output_scale=128,
        base_channels=64,
        attention_cfg=dict(type=SelfAttentionBlock),
        attention_after_nth_block=4,
        with_spectral_norm=True),
    discriminator=dict(
        type=ProjDiscriminator,
        input_scale=128,
        base_channels=64,
        attention_cfg=dict(type=SelfAttentionBlock),
        attention_after_nth_block=1,
        with_spectral_norm=True),
    generator_steps=1,
    discriminator_steps=1)

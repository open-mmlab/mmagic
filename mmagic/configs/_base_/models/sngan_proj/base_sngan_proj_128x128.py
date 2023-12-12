# Copyright (c) OpenMMLab. All rights reserved.
from mmagic.models import DataPreprocessor
from mmagic.models.editors.sagan import (SAGAN, ProjDiscriminator,
                                         SNGANGenerator)

# define GAN model
model = dict(
    type=SAGAN,
    num_classes=1000,
    data_preprocessor=dict(type=DataPreprocessor),
    generator=dict(type=SNGANGenerator, output_scale=128, base_channels=64),
    discriminator=dict(
        type=ProjDiscriminator, input_scale=128, base_channels=64),
    discriminator_steps=2)

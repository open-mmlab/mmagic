# Copyright (c) OpenMMLab. All rights reserved.
from mmagic.models import DataPreprocessor
from mmagic.models.editors.sagan import (SAGAN, ProjDiscriminator,
                                         SNGANGenerator)

# define GAN model
model = dict(
    type=SAGAN,
    num_classes=10,
    data_preprocessor=dict(type=DataPreprocessor),
    generator=dict(type=SNGANGenerator, output_scale=32, base_channels=256),
    discriminator=dict(
        type=ProjDiscriminator, input_scale=32, base_channels=128),
    discriminator_steps=5)

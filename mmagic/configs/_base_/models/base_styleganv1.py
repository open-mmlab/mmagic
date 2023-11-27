# Copyright (c) OpenMMLab. All rights reserved.
from mmagic.models import DataPreprocessor
from mmagic.models.editors.stylegan1 import (StyleGAN1, StyleGAN1Discriminator,
                                             StyleGAN1Generator)

model = dict(
    type=StyleGAN1,
    data_preprocessor=dict(type=DataPreprocessor),
    style_channels=512,
    generator=dict(type=StyleGAN1Generator, out_size=None, style_channels=512),
    discriminator=dict(type=StyleGAN1Discriminator, in_size=None))

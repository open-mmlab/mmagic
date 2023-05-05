# Copyright (c) OpenMMLab. All rights reserved.
from .singan import SinGAN
from .singan_discriminator import SinGANMultiScaleDiscriminator
from .singan_generator import SinGANMultiScaleGenerator

__all__ = [
    'SinGAN', 'SinGANMultiScaleDiscriminator', 'SinGANMultiScaleGenerator'
]

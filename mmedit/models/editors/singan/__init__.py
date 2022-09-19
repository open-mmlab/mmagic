# Copyright (c) OpenMMLab. All rights reserved.
from .singan import PESinGAN, SinGAN
from .singan_discriminator import SinGANMultiScaleDiscriminator
from .singan_generator import SinGANMSGeneratorPE, SinGANMultiScaleGenerator

__all__ = [
    'SinGAN', 'SinGANMultiScaleDiscriminator', 'SinGANMultiScaleGenerator',
    'SinGANMSGeneratorPE', 'PESinGAN'
]

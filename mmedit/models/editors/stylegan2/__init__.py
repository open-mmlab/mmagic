# Copyright (c) OpenMMLab. All rights reserved.
from .stylegan2 import StyleGAN2
from .stylegan2_discriminator import (ADAAug, ADAStyleGAN2Discriminator,
                                      StyleGAN2Discriminator)
from .stylegan2_generator import StyleGAN2Generator

__all__ = [
    'StyleGAN2', 'StyleGAN2Discriminator', 'StyleGAN2Generator',
    'ADAStyleGAN2Discriminator', 'ADAAug'
]

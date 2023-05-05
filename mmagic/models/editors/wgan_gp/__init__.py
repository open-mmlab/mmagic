# Copyright (c) OpenMMLab. All rights reserved.
from .wgan_discriminator import WGANGPDiscriminator
from .wgan_generator import WGANGPGenerator
from .wgan_gp import WGANGP

__all__ = ['WGANGPDiscriminator', 'WGANGPGenerator', 'WGANGP']

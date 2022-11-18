# Copyright (c) OpenMMLab. All rights reserved.
from .dcgan import DCGAN
from .dcgan_discriminator import DCGANDiscriminator
from .dcgan_generator import DCGANGenerator

__all__ = ['DCGAN', 'DCGANDiscriminator', 'DCGANGenerator']

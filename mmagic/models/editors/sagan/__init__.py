# Copyright (c) OpenMMLab. All rights reserved.
from .sagan import SAGAN
from .sagan_discriminator import ProjDiscriminator
from .sagan_generator import SNGANGenerator
from .sagan_modules import (SNGANDiscHeadResBlock, SNGANDiscResBlock,
                            SNGANGenResBlock)

__all__ = [
    'SAGAN', 'SNGANGenerator', 'ProjDiscriminator', 'SNGANDiscHeadResBlock',
    'SNGANDiscResBlock', 'SNGANGenResBlock'
]

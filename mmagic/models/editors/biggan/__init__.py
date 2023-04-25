# Copyright (c) OpenMMLab. All rights reserved.
from .biggan import BigGAN
from .biggan_deep_discriminator import BigGANDeepDiscriminator
from .biggan_deep_generator import BigGANDeepGenerator
from .biggan_discriminator import BigGANDiscriminator
from .biggan_generator import BigGANGenerator
# from .generator_discriminator import BigGANDiscriminator, BigGANGenerator
from .biggan_modules import (BigGANConditionBN, BigGANDeepDiscResBlock,
                             BigGANDeepGenResBlock, BigGANDiscResBlock,
                             BigGANGenResBlock, SelfAttentionBlock,
                             SNConvModule)

__all__ = [
    'BigGAN',
    'BigGANGenerator',
    'BigGANDiscriminator',
    'BigGANGenResBlock',
    'BigGANConditionBN',
    'SelfAttentionBlock',
    'BigGANDiscResBlock',
    'BigGANDeepDiscriminator',
    'BigGANDeepGenerator',
    'BigGANDeepDiscResBlock',
    'BigGANDeepGenResBlock',
    'SNConvModule',
]

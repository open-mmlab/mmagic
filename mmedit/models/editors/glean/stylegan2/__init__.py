# Copyright (c) OpenMMLab. All rights reserved.
# TODO Move to common if other model use StyleGAN
from .generator_discriminator import (StyleGANv2Discriminator,
                                      StyleGANv2Generator)

__all__ = ['StyleGANv2Generator', 'StyleGANv2Discriminator']

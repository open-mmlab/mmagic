from .discriminators import (DeepFillv1Discriminators, GLDiscs, ModifiedVGG,
                             MultiLayerDiscriminator, PatchDiscriminator)
from .refiners import DeepFillRefiner, PlainRefiner
from .stylegan2 import StyleGAN2Discriminator, StyleGANv2Generator

__all__ = [
    'PlainRefiner', 'GLDiscs', 'ModifiedVGG', 'MultiLayerDiscriminator',
    'DeepFillv1Discriminators', 'DeepFillRefiner', 'PatchDiscriminator',
    'StyleGAN2Discriminator', 'StyleGANv2Generator'
]

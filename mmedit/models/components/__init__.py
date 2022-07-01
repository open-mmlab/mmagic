# Copyright (c) OpenMMLab. All rights reserved.
from .discriminators import (DeepFillv1Discriminators, GLDiscs, ModifiedVGG,
                             MultiLayerDiscriminator, PatchDiscriminator,
                             UNetDiscriminatorWithSpectralNorm)
from .refiners import DeepFillRefiner, MLPRefiner
from .stylegan2 import StyleGAN2Discriminator, StyleGANv2Generator

__all__ = [
    'GLDiscs',
    'ModifiedVGG',
    'MultiLayerDiscriminator',
    'DeepFillv1Discriminators',
    'DeepFillRefiner',
    'PatchDiscriminator',
    'StyleGAN2Discriminator',
    'StyleGANv2Generator',
    'UNetDiscriminatorWithSpectralNorm',
    'MLPRefiner',
]

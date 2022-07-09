# Copyright (c) OpenMMLab. All rights reserved.
from .discriminators import (DeepFillv1Discriminators, GLDiscs,
                             MultiLayerDiscriminator, PatchDiscriminator)
from .refiners import DeepFillRefiner

__all__ = [
    'GLDiscs',
    'MultiLayerDiscriminator',
    'DeepFillv1Discriminators',
    'DeepFillRefiner',
    'PatchDiscriminator',
]

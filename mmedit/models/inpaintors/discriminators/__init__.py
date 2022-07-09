# Copyright (c) OpenMMLab. All rights reserved.
from .deepfill_disc import DeepFillv1Discriminators
from .gl_disc import GLDiscs
from .multi_layer_disc import MultiLayerDiscriminator
from .smpatch_disc import SoftMaskPatchDiscriminator

__all__ = [
    'GLDiscs', 'MultiLayerDiscriminator', 'DeepFillv1Discriminators',
    'SoftMaskPatchDiscriminator'
]

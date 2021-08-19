# Copyright (c) OpenMMLab. All rights reserved.
from .deepfill_disc import DeepFillv1Discriminators
from .gl_disc import GLDiscs
from .light_cnn import LightCNN
from .modified_vgg import ModifiedVGG
from .multi_layer_disc import MultiLayerDiscriminator
from .patch_disc import PatchDiscriminator
from .ttsr_disc import TTSRDiscriminator

__all__ = [
    'GLDiscs', 'ModifiedVGG', 'MultiLayerDiscriminator', 'TTSRDiscriminator',
    'DeepFillv1Discriminators', 'PatchDiscriminator', 'LightCNN'
]

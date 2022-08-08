# Copyright (c) OpenMMLab. All rights reserved.
from .base_backbone import BaseBackbone
from .base_edit_model import BaseEditModel
from .base_mattor import BaseMattor
from .basic_interpolator import BasicInterpolator
from .multi_layer_disc import MultiLayerDiscriminator
from .one_stage import OneStageInpaintor
from .smpatch_disc import SoftMaskPatchDiscriminator
from .two_stage import TwoStageInpaintor

__all__ = [
    'BaseBackbone', 'BaseEditModel', 'BaseMattor', 'BasicInterpolator',
    'MultiLayerDiscriminator', 'OneStageInpaintor',
    'SoftMaskPatchDiscriminator', 'TwoStageInpaintor'
]

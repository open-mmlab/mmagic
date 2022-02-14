# Copyright (c) OpenMMLab. All rights reserved.
from .aot_inpaintor import AOTInpaintor
from .deepfillv1 import DeepFillv1Inpaintor
from .gl_inpaintor import GLInpaintor
from .one_stage import OneStageInpaintor
from .pconv_inpaintor import PConvInpaintor
from .two_stage import TwoStageInpaintor

__all__ = [
    'OneStageInpaintor', 'GLInpaintor', 'PConvInpaintor', 'TwoStageInpaintor',
    'DeepFillv1Inpaintor', 'AOTInpaintor'
]

# Copyright (c) OpenMMLab. All rights reserved.
from .average_model import ExponentialMovingAverage, RampUpEMA
from .base_conditional_gan import BaseConditionalGAN
from .base_edit_model import BaseEditModel
from .base_gan import BaseGAN
from .base_mattor import BaseMattor
from .base_translation_model import BaseTranslationModel
from .basic_interpolator import BasicInterpolator
from .one_stage import OneStageInpaintor
from .two_stage import TwoStageInpaintor

__all__ = [
    'BaseEditModel',
    'BaseGAN',
    'BaseConditionalGAN',
    'BaseMattor',
    'BasicInterpolator',
    'BaseTranslationModel',
    'OneStageInpaintor',
    'TwoStageInpaintor',
    'ExponentialMovingAverage',
    'RampUpEMA',
]

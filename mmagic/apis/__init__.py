# Copyright (c) OpenMMLab. All rights reserved.
from .inferencers.inference_functions import init_model
from .mmagic_inferencer import MMagicInferencer

__all__ = ['MMagicInferencer', 'init_model']

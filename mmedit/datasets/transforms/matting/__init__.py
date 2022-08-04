# Copyright (c) OpenMMLab. All rights reserved.
"""Data loading, transformation and augmentation for matting."""

from .alpha import GenerateSeg, GenerateSoftSeg
from .crop import CropAroundCenter, CropAroundFg, CropAroundUnknown
from .fgbg import (CompositeFg, MergeFgAndBg, PerturbBg, RandomJitter,
                   RandomLoadResizeBg)
from .trimap import (FormatTrimap, GenerateTrimap,
                     GenerateTrimapWithDistTransform, TransformTrimap)
from .utils import add_gaussian_noise, adjust_gamma, random_choose_unknown

__all__ = [
    'GenerateSeg',
    'GenerateSoftSeg',
    'CropAroundCenter',
    'CropAroundFg',
    'CropAroundUnknown',
    'CompositeFg',
    'MergeFgAndBg',
    'PerturbBg',
    'RandomJitter',
    'RandomLoadResizeBg',
    'FormatTrimap',
    'GenerateTrimap',
    'GenerateTrimapWithDistTransform',
    'TransformTrimap',
    'add_gaussian_noise',
    'adjust_gamma',
    'random_choose_unknown',
]

# Copyright (c) OpenMMLab. All rights reserved.
from .linear_lr_scheduler_with_interval import LinearLRWithInterval
from .reduce_lr_scheduler import ReduceLR

__all__ = [
    'LinearLRWithInterval',
    'ReduceLR',
]

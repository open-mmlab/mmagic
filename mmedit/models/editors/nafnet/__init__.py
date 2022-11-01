# Copyright (c) OpenMMLab. All rights reserved.
from .baseline_net import Baseline, BaselineLocal
from .nafnet_net import NAFNet, NAFNetLocal

__all__ = [
    'NAFNet',
    'NAFNetLocal',
    'Baseline',
    'BaselineLocal',
]

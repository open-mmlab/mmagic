# Copyright (c) OpenMMLab. All rights reserved.
from .nafbaseline_net import NAFBaseline, NAFBaselineLocal
from .nafnet_net import NAFNet, NAFNetLocal

__all__ = [
    'NAFNet',
    'NAFNetLocal',
    'NAFBaseline',
    'NAFBaselineLocal',
]

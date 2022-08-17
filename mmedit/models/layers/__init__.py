# Copyright (c) OpenMMLab. All rights reserved.
# To register Deconv
from .aspp import ASPP
from .conv import *  # noqa: F401, F403
from .gated_conv_module import SimpleGatedConvModule
from .linear_module import LinearModule
from .separable_conv_module import DepthwiseSeparableConvModule

__all__ = [
    'ASPP', 'DepthwiseSeparableConvModule', 'SimpleGatedConvModule',
    'LinearModule'
]

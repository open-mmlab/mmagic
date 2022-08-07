# Copyright (c) OpenMMLab. All rights reserved.
# To register Deconv
import mmedit.models.common.conv  # noqa: F401
from .aspp import ASPP
from .gca_module import GCAModule
from .separable_conv_module import DepthwiseSeparableConvModule

__all__ = ['ASPP', 'DepthwiseSeparableConvModule', 'GCAModule']

# Copyright (c) OpenMMLab. All rights reserved.
# To register Deconv
import mmedit.models.common.conv  # noqa: F401
from .aspp import ASPP
from .gated_conv_module import SimpleGatedConvModule
from .gca_module import GCAModule
from .resnet_dec import ResGCADecoder
from .separable_conv_module import DepthwiseSeparableConvModule
from .simple_encoder_decoder import SimpleEncoderDecoder
from .vgg import VGG16

__all__ = [
    'ASPP', 'DepthwiseSeparableConvModule', 'GCAModule',
    'SimpleEncoderDecoder', 'VGG16', 'ResGCADecoder', 'SimpleGatedConvModule'
]

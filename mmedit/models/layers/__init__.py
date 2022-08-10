# Copyright (c) OpenMMLab. All rights reserved.
# To register Deconv
from .aspp import ASPP
from .conv import *  # noqa: F401, F403
from .gated_conv_module import SimpleGatedConvModule
from .gca_module import GCAModule
from .linear_module import LinearModule
from .patch_disc import PatchDiscriminator
from .resnet_dec import ResGCADecoder, ResNetDec, ResShortcutDec
from .resnet_enc import ResGCAEncoder, ResNetEnc, ResShortcutEnc
from .separable_conv_module import DepthwiseSeparableConvModule
from .simple_encoder_decoder import SimpleEncoderDecoder
from .vgg import VGG16

__all__ = [
    'ASPP', 'DepthwiseSeparableConvModule', 'GCAModule',
    'SimpleEncoderDecoder', 'VGG16', 'ResNetEnc', 'ResShortcutEnc',
    'ResGCAEncoder', 'ResNetDec', 'ResShortcutDec', 'ResGCADecoder',
    'SimpleGatedConvModule', 'PatchDiscriminator', 'LinearModule'
]

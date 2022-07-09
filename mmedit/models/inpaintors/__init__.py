# Copyright (c) OpenMMLab. All rights reserved.
from .aot_inpaintor import AOTInpaintor
from .deepfillv1 import DeepFillv1Inpaintor
from .discriminators import (DeepFillv1Discriminators, GLDiscs,
                             MultiLayerDiscriminator,
                             SoftMaskPatchDiscriminator)
from .encoder_decoders import (AOTBlockNeck, AOTEncoderDecoder,
                               ContextualAttentionNeck, DeepFillDecoder,
                               DeepFillEncoder, DeepFillEncoderDecoder,
                               GLDecoder, GLDilationNeck, GLEncoder,
                               GLEncoderDecoder, PConvDecoder, PConvEncoder,
                               PConvEncoderDecoder)
from .gl_inpaintor import GLInpaintor
from .one_stage import OneStageInpaintor
from .pconv_inpaintor import PConvInpaintor
from .refiners import DeepFillRefiner
from .two_stage import TwoStageInpaintor

__all__ = [
    'GLDiscs',
    'MultiLayerDiscriminator',
    'DeepFillv1Discriminators',
    'SoftMaskPatchDiscriminator',
    'AOTInpaintor',
    'DeepFillv1Inpaintor',
    'GLInpaintor',
    'OneStageInpaintor',
    'PConvInpaintor',
    'TwoStageInpaintor',
    'AOTEncoderDecoder',
    'AOTBlockNeck',
    'GLEncoderDecoder',
    'GLEncoder',
    'GLDecoder',
    'GLDilationNeck',
    'PConvEncoderDecoder',
    'PConvEncoder',
    'PConvDecoder',
    'DeepFillEncoder',
    'ContextualAttentionNeck',
    'DeepFillDecoder',
    'DeepFillEncoderDecoder',
    'DeepFillRefiner',
]

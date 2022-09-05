# Copyright (c) OpenMMLab. All rights reserved.
from .base_backbone import BaseBackbone
from .base_edit_model import BaseEditModel
from .base_gan import BaseConditionalGAN, BaseGAN
from .base_mattor import BaseMattor
from .base_translation_model import BaseTranslationModel
from .basic_interpolator import BasicInterpolator
from .fid_inception import InceptionV3
from .multi_layer_disc import MultiLayerDiscriminator
from .one_stage import OneStageInpaintor
from .patch_disc import PatchDiscriminator
from .resnet import ResNet
from .simple_encoder_decoder import SimpleEncoderDecoder
from .smpatch_disc import SoftMaskPatchDiscriminator
from .two_stage import TwoStageInpaintor
from .vgg import VGG16

__all__ = [
    'BaseBackbone', 'BaseEditModel', 'BaseGAN', 'BaseConditionalGAN',
    'BaseMattor', 'BasicInterpolator', 'BaseTranslationModel',
    'MultiLayerDiscriminator', 'OneStageInpaintor',
    'SoftMaskPatchDiscriminator', 'TwoStageInpaintor', 'SimpleEncoderDecoder',
    'PatchDiscriminator', 'VGG16', 'ResNet', 'InceptionV3'
]

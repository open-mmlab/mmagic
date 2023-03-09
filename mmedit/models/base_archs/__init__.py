# Copyright (c) OpenMMLab. All rights reserved.
# To register Deconv
import warnings
from typing import List

from mmedit.utils import try_import
from .all_gather_layer import AllGatherLayer
from .aspp import ASPP
from .conv import *  # noqa: F401, F403
from .conv2d_gradfix import conv2d, conv_transpose2d
from .downsample import pixel_unshuffle
from .ensemble import SpatialTemporalEnsemble
from .gated_conv_module import SimpleGatedConvModule
from .img_normalize import ImgNormalize
from .linear_module import LinearModule
from .multi_layer_disc import MultiLayerDiscriminator
from .patch_disc import PatchDiscriminator
from .resnet import ResNet
from .separable_conv_module import DepthwiseSeparableConvModule
from .simple_encoder_decoder import SimpleEncoderDecoder
from .smpatch_disc import SoftMaskPatchDiscriminator
from .sr_backbone import ResidualBlockNoBN
from .upsample import PixelShufflePack
from .vgg import VGG16


def register_diffusers_models() -> List[str]:
    """Register models in ``diffusers.models`` to the ``MODELS`` registry.

    Returns:
        List[str]: A list of registered DIFFUSION_MODELS' name.
    """
    import inspect

    from mmedit.registry import MODELS

    diffusers = try_import('diffusers')
    if diffusers is None:
        warnings.warn('Diffusion Models are not registered as expect. '
                      'If you want to use diffusion models, '
                      'please install diffusers>=0.12.0.')
        return None

    DIFFUSERS_MODELS = []
    for module_name in dir(diffusers.models):
        module = getattr(diffusers.models, module_name)
        if inspect.isclass(module):
            MODELS.register_module(name=module_name, module=module)
            DIFFUSERS_MODELS.append(module_name)
    return DIFFUSERS_MODELS


REGISTERED_DIFFUSERS_MODELS = register_diffusers_models()

__all__ = [
    'ASPP', 'DepthwiseSeparableConvModule', 'SimpleGatedConvModule',
    'LinearModule', 'conv2d', 'conv_transpose2d', 'pixel_unshuffle',
    'PixelShufflePack', 'ImgNormalize', 'SpatialTemporalEnsemble',
    'SoftMaskPatchDiscriminator', 'SimpleEncoderDecoder',
    'MultiLayerDiscriminator', 'PatchDiscriminator', 'VGG16', 'ResNet',
    'AllGatherLayer', 'ResidualBlockNoBN'
]

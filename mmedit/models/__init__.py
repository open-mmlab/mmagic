# Copyright (c) OpenMMLab. All rights reserved.
from ..registry import BACKBONES, COMPONENTS, LOSSES, MODELS
from .backbones import *  # noqa: F401, F403
from .base import BaseModel
from .base_edit_model import BaseEditModel
from .builder import (build, build_backbone, build_component, build_loss,
                      build_model)
from .common import *  # noqa: F401, F403
from .components import *  # noqa: F401, F403
from .data_processor import EditDataPreprocessor
from .image_restorers import *  # noqa: F401, F403
from .inpaintors import (AOTInpaintor, DeepFillv1Inpaintor, GLInpaintor,
                         OneStageInpaintor, PConvInpaintor, TwoStageInpaintor)
from .losses import *  # noqa: F401, F403
from .mattors import DIM, GCA, BaseMattor, IndexNet
from .synthesizers import CycleGAN, Pix2Pix
from .video_interpolators import (CAIN, FLAVR, BasicInterpolator, CAINNet,
                                  FLAVRNet, TOFlowVFINet)
from .video_restorers import *  # noqa: F401, F403

__all__ = [
    'BaseEditModel',
    'EditDataPreprocessor',
    'BasicInterpolator',
    'CAIN',
    'CAINNet',
    'FLAVR',
    'FLAVRNet',
    'TOFlowVFINet',
    'AOTInpaintor',
    'BaseModel',
    'OneStageInpaintor',
    'build',
    'build_backbone',
    'build_component',
    'build_loss',
    'build_model',
    'BACKBONES',
    'COMPONENTS',
    'LOSSES',
    'BaseMattor',
    'DIM',
    'MODELS',
    'GLInpaintor',
    'PConvInpaintor',
    'GCA',
    'TwoStageInpaintor',
    'IndexNet',
    'DeepFillv1Inpaintor',
    'Pix2Pix',
    'CycleGAN',
    'BasicInterpolator',
    'CAIN',
]

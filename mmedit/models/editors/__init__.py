# Copyright (c) OpenMMLab. All rights reserved.
# from deepfillv2 import DeepFillEncoderDecoder

from .aotgan import AOTBlockNeck, AOTEncoderDecoder, AOTInpaintor
from .cain import CAIN, CAINNet
from .deepfillv1 import (ContextualAttentionModule, ContextualAttentionNeck,
                         DeepFillDecoder, DeepFillEncoder, DeepFillv1Inpaintor)
from .deepfillv2 import DeepFillEncoderDecoder, SimpleGatedConvModule
from .flavr import FLAVR, FLAVRNet
from .gca import GCA
from .global_local import (GLDecoder, GLDilationNeck, GLEncoder,
                           GLEncoderDecoder)
from .indexnet import IndexNet
from .pconv import (PConvDecoder, PConvEncoder, PConvEncoderDecoder,
                    PConvInpaintor)
from .srcnn import SRCNNNet
from .tof import TOFlowVFINet

__all__ = [
    'AOTEncoderDecoder',
    'AOTBlockNeck',
    'AOTInpaintor',
    'ContextualAttentionNeck',
    'ContextualAttentionModule',
    'CAIN',
    'CAINNet',
    'FLAVR',
    'FLAVRNet',
    'TOFlowVFINet',
    'DeepFillEncoder',
    'DeepFillEncoderDecoder',
    'DeepFillDecoder',
    'DeepFillv1Inpaintor',
    'IndexNet',
    'GCA',
    'GLEncoderDecoder',
    'GLEncoder',
    'GLDecoder',
    'GLDilationNeck',
    'PConvEncoderDecoder',
    'PConvEncoder',
    'PConvDecoder',
    'PConvInpaintor',
    'SimpleGatedConvModule',
    'SRCNNNet',
]

# Copyright (c) OpenMMLab. All rights reserved.
# from deepfillv2 import DeepFillEncoderDecoder

from .aotgan import AOTBlockNeck, AOTEncoderDecoder, AOTInpaintor
from .basicvsr import BasicVSR, BasicVSRNet
from .basicvsr_plusplus_net import BasicVSRPlusPlusNet
from .cain import CAIN, CAINNet
from .deepfillv1 import (ContextualAttentionModule, ContextualAttentionNeck,
                         DeepFillDecoder, DeepFillEncoder, DeepFillv1Inpaintor)
from .deepfillv2 import DeepFillEncoderDecoder, SimpleGatedConvModule
from .edvr import EDVR, EDVRNet
from .esrgan import RRDBNet
from .flavr import FLAVR, FLAVRNet
from .gca import GCA
from .global_local import (GLDecoder, GLDilationNeck, GLEncoder,
                           GLEncoderDecoder)
from .iconvsr import IconVSRNet
from .indexnet import IndexNet
from .pconv import (PConvDecoder, PConvEncoder, PConvEncoderDecoder,
                    PConvInpaintor)
from .real_basicvsr import RealBasicVSR, RealBasicVSRNet
from .real_esrgan import RealESRGAN, UNetDiscriminatorWithSpectralNorm
from .srcnn import SRCNNNet
from .tdan import TDAN, TDANNet
from .tof import TOFlowVFINet, TOFlowVSRNet

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
    'TOFlowVSRNet',
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
    'RRDBNet',
    'RealESRGAN',
    'UNetDiscriminatorWithSpectralNorm',
    'EDVR',
    'EDVRNet',
    'TDAN',
    'TDANNet',
    'BasicVSR',
    'BasicVSRNet',
    'BasicVSRPlusPlusNet',
    'IconVSRNet',
    'RealBasicVSR',
    'RealBasicVSRNet',
]

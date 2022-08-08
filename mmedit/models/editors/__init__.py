# Copyright (c) OpenMMLab. All rights reserved.
# from deepfillv2 import DeepFillEncoderDecoder

from .aotgan import AOTBlockNeck, AOTEncoderDecoder, AOTInpaintor
from .cain import CAIN, CAINNet
from .deepfillv1 import (ContextualAttentionModule, ContextualAttentionNeck,
                         DeepFillDecoder, DeepFillEncoder, DeepFillv1Inpaintor)
from .deepfillv2 import DeepFillEncoderDecoder, SimpleGatedConvModule
from .dic import (DIC, DICNet, FeedbackBlock, FeedbackBlockCustom,
                  FeedbackBlockHeatmapAttention, LightCNN, MaxFeature)
from .edsr import EDSRNet
from .esrgan import ESRGAN, RRDBNet
from .flavr import FLAVR, FLAVRNet
from .gca import GCA
from .glean import GLEANStyleGANv2
from .global_local import (GLDecoder, GLDilationNeck, GLEncoder,
                           GLEncoderDecoder)
from .indexnet import IndexNet
from .liif import LIIF
from .pconv import (PConvDecoder, PConvEncoder, PConvEncoderDecoder,
                    PConvInpaintor)
from .rdn import RDNNet
from .srcnn import SRCNNNet
from .srgan import SRGAN, ModifiedVGG, MSRResNet
from .tof import TOFlowVFINet, TOFlowVSRNet
from .ttsr import LTE, TTSR, SearchTransformer, TTSRDiscriminator, TTSRNet

__all__ = [
    'AOTEncoderDecoder',
    'AOTBlockNeck',
    'AOTInpaintor',
    'ContextualAttentionNeck',
    'ContextualAttentionModule',
    'CAIN',
    'CAINNet',
    'DIC',
    'DICNet',
    'LightCNN',
    'FeedbackBlock',
    'FeedbackBlockHeatmapAttention',
    'FeedbackBlockCustom',
    'MaxFeature',
    'FLAVR',
    'FLAVRNet',
    'TOFlowVFINet',
    'TOFlowVSRNet',
    'DeepFillEncoder',
    'DeepFillEncoderDecoder',
    'DeepFillDecoder',
    'DeepFillv1Inpaintor',
    'EDSRNet',
    'ESRGAN',
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
    'RRDBNet',
    'SimpleGatedConvModule',
    'SRCNNNet',
    'SRGAN',
    'MaxFeature',
    'ModifiedVGG',
    'MSRResNet',
    'RDNNet',
    'LTE',
    'TTSR',
    'TTSRNet',
    'TTSRDiscriminator',
    'TTSRNet',
    'SearchTransformer',
    'GLEANStyleGANv2',
    'LIIF',
]

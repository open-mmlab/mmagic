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
from .dic import (DIC, DICNet, FeedbackBlock, FeedbackBlockCustom,
                  FeedbackBlockHeatmapAttention, LightCNN, MaxFeature)
from .dim import DIM
from .edsr import EDSRNet
from .esrgan import ESRGAN, RRDBNet
from .flavr import FLAVR, FLAVRNet
from .gca import GCA
from .glean import GLEANStyleGANv2
from .global_local import (GLDecoder, GLDilationNeck, GLEncoder,
                           GLEncoderDecoder)
from .iconvsr import IconVSRNet
from .pconv import (PConvDecoder, PConvEncoder, PConvEncoderDecoder,
                    PConvInpaintor)
from .real_basicvsr import RealBasicVSR, RealBasicVSRNet
from .real_esrgan import RealESRGAN, UNetDiscriminatorWithSpectralNorm
from .srcnn import SRCNNNet
from .tdan import TDAN, TDANNet
from .tof import TOFlowVFINet, TOFlowVSRNet
from .indexnet import (DepthwiseIndexBlock, HolisticIndexBlock,
                       IndexedUpsample, IndexNet, IndexNetDecoder,
                       IndexNetEncoder)
from .liif import LIIF, MLPRefiner
from .pconv import (PConvDecoder, PConvEncoder, PConvEncoderDecoder,
                    PConvInpaintor)
from .plain import PlainDecoder, PlainRefiner
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
    'DIM',
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
    'DepthwiseIndexBlock',
    'HolisticIndexBlock',
    'IndexNet',
    'IndexNetEncoder',
    'IndexedUpsample',
    'IndexNetDecoder',
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
    'MLPRefiner',
    'PlainRefiner',
    'PlainDecoder',
]

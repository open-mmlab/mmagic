# Copyright (c) OpenMMLab. All rights reserved.
from .aotgan import AOTBlockNeck, AOTEncoderDecoder, AOTInpaintor
from .arcface import IDLossModel
from .basicvsr import BasicVSR, BasicVSRNet
from .basicvsr_plusplus_net import BasicVSRPlusPlusNet
from .biggan import BigGAN
from .cain import CAIN, CAINNet
from .cyclegan import CycleGAN
from .dcgan import DCGAN
from .deepfillv1 import (ContextualAttentionModule, ContextualAttentionNeck,
                         DeepFillDecoder, DeepFillEncoder, DeepFillRefiner,
                         DeepFillv1Discriminators, DeepFillv1Inpaintor)
from .deepfillv2 import DeepFillEncoderDecoder
from .dic import (DIC, DICNet, FeedbackBlock, FeedbackBlockCustom,
                  FeedbackBlockHeatmapAttention, LightCNN, MaxFeature)
from .dim import DIM
from .edsr import EDSRNet
from .edvr import EDVR, EDVRNet
from .esrgan import ESRGAN, RRDBNet
from .fba import FBADecoder, FBAResnetDilated
from .flavr import FLAVR, FLAVRNet
from .gca import GCA
from .ggan import GGAN
from .glean import GLEANStyleGANv2
from .global_local import (GLDecoder, GLDilationNeck, GLEncoder,
                           GLEncoderDecoder)
from .iconvsr import IconVSRNet
from .indexnet import (DepthwiseIndexBlock, HolisticIndexBlock,
                       IndexedUpsample, IndexNet, IndexNetDecoder,
                       IndexNetEncoder)
from .inst_colorization import InstColorization
from .liif import LIIF, MLPRefiner
from .lsgan import LSGAN
from .mspie import MSPIEStyleGAN2, PESinGAN
from .pconv import (MaskConvModule, PartialConv2d, PConvDecoder, PConvEncoder,
                    PConvEncoderDecoder, PConvInpaintor)
from .pggan import ProgressiveGrowingGAN
from .pix2pix import Pix2Pix
from .plain import PlainDecoder, PlainRefiner
from .rdn import RDNNet
from .real_basicvsr import RealBasicVSR, RealBasicVSRNet
from .real_esrgan import RealESRGAN, UNetDiscriminatorWithSpectralNorm
from .sagan import SAGAN
from .singan import SinGAN
from .srcnn import SRCNNNet
from .srgan import SRGAN, ModifiedVGG, MSRResNet
from .stylegan1 import StyleGAN1
from .stylegan2 import StyleGAN2
from .stylegan3 import StyleGAN3, StyleGAN3Generator
from .tdan import TDAN, TDANNet
from .tof import TOFlowVFINet, TOFlowVSRNet, ToFResBlock
from .ttsr import LTE, TTSR, SearchTransformer, TTSRDiscriminator, TTSRNet
from .wgan_gp import WGANGP

__all__ = [
    'AOTEncoderDecoder', 'AOTBlockNeck', 'AOTInpaintor',
    'ContextualAttentionNeck', 'ContextualAttentionModule', 'CAIN', 'CAINNet',
    'DIM', 'DIC', 'DICNet', 'LightCNN', 'FeedbackBlock',
    'FeedbackBlockHeatmapAttention', 'FeedbackBlockCustom', 'MaxFeature',
    'FLAVR', 'FLAVRNet', 'ToFResBlock', 'TOFlowVFINet', 'TOFlowVSRNet',
    'DeepFillEncoder', 'DeepFillEncoderDecoder', 'DeepFillDecoder',
    'DeepFillRefiner', 'DeepFillv1Inpaintor', 'DeepFillv1Discriminators',
    'EDSRNet', 'ESRGAN', 'DepthwiseIndexBlock', 'HolisticIndexBlock',
    'IndexNet', 'IndexNetEncoder', 'IndexedUpsample', 'IndexNetDecoder', 'GCA',
    'GLEncoderDecoder', 'GLEncoder', 'GLDecoder', 'GLDilationNeck',
    'PartialConv2d', 'PConvEncoderDecoder', 'PConvEncoder', 'PConvDecoder',
    'PConvInpaintor', 'MaskConvModule', 'RRDBNet', 'SRCNNNet', 'RRDBNet',
    'RealESRGAN', 'UNetDiscriminatorWithSpectralNorm', 'EDVR', 'EDVRNet',
    'TDAN', 'TDANNet', 'BasicVSR', 'BasicVSRNet', 'BasicVSRPlusPlusNet',
    'IconVSRNet', 'RealBasicVSR', 'RealBasicVSRNet', 'SRGAN', 'MaxFeature',
    'ModifiedVGG', 'MSRResNet', 'RDNNet', 'LTE', 'TTSR', 'TTSRNet',
    'TTSRDiscriminator', 'TTSRNet', 'SearchTransformer', 'GLEANStyleGANv2',
    'LIIF', 'MLPRefiner', 'PlainRefiner', 'PlainDecoder', 'FBAResnetDilated',
    'FBADecoder', 'WGANGP', 'CycleGAN', 'SAGAN', 'LSGAN', 'GGAN', 'Pix2Pix',
    'StyleGAN1', 'StyleGAN2', 'StyleGAN3', 'BigGAN', 'DCGAN',
    'ProgressiveGrowingGAN', 'SinGAN', 'IDLossModel', 'PESinGAN',
    'MSPIEStyleGAN2', 'StyleGAN3Generator', 'InstColorization'
]

# Copyright (c) OpenMMLab. All rights reserved.
from .dic import DIC, DICNet, LightCNN
from .edsr_net import EDSRNet
from .esrgan import ESRGAN, RRDBNet
from .glean import (GLEANStyleGANv2, StyleGANv2Discriminator,
                    StyleGANv2Generator)
from .liif import LIIF, LIIFEDSRNet, LIIFRDNNet
from .rdn_net import RDNNet
from .real_esrgan import RealESRGAN, UNetDiscriminatorWithSpectralNorm
from .srcnn_net import SRCNNNet
from .srgan import SRGAN, ModifiedVGG, MSRResNet
from .ttsr import LTE, TTSR, SearchTransformer, TTSRDiscriminator, TTSRNet

__all__ = [
    'DIC',
    'DICNet',
    'EDSRNet',
    'ESRGAN',
    'GLEANStyleGANv2',
    'LightCNN',
    'LIIF',
    'LIIFEDSRNet',
    'LIIFRDNNet',
    'LTE',
    'ModifiedVGG',
    'MSRResNet',
    'RealESRGAN',
    'SearchTransformer',
    'SRCNNNet',
    'SRGAN',
    'StyleGANv2Discriminator',
    'StyleGANv2Generator',
    'RDNNet',
    'RRDBNet',
    'TTSR',
    'TTSRDiscriminator',
    'TTSRNet',
    'UNetDiscriminatorWithSpectralNorm',
]

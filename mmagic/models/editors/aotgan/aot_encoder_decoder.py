# Copyright (c) OpenMMLab. All rights reserved.

from mmagic.registry import MODELS
from ..global_local import GLEncoderDecoder


@MODELS.register_module()
class AOTEncoderDecoder(GLEncoderDecoder):
    """Encoder-Decoder used in AOT-GAN model.

    This implementation follows:
    Aggregated Contextual Transformations for High-Resolution Image Inpainting
    The architecture of the encoder-decoder is:
    (conv2d x 3) --> (dilated conv2d x 8) --> (conv2d or deconv2d x 3).

    Args:
        encoder (dict): Config dict to encoder.
        decoder (dict): Config dict to build decoder.
        dilation_neck (dict): Config dict to build dilation neck.
    """

    def __init__(self,
                 encoder=dict(type='AOTEncoder'),
                 decoder=dict(type='AOTDecoder'),
                 dilation_neck=dict(type='AOTBlockNeck')):
        super().__init__()
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)
        self.dilation_neck = MODELS.build(dilation_neck)

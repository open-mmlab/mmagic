# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModule

from mmagic.registry import MODELS


@MODELS.register_module()
class GLEncoderDecoder(BaseModule):
    """Encoder-Decoder used in Global&Local model.

    This implementation follows:
    Globally and locally Consistent Image Completion

    The architecture of the encoder-decoder is:\
        (conv2d x 6) --> (dilated conv2d x 4) --> (conv2d or deconv2d x 7)

    Args:
        encoder (dict): Config dict to encoder.
        decoder (dict): Config dict to build decoder.
        dilation_neck (dict): Config dict to build dilation neck.
    """

    def __init__(self,
                 encoder=dict(type='GLEncoder'),
                 decoder=dict(type='GLDecoder'),
                 dilation_neck=dict(type='GLDilationNeck')):
        super().__init__()
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)
        self.dilation_neck = MODELS.build(dilation_neck)

        # support fp16
        self.fp16_enabled = False

    def forward(self, x):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """
        x = self.encoder(x)
        if isinstance(x, dict):
            x = x['out']
        x = self.dilation_neck(x)
        x = self.decoder(x)

        return x

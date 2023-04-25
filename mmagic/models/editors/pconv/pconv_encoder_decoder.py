# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModule

from mmagic.registry import MODELS


@MODELS.register_module()
class PConvEncoderDecoder(BaseModule):
    """Encoder-Decoder with partial conv module.

    Args:
        encoder (dict): Config of the encoder.
        decoder (dict): Config of the decoder.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)

        # support fp16
        self.fp16_enabled = False

    def forward(self, x, mask_in):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).
            mask_in (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """
        enc_outputs = self.encoder(x, mask_in)
        x, final_mask = self.decoder(enc_outputs)

        return x, final_mask

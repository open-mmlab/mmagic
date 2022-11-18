# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn as nn
from mmengine import MMLogger
from mmengine.runner import load_checkpoint

from mmedit.registry import BACKBONES


@BACKBONES.register_module()
class PConvEncoderDecoder(nn.Module):
    """Encoder-Decoder with partial conv module.

    Args:
        encoder (dict): Config of the encoder.
        decoder (dict): Config of the decoder.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = BACKBONES.build(encoder)
        self.decoder = BACKBONES.build(decoder)

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

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """

        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

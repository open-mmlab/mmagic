# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.models.base_models import BaseBackbone
from mmedit.registry import BACKBONES


@BACKBONES.register_module()
class PConvEncoderDecoder(BaseBackbone):
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

    def init_weights(self, pretrained=None, strict=False):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to False.
        """

        return super().init_weights(pretrained, strict)

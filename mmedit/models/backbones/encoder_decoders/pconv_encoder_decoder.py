import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmedit.models.builder import build_component
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


@BACKBONES.register_module()
class PConvEncoderDecoder(nn.Module):
    """Encoder-Decoder with partial conv module.

    Args:
        encoder (dict): Config of the encoder.
        decoder (dict): Config of the decoder.
    """

    def __init__(self, encoder, decoder):
        super(PConvEncoderDecoder, self).__init__()
        self.encoder = build_component(encoder)
        self.decoder = build_component(decoder)

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

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            # Here, we just use the default initialization in `ConvModule`.
            pass
        else:
            raise TypeError('pretrained must be a str or None')

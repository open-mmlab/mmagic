import torch.nn as nn

from mmedit.models.builder import build_component
from mmedit.models.registry import BACKBONES


@BACKBONES.register_module()
class SimpleEncoderDecoder(nn.Module):
    """Simple encoder-decoder model from matting.

    Args:
        encoder (dict): Config of the encoder.
        decoder (dict): Config of the decoder.
    """

    def __init__(self, encoder, decoder):
        super(SimpleEncoderDecoder, self).__init__()

        self.encoder = build_component(encoder)
        decoder['in_channels'] = self.encoder.out_channels
        self.decoder = build_component(decoder)

    def init_weights(self, pretrained=None):
        self.encoder.init_weights(pretrained)
        self.decoder.init_weights()

    def forward(self, *args, **kwargs):
        """Forward function.

        Returns:
            Tensor: The output tensor of the decoder.
        """
        out = self.encoder(*args, **kwargs)
        out = self.decoder(out)
        return out

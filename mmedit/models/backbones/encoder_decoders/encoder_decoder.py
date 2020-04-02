from mmedit.models.registry import BACKBONES

from .simple_encoder_decoder import SimpleEncoderDecoder


@BACKBONES.register_module
class EncoderDecoder(SimpleEncoderDecoder):
    """Encoder-decoder model with shortcut connection."""

    def forward(self, x):
        out, mid_feat = self.encoder(x)
        out = self.decoder(out, mid_feat)
        return out

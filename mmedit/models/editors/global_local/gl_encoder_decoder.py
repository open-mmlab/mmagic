# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine import MMLogger
from mmengine.runner import load_checkpoint

from mmedit.registry import BACKBONES, COMPONENTS


@BACKBONES.register_module()
class GLEncoderDecoder(nn.Module):
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
        self.encoder = COMPONENTS.build(encoder)
        self.decoder = COMPONENTS.build(decoder)
        self.dilation_neck = COMPONENTS.build(dilation_neck)

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

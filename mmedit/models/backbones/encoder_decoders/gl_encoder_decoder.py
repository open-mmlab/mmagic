# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import auto_fp16, load_checkpoint

from mmedit.models.builder import build_component
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


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
        self.encoder = build_component(encoder)
        self.decoder = build_component(decoder)
        self.dilation_neck = build_component(dilation_neck)

        # support fp16
        self.fp16_enabled = False

    @auto_fp16()
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

# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine import MMLogger
from mmengine.runner import load_checkpoint

from mmedit.registry import BACKBONES


@BACKBONES.register_module()
class BaseBackbone(nn.Module):
    """Base backbone for image and video editing."""

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

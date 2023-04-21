# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmagic.models.archs import MultiLayerDiscriminator
from mmagic.registry import MODELS


@MODELS.register_module()
class GLDiscs(BaseModule):
    """Discriminators in Global&Local.

    This discriminator contains a local discriminator and a global
    discriminator as described in the original paper:
    Globally and locally Consistent Image Completion

    Args:
        global_disc_cfg (dict): Config dict to build global discriminator.
        local_disc_cfg (dict): Config dict to build local discriminator.
    """

    def __init__(self, global_disc_cfg, local_disc_cfg):
        super().__init__()
        self.global_disc = MultiLayerDiscriminator(**global_disc_cfg)
        self.local_disc = MultiLayerDiscriminator(**local_disc_cfg)

        self.fc = nn.Linear(2048, 1, bias=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (tuple[torch.Tensor]): Contains global image and the local image
                patch.

        Returns:
            tuple[torch.Tensor]: Contains the prediction from discriminators \
                in global image and local image patch.
        """
        g_img, l_img = x
        g_pred = self.global_disc(g_img)
        l_pred = self.local_disc(l_img)

        pred = self.fc(torch.cat([g_pred, l_pred], dim=1))

        return pred

    def init_weights(self):
        """Init weights for models."""

        for m in self.modules():
            # Here, we only initialize the module with fc layer since the
            # conv and norm layers has been initialized in `ConvModule`.
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        self._is_init = True

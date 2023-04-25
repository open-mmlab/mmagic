# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.model.weight_init import normal_init

from mmagic.registry import MODELS


@MODELS.register_module()
class DeepFillv1Discriminators(BaseModule):
    """Discriminators used in DeepFillv1 model.

    In DeepFillv1 model, the discriminators are independent without any
    concatenation like Global&Local model. Thus, we call this model
    `DeepFillv1Discriminators`. There exist a global discriminator and a local
    discriminator with global and local input respectively.

    The details can be found in:
    Generative Image Inpainting with Contextual Attention.

    Args:
        global_disc_cfg (dict): Config dict for global discriminator.
        local_disc_cfg (dict): Config dict for local discriminator.
    """

    def __init__(self, global_disc_cfg, local_disc_cfg):
        super().__init__()
        self.global_disc = MODELS.build(global_disc_cfg)
        self.local_disc = MODELS.build(local_disc_cfg)

    def forward(self, x):
        """Forward function.

        Args:
            x (tuple[torch.Tensor]): Contains global image and the local image
                patch.

        Returns:
            tuple[torch.Tensor]: Contains the prediction from discriminators \
                in global image and local image patch.
        """
        global_img, local_img = x

        global_pred = self.global_disc(global_img)
        local_pred = self.local_disc(local_img)

        return global_pred, local_pred

    def init_weights(self):
        """Init weights for models."""

        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, 0, std=0.02)
            elif isinstance(m, nn.Conv2d):
                normal_init(m, 0.0, std=0.02)

        self._is_init = True

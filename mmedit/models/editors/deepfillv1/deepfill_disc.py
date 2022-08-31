# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine import MMLogger
from mmengine.model.weight_init import normal_init
from mmengine.runner import load_checkpoint

from mmedit.registry import COMPONENTS


@COMPONENTS.register_module()
class DeepFillv1Discriminators(nn.Module):
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
        self.global_disc = COMPONENTS.build(global_disc_cfg)
        self.local_disc = COMPONENTS.build(local_disc_cfg)

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

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    normal_init(m, 0, std=0.02)
                elif isinstance(m, nn.Conv2d):
                    normal_init(m, 0.0, std=0.02)
        else:
            raise TypeError('pretrained must be a str or None but got'
                            f'{type(pretrained)} instead.')

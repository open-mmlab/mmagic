# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.models.base_models import BaseEditModel
from mmedit.registry import MODELS


@MODELS.register_module()
class AirNetRestorer(BaseEditModel):
    """AirNet restorer model for single image restoration for unknown tasks.

    Ref: "All-In-One Image Restoration for Unknown Corruption"

    Note: This class mainly handle the problem that:
        AirNet training will contain two losses:
            reconstruction loss and contrastive loss

    Args:
        generator (dict): Config for the generator.
        pixel_loss (dict): Config for the pixel loss.
        pretrained (str): Path for pretrained model. Default: None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
    """

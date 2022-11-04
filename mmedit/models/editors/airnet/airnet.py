# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule

from mmedit.registry import MODELS


@MODELS.register_module()
class AirNet(BaseModule):
    """AirNet in All-In-One Image Restoration for Unknown Corruption.

    Args:
        encoder_cfg (dict): Config for encoder.
        restorer_cfg (dict): Config for restorer.
    """

    def __init__(self, encoder_cfg=dict(), restorer_cfg=dict()):
        super().__init__()

        # Encoder
        self.E = MODELS.build(encoder_cfg)

        # Restorer
        self.R = MODELS.build(restorer_cfg)

    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            x_query = x_key = inputs
        elif isinstance(inputs, dict):
            x_query = inputs['degrad_patch_1']
            x_key = inputs['degrad_patch_2']
        else:
            raise ValueError(
                'Inputs should be a dict for training or a tensor for testing')

        outputs = self.E(x_query, x_key)

        restored = self.R(x_query, outputs['inter'])
        outputs['restored'] = restored

        return outputs

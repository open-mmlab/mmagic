# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn

from mmagic.registry import MODELS


@MODELS.register_module()
class FaceIdLoss(nn.Module):
    """Face similarity loss. Generally this loss is used to keep the id
    consistency of the input face image and output face image.

    In this loss, we may need to provide ``gt``, ``pred`` and ``x``. Thus,
    an example of the ``data_info`` is:

    .. code-block:: python
        :linenos:

        data_info = dict(
            gt='real_imgs',
            pred='fake_imgs')

    Then, the module will automatically construct this mapping from the input
    data dictionary.

    Args:
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary. If ``None``, this module will
            directly pass the input data to the loss function.
            Defaults to None.
        facenet (dict, optional): Config dict for facenet. Defaults to
            dict(type='ArcFace', ir_se50_weights=None, device='cuda').
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_id'.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 data_info: Optional[dict] = None,
                 facenet: dict = dict(type='ArcFace', ir_se50_weights=None),
                 loss_name: str = 'loss_id') -> None:

        super(FaceIdLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_info = data_info
        self.net = MODELS.build(facenet)
        self._loss_name = loss_name

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Forward function."""

        # NOTE: only return the loss term
        return self.net(pred, gt)[0] * self.loss_weight

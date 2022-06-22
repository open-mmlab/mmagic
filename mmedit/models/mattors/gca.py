# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch

from mmedit.registry import MODELS
from .base_mattor import BaseMattor
from .utils import get_unknown_tensor


@MODELS.register_module()
class GCA(BaseMattor):
    """Guided Contextual Attention image matting model.

    https://arxiv.org/abs/2001.04069

    Args:
        backbone (dict): Config of backbone.
        train_cfg (dict): Config of training. In ``train_cfg``,
            ``train_backbone`` should be specified. If the model has a refiner,
            ``train_refiner`` should be specified.
        test_cfg (dict): Config of testing. In ``test_cfg``, If the model has a
            refiner, ``train_refiner`` should be specified.
        pretrained (str): Path of the pretrained model.
        loss_alpha (dict): Config of the alpha prediction loss. Default: None.
    """

    def __init__(self,
                 data_preprocessor,
                 backbone,
                 loss_alpha=None,
                 init_cfg: Optional[dict] = None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(
            backbone=backbone,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

        self.loss_alpha = MODELS.build(loss_alpha)

    def _forward(self, inputs):
        raw_alpha = self.backbone(inputs)
        pred_alpha = (raw_alpha.tanh() + 1.0) / 2.0
        return pred_alpha

    def _forward_test(self, inputs):
        return self._forward(inputs)

    def _forward_train(self, inputs, data_samples):
        """Forward function for training GCA model.

        Args:
            merged (Tensor): with shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            trimap (Tensor): with shape (N, C', H, W). Tensor of trimap. C'
                might be 1 or 3.
            meta (list[dict]): Meta data about the current data batch.
            alpha (Tensor): with shape (N, 1, H, W). Tensor of alpha.

        Returns:
            dict: Contains the loss items and batch information.
        """
        trimap = inputs[:, 3:, :, :]
        gt_alpha = torch.stack(tuple(ds.gt_alpha.data for ds in data_samples))
        pred_alpha = self._forward(inputs)

        # FormatTrimap(to_onehot=False) will change unknown_value to 1
        # FormatTrimap(to_onehot=True) will shift to 3 dim,
        # get_unknown_tensor can handle that directly without knowing
        # unknown_value.
        weight = get_unknown_tensor(trimap, unknown_value=1)

        losses = {'loss': self.loss_alpha(pred_alpha, gt_alpha, weight)}
        return losses

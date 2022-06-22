# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.registry import MODELS
from .base_mattor import BaseMattor
from .utils import get_unknown_tensor


@MODELS.register_module()
class IndexNet(BaseMattor):
    """IndexNet matting model.

    This implementation follows:
    Indices Matter: Learning to Index for Deep Image Matting

    Args:
        backbone (dict): Config of backbone.
        train_cfg (dict): Config of training. In 'train_cfg', 'train_backbone'
            should be specified.
        test_cfg (dict): Config of testing.
        pretrained (str): path of pretrained model.
        loss_alpha (dict): Config of the alpha prediction loss. Default: None.
        loss_comp (dict): Config of the composition loss. Default: None.
    """

    def __init__(self,
                 data_preprocessor,
                 backbone,
                 loss_alpha=None,
                 loss_comp=None,
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(
            backbone=backbone,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

        self.loss_alpha = (
            MODELS.build(loss_alpha) if loss_alpha is not None else None)
        self.loss_comp = (
            MODELS.build(loss_comp) if loss_comp is not None else None)

    def _forward(self, inputs):
        pred_alpha = self.backbone(inputs)
        return pred_alpha

    def _forward_test(self, inputs):
        return self._forward(inputs)

    def _forward_train(self, inputs, data_samples):
        """Forward function for training IndexNet model.

        Args:
            merged (Tensor): Input images tensor with shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            trimap (Tensor): Tensor of trimap with shape (N, 1, H, W).
            meta (list[dict]): Meta data about the current data batch.
            alpha (Tensor): Tensor of alpha with shape (N, 1, H, W).
            ori_merged (Tensor): Tensor of origin merged images (not
                normalized) with shape (N, C, H, W).
            fg (Tensor): Tensor of foreground with shape (N, C, H, W).
            bg (Tensor): Tensor of background with shape (N, C, H, W).

        Returns:
            dict: Contains the loss items and batch information.
        """
        trimap = inputs[:, 3:, :, :]
        gt_alpha = torch.stack(tuple(ds.gt_alpha.data for ds in data_samples))
        gt_fg = torch.stack(tuple(ds.gt_fg.data for ds in data_samples))
        gt_bg = torch.stack(tuple(ds.gt_bg.data for ds in data_samples))
        gt_merged = torch.stack(
            tuple(ds.gt_merged.data for ds in data_samples))

        pred_alpha = self.backbone(inputs)

        weight = get_unknown_tensor(trimap, unknown_value=128 / 255)

        losses = dict()

        if self.loss_alpha is not None:
            losses['loss_alpha'] = self.loss_alpha(pred_alpha, gt_alpha,
                                                   weight)
        if self.loss_comp is not None:
            losses['loss_comp'] = self.loss_comp(pred_alpha, gt_fg, gt_bg,
                                                 gt_merged, weight)

        return losses

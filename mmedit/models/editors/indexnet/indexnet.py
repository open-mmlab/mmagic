# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.base_models import BaseMattor
from mmedit.models.utils import get_unknown_tensor
from mmedit.registry import MODELS


@MODELS.register_module()
class IndexNet(BaseMattor):
    """IndexNet matting model.

    This implementation follows:
    Indices Matter: Learning to Index for Deep Image Matting

    Args:
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        backbone (dict): Config of backbone.
        train_cfg (dict): Config of training. In 'train_cfg', 'train_backbone'
            should be specified.
        test_cfg (dict): Config of testing.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
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
        """Forward function.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        pred_alpha = self.backbone(inputs)
        return pred_alpha

    def _forward_test(self, inputs):
        """Forward function for testing IndexNet model.

        Args:
            inputs (torch.Tensor): batch input tensor.

        Returns:
            Tensor: Output tensor of model.
        """
        return self._forward(inputs)

    def _forward_train(self, inputs, data_samples):
        """Forward function for training IndexNet model.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement]): data samples collated by
                :attr:`data_preprocessor`.

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

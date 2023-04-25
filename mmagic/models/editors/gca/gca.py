# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmagic.models.base_models import BaseMattor
from mmagic.models.utils import get_unknown_tensor
from mmagic.registry import MODELS


@MODELS.register_module()
class GCA(BaseMattor):
    """Guided Contextual Attention image matting model.

    https://arxiv.org/abs/2001.04069

    Args:
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        backbone (dict): Config of backbone.
        loss_alpha (dict): Config of the alpha prediction loss. Default: None.
        init_cfg (dict, optional): Initialization config dict. Default: None.
        train_cfg (dict): Config of training. In ``train_cfg``,
            ``train_backbone`` should be specified. If the model has a refiner,
            ``train_refiner`` should be specified.
        test_cfg (dict): Config of testing. In ``test_cfg``, If the model has a
            refiner, ``train_refiner`` should be specified.
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
        """Forward function.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        raw_alpha = self.backbone(inputs)
        pred_alpha = (raw_alpha.tanh() + 1.0) / 2.0
        return pred_alpha

    def _forward_test(self, inputs):
        """Forward function for testing GCA model.

        Args:
            inputs (torch.Tensor): batch input tensor.

        Returns:
            Tensor: Output tensor of model.
        """
        return self._forward(inputs)

    def _forward_train(self, inputs, data_samples):
        """Forward function for training GCA model.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement]): data samples collated by
                :attr:`data_preprocessor`.

        Returns:
            dict: Contains the loss items and batch information.
        """
        trimap = inputs[:, 3:, :, :]
        gt_alpha = data_samples.gt_alpha
        pred_alpha = self._forward(inputs)

        # FormatTrimap(to_onehot=False) will change unknown_value to 1
        # FormatTrimap(to_onehot=True) will shift to 3 dim,
        # get_unknown_tensor can handle that directly without knowing
        # unknown_value.
        weight = get_unknown_tensor(trimap, unknown_value=1)

        losses = {'loss': self.loss_alpha(pred_alpha, gt_alpha, weight)}
        return losses

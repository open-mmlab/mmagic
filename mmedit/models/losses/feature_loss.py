# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmedit.models.components.discriminators import LightCNN
from mmedit.utils import get_root_logger
from ..registry import LOSSES


class LightCNNFeature(nn.Module):
    """Feature of LightCNN.

    It is used to train DICGAN.
    """

    def __init__(self) -> None:
        super().__init__()

        model = LightCNN(3)
        self.features = nn.Sequential(*list(model.features.children()))
        self.features.requires_grad_(False)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Forward results.
        """

        return self.features(x)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


@LOSSES.register_module()
class LightCNNFeatureLoss(nn.Module):
    """Feature loss of DICGAN, based on LightCNN.

    Args:
        pretrained (str): Path for pretrained weights.
        loss_weight (float): Loss weight. Default: 1.0.
        criterion (str): Criterion type. Options are 'l1' and 'mse'.
            Default: 'l1'.
    """

    def __init__(self, pretrained, loss_weight=1.0, criterion='l1'):
        super().__init__()
        self.model = LightCNNFeature()
        assert isinstance(pretrained, str), 'Model must be pretrained'
        self.model.init_weights(pretrained)
        self.model.eval()
        self.loss_weight = loss_weight
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError("'criterion' should be 'l1' or 'mse', "
                             f'but got {criterion}')

    def forward(self, pred, gt):
        """Forward function.

        Args:
            pred (Tensor): Predicted tensor.
            gt (Tensor): GT tensor.

        Returns:
            Tensor: Forward results.
        """

        assert self.model.training is False
        pred_feature = self.model(pred)
        gt_feature = self.model(gt).detach()
        feature_loss = self.criterion(pred_feature, gt_feature)

        return feature_loss * self.loss_weight

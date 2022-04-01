# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .pixelwise_loss import l1_loss

_reduction_modes = ['none', 'mean', 'sum']


@LOSSES.register_module()
class GradientLoss(nn.Module):
    """Gradient loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        kx = torch.Tensor([[1, 0, -1], [2, 0, -2],
                           [1, 0, -1]]).view(1, 1, 3, 3).to(target)
        ky = torch.Tensor([[1, 2, 1], [0, 0, 0],
                           [-1, -2, -1]]).view(1, 1, 3, 3).to(target)

        pred_grad_x = F.conv2d(pred, kx, padding=1)
        pred_grad_y = F.conv2d(pred, ky, padding=1)
        target_grad_x = F.conv2d(target, kx, padding=1)
        target_grad_y = F.conv2d(target, ky, padding=1)

        loss = (
            l1_loss(
                pred_grad_x, target_grad_x, weight, reduction=self.reduction) +
            l1_loss(
                pred_grad_y, target_grad_y, weight, reduction=self.reduction))
        return loss * self.loss_weight

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn

from mmagic.registry import MODELS
from .pixelwise_loss import charbonnier_loss, l1_loss, mse_loss

_reduction_modes = ['none', 'mean', 'sum']


@MODELS.register_module()
class L1CompositionLoss(nn.Module):
    """L1 composition loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean',
                 sample_wise: bool = False) -> None:
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self,
                pred_alpha: torch.Tensor,
                fg: torch.Tensor,
                bg: torch.Tensor,
                ori_merged: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Args:
            pred_alpha (Tensor): of shape (N, 1, H, W). Predicted alpha matte.
            fg (Tensor): of shape (N, 3, H, W). Tensor of foreground object.
            bg (Tensor): of shape (N, 3, H, W). Tensor of background object.
            ori_merged (Tensor): of shape (N, 3, H, W). Tensor of origin merged
                image before normalized by ImageNet mean and std.
            weight (Tensor, optional): of shape (N, 1, H, W). It is an
                indicating matrix: weight[trimap == 128] = 1. Default: None.
        """
        pred_merged = pred_alpha * fg + (1. - pred_alpha) * bg
        if weight is not None:
            weight = weight.expand(-1, 3, -1, -1)
        return self.loss_weight * l1_loss(
            pred_merged,
            ori_merged,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@MODELS.register_module()
class MSECompositionLoss(nn.Module):
    """MSE (L2) composition loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean',
                 sample_wise: bool = False) -> None:
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self,
                pred_alpha: torch.Tensor,
                fg: torch.Tensor,
                bg: torch.Tensor,
                ori_merged: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Args:
            pred_alpha (Tensor): of shape (N, 1, H, W). Predicted alpha matte.
            fg (Tensor): of shape (N, 3, H, W). Tensor of foreground object.
            bg (Tensor): of shape (N, 3, H, W). Tensor of background object.
            ori_merged (Tensor): of shape (N, 3, H, W). Tensor of origin merged
                image before normalized by ImageNet mean and std.
            weight (Tensor, optional): of shape (N, 1, H, W). It is an
                indicating matrix: weight[trimap == 128] = 1. Default: None.
        """
        pred_merged = pred_alpha * fg + (1. - pred_alpha) * bg
        if weight is not None:
            weight = weight.expand(-1, 3, -1, -1)
        return self.loss_weight * mse_loss(
            pred_merged,
            ori_merged,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@MODELS.register_module()
class CharbonnierCompLoss(nn.Module):
    """Charbonnier composition loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean',
                 sample_wise: bool = False,
                 eps: bool = 1e-12) -> None:
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def forward(self,
                pred_alpha: torch.Tensor,
                fg: torch.Tensor,
                bg: torch.Tensor,
                ori_merged: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Args:
            pred_alpha (Tensor): of shape (N, 1, H, W). Predicted alpha matte.
            fg (Tensor): of shape (N, 3, H, W). Tensor of foreground object.
            bg (Tensor): of shape (N, 3, H, W). Tensor of background object.
            ori_merged (Tensor): of shape (N, 3, H, W). Tensor of origin merged
                image before normalized by ImageNet mean and std.
            weight (Tensor, optional): of shape (N, 1, H, W). It is an
                indicating matrix: weight[trimap == 128] = 1. Default: None.
        """
        pred_merged = pred_alpha * fg + (1. - pred_alpha) * bg
        if weight is not None:
            weight = weight.expand(-1, 3, -1, -1)
        return self.loss_weight * charbonnier_loss(
            pred_merged,
            ori_merged,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            sample_wise=self.sample_wise)

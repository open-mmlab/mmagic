import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module()
class ExclLoss(nn.Module):

    def __init__(self, channels=3, loss_weight=1.0):
        super(ExclLoss, self).__init__()
        self.channels = channels
        self.loss_weight = loss_weight

    def forward(self, f_pred, b_pred):
        kx = torch.Tensor([[1, 0, -1], [2, 0, -2],
                           [1, 0, -1]]).view(1, 1, 3, 3).to(b_pred)
        kx = kx.repeat(1, self.channels, 1, 1)
        ky = torch.Tensor([[1, 2, 1], [0, 0, 0],
                           [-1, -2, -1]]).view(1, 1, 3, 3).to(b_pred)
        ky = ky.repeat(1, self.channels, 1, 1)
        f_pred_grad_x = F.conv2d(f_pred, kx, padding=1)
        f_pred_grad_y = F.conv2d(f_pred, ky, padding=1)
        b_pred_grad_x = F.conv2d(b_pred, kx, padding=1)
        b_pred_grad_y = F.conv2d(b_pred, ky, padding=1)

        grad_b_x = abs(b_pred_grad_x)
        grad_f_x = abs(f_pred_grad_x)
        grad_b_y = abs(b_pred_grad_y)
        grad_f_y = abs(f_pred_grad_y)

        loss_excl = grad_b_x.mul(grad_f_x).sum() + grad_b_y.mul(grad_f_y).sum()

        return loss_excl * self.loss_weight

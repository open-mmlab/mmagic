# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmedit.registry import LOSSES


@LOSSES.register_module()
class HuberLoss(nn.Module):

    def __init__(self, delta=.01):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, in0, in1):
        mask = torch.zeros_like(in0)
        mann = torch.abs(in0 - in1)
        eucl = .5 * (mann**2)
        mask[...] = mann < self.delta

        loss = eucl * mask / self.delta + (mann - .5 * self.delta) * (1 - mask)
        return torch.sum(loss, dim=1, keepdim=True)

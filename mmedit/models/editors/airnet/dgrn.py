# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine.model import BaseModule

from mmedit.registry import MODELS
from .dgrn_modules import DGG, default_conv


@MODELS.register_module()
class DGRN(BaseModule):
    """Degradation-Guided Restoration Network.

    Args:
        n_groups (int): Number of DGG groups.
            Defaults to 5.
        n_blocks (int): Number of DGB groups in one DGG.
            Defaults to 5.
        n_feats (int): Dimension of features.
            Defaults to 64.
        kernel_size (int): Kernel size for DCN_layers.
            Defaults to 3:
        conv (nn.Module): Convolution module.
            Defaults to default_conv
    """

    def __init__(self,
                 n_groups=5,
                 n_blocks=5,
                 n_feats=64,
                 kernel_size=3,
                 conv=default_conv):

        super(DGRN, self).__init__()

        self.n_groups = n_groups

        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # body
        modules_body = [
            DGG(default_conv, n_feats, kernel_size, n_blocks)
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, inter):
        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            res = self.body[i](res, inter)
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        return x

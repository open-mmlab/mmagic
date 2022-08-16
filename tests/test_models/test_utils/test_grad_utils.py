# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.utils import set_requires_grad


def test_set_requires_grad():
    model = torch.nn.Conv2d(1, 3, 1, 1)
    set_requires_grad(model, False)
    for param in model.parameters():
        assert not param.requires_grad

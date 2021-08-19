# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.common import ImgNormalize


def test_normalize_layer():
    rgb_mean = (1, 2, 3)
    rgb_std = (1, 0.5, 0.25)
    layer = ImgNormalize(1, rgb_mean, rgb_std)
    x = torch.randn((2, 3, 64, 64))
    y = layer(x)
    x = x.permute((1, 0, 2, 3)).reshape((3, -1))
    y = y.permute((1, 0, 2, 3)).reshape((3, -1))
    rgb_mean = torch.tensor(rgb_mean)
    rgb_std = torch.tensor(rgb_std)
    mean_x = x.mean(dim=1)
    mean_y = y.mean(dim=1)
    std_x = x.std(dim=1)
    std_y = y.std(dim=1)
    assert sum(torch.div(std_x, std_y) - rgb_std) < 1e-5
    assert sum(torch.div(mean_x - rgb_mean, rgb_std) - mean_y) < 1e-5

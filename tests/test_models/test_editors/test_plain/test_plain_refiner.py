# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors import PlainRefiner


def test_plain_refiner():
    plain_refiner = PlainRefiner()
    plain_refiner.init_weights()
    input = torch.rand((2, 4, 128, 128))
    raw_alpha = torch.rand((2, 1, 128, 128))
    output = plain_refiner(input, raw_alpha)
    assert output.detach().numpy().shape == (2, 1, 128, 128)

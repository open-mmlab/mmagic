# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors.ddpm.attention import (ApproximateGELU,
                                                  CrossAttention)


def test_ApproximateGELU():
    input = torch.rand((16, 16))
    gelu = ApproximateGELU(16, 24)
    output = gelu.forward(input)
    assert output.shape == (16, 24)


def test_crossattention():
    input = torch.rand((2, 64, 64))
    crossattention = CrossAttention(64)
    crossattention._slice_size = 2
    output = crossattention.forward(input)
    assert output.shape == (2, 64, 64)


if __name__ == '__main__':
    test_crossattention()

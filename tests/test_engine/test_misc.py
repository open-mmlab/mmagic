# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.engine import tensor2img


def test_tensor2img():
    input = torch.rand(1, 3, 8, 8)
    result = tensor2img(input)
    assert result.shape == (8, 8, 3)

    input = torch.rand(1, 1, 8, 8)
    result = tensor2img(input)
    assert result.shape == (8, 8)

    input = torch.rand(4, 3, 8, 8)
    result = tensor2img(input)
    assert result.shape == (22, 22, 3)

    input = [torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)]
    result = tensor2img(input)
    assert len(result) == len(input)
    for r in result:
        assert r.shape == (8, 8, 3)

# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmedit.utils import tensor2img, to_numpy


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

    with pytest.raises(TypeError):
        tensor2img('wrong type')

    with pytest.raises(ValueError):
        tensor2img([torch.randn(2, 3, 4, 4, 4) for _ in range(2)])


def test_to_numpy():
    input = torch.rand(1, 3, 8, 8)
    output = to_numpy(input)
    assert isinstance(output, np.ndarray)

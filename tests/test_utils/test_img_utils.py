# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmagic.utils import (all_to_tensor, can_convert_to_image, tensor2img,
                          to_numpy)


def test_all_to_tensor():

    data = [np.random.rand(64, 64, 3), np.random.rand(64, 64, 3)]
    tensor = all_to_tensor(data)
    assert tensor.shape == torch.Size([2, 3, 64, 64])

    data = np.random.rand(64, 64, 3)
    tensor = all_to_tensor(data)
    assert tensor.shape == torch.Size([3, 64, 64])

    data = 1
    tensor = all_to_tensor(data)
    assert tensor == torch.tensor(1)


def test_can_convert_to_image():
    values = [
        np.random.rand(64, 64, 3),
        [np.random.rand(64, 61, 3),
         np.random.rand(64, 61, 3)], (64, 64), 'b'
    ]
    targets = [True, True, False, False]
    for val, tar in zip(values, targets):
        assert can_convert_to_image(val) == tar


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


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

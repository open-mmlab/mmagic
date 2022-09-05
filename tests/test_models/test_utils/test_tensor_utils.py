# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.utils.tensor_utils import get_unknown_tensor


def test_tensor_utils():
    input = torch.rand((2, 3, 128, 128))
    output = get_unknown_tensor(input)
    assert output.detach().numpy().shape == (2, 1, 128, 128)
    input = torch.rand((2, 1, 128, 128))
    output = get_unknown_tensor(input)
    assert output.detach().numpy().shape == (2, 1, 128, 128)

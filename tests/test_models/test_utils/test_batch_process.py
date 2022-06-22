# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models.utils import split_batch, stack_batch


def test_batch_process():

    tensor_list = [torch.ones((1, 1, 3, 4)), torch.ones((1, 1, 7, 7))]

    padded_tensor, padded_sizes = stack_batch(
        tensor_list, pad_size_divisor=16, pad_args=dict(value=0))
    assert padded_tensor.shape == (2, 1, 1, 16, 16)
    assert padded_tensor[0].sum() == 12
    assert (padded_tensor[0, ..., :3, :4] - tensor_list[0]).sum() < 1e-8
    assert padded_tensor[1].sum() == 49
    assert (padded_tensor[1, ..., :7, :7] - tensor_list[1]).sum() < 1e-8
    assert (padded_sizes -
            torch.tensor([[0., 0., 13., 12.], [0., 0., 9., 9.]])).sum() < 1e-8

    free_tensors = split_batch(padded_tensor, padded_sizes)
    for tensor, free in zip(tensor_list, free_tensors):
        assert (tensor - free).sum() < 1e-8

    tensor_list = [torch.ones((1, 1, 16, 16)), torch.ones((1, 1, 16, 16))]

    padded_tensor, padded_sizes = stack_batch(
        tensor_list, pad_size_divisor=16, pad_args=dict(value=0))
    assert isinstance(padded_sizes, torch.Tensor)
    assert isinstance(padded_tensor, torch.Tensor)
    assert padded_sizes.sum() == 0
    for tensor, free in zip(tensor_list, padded_tensor):
        assert (tensor - free).sum() < 1e-8

    with pytest.raises(AssertionError):
        stack_batch(tensor_list[0])
    with pytest.raises(AssertionError):
        stack_batch([])
    with pytest.raises(AssertionError):
        stack_batch([torch.ones((1, 1, 3)), torch.ones((1, 1, 7, 7))])

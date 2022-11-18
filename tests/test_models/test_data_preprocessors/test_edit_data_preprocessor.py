# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import pytest
import torch

from mmedit.models.data_preprocessors import (EditDataPreprocessor,
                                              split_batch, stack_batch)
from mmedit.structures import EditDataSample, PixelData


def test_edit_data_preprocessor():

    # test frames
    processor = EditDataPreprocessor(
        input_view=(1, -1, 1, 1), output_view=(1, -1, 1, 1))

    # prepare data
    inputs = torch.rand(1, 2, 3, 20, 20)
    target = torch.rand(3, 20, 20)
    data_sample = EditDataSample(gt_img=PixelData(data=target))
    data = dict(inputs=inputs, data_samples=[data_sample])

    # process
    output_data = processor(data)
    batch_inputs, batch_data_samples = output_data['inputs'], output_data[
        'data_samples']
    assert isinstance(batch_inputs, torch.Tensor)
    assert batch_inputs.shape == (1, 2, 3, 20, 20)
    assert isinstance(batch_data_samples, List)
    assert isinstance(batch_data_samples[0], EditDataSample)
    assert batch_data_samples[0].gt_img.data.shape == (3, 20, 20)
    assert processor.padded_sizes is not None

    # destructor
    outputs = processor.destructor(batch_inputs)
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (1, 2, 3, 20, 20)

    # test image
    processor = EditDataPreprocessor()

    # prepare data
    inputs = torch.rand(1, 3, 20, 20)
    target = torch.rand(3, 20, 20)
    data_sample = EditDataSample(gt_img=PixelData(data=target))
    data = dict(inputs=inputs, data_samples=[data_sample])

    # process
    output_data = processor(data)
    batch_inputs, batch_data_samples = output_data['inputs'], output_data[
        'data_samples']
    assert isinstance(batch_inputs, torch.Tensor)
    assert batch_inputs.shape == (1, 3, 20, 20)
    assert isinstance(batch_data_samples, List)
    assert isinstance(batch_data_samples[0], EditDataSample)
    assert batch_data_samples[0].gt_img.data.shape == (3, 20, 20)
    assert processor.padded_sizes is not None
    # destructor
    outputs = processor.destructor(batch_inputs)
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (1, 3, 20, 20)

    with pytest.raises(AssertionError):
        EditDataPreprocessor(mean=(1, 1))
    with pytest.raises(AssertionError):
        EditDataPreprocessor(std=(1, 1))


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

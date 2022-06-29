# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import pytest
import torch

from mmedit.data_element import EditDataSample, PixelData
from mmedit.models import EditDataPreprocessor


def test_edit_data_preprocessor():

    # test frames
    processor = EditDataPreprocessor(
        input_view=(1, -1, 1, 1), output_view=(1, -1, 1, 1))
    # prepare data
    inputs = torch.rand(2, 3, 20, 20)
    target = torch.rand(3, 20, 20)
    data_sample = EditDataSample(gt_img=PixelData(data=target))
    data = [dict(inputs=inputs, data_sample=data_sample)]
    # process
    batch_inputs, batch_data_samples = processor(data)
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
    inputs = torch.rand(3, 20, 20)
    target = torch.rand(3, 20, 20)
    data_sample = EditDataSample(gt_img=PixelData(data=target))
    data = [dict(inputs=inputs, data_sample=data_sample)]
    # process
    batch_inputs, batch_data_samples = processor(data)
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

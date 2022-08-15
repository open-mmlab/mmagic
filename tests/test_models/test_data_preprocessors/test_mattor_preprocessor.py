# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

from mmedit.models.data_preprocessors import MattorPreprocessor
from mmedit.structures import EditDataSample, PixelData


def test_mattor_preprocessor():

    processor = MattorPreprocessor()
    # prepare data
    inputs = torch.rand(3, 20, 20)
    target = torch.rand(3, 20, 20)
    data_sample = EditDataSample(trimap=PixelData(data=target))
    data = [dict(inputs=inputs, data_sample=data_sample)]
    # process
    batch_inputs, batch_data_samples = processor(data)
    assert isinstance(batch_inputs, torch.Tensor)
    assert batch_inputs.shape == (1, 6, 20, 20)
    assert isinstance(batch_data_samples, List)
    assert isinstance(batch_data_samples[0], EditDataSample)
    assert batch_data_samples[0].trimap.data.shape == (3, 20, 20)

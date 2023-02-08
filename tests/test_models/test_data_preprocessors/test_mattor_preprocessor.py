# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import pytest
import torch
from mmengine.testing import assert_allclose

from mmedit.models.data_preprocessors import MattorPreprocessor
from mmedit.structures import EditDataSample


def test_mattor_preprocessor():

    processor = MattorPreprocessor()
    # prepare data
    inputs = torch.rand(1, 3, 20, 20)
    target = torch.rand(3, 20, 20)
    gt_fg = torch.rand(3, 20, 20)
    gt_bg = torch.rand(3, 20, 20)
    data_sample = EditDataSample(trimap=target, gt_fg=gt_fg, gt_bg=gt_bg)
    data = dict(inputs=inputs, data_samples=[data_sample])
    # process
    data = processor(data, True)
    batch_inputs, batch_data_samples = data['inputs'], data['data_samples']
    assert isinstance(batch_inputs, torch.Tensor)
    assert batch_inputs.shape == (1, 6, 20, 20)
    assert isinstance(batch_data_samples, List)
    assert isinstance(batch_data_samples[0], EditDataSample)
    assert batch_data_samples[0].trimap.shape == (3, 20, 20)

    # test proc_batch_trimap
    processor = MattorPreprocessor(proc_trimap='as_is')
    inputs = torch.rand(1, 3, 20, 20)
    target = torch.rand(3, 20, 20)
    data_sample = EditDataSample(trimap=target)
    data = dict(inputs=inputs, data_samples=[data_sample])
    data = processor(data, True)
    batch_inputs, batch_data_samples = data['inputs'], data['data_samples']
    assert isinstance(batch_inputs, torch.Tensor)
    assert batch_inputs.shape == (1, 6, 20, 20)
    assert isinstance(batch_data_samples, List)
    assert isinstance(batch_data_samples[0], EditDataSample)
    assert batch_data_samples[0].trimap.shape == (3, 20, 20)
    assert_allclose(batch_data_samples[0].trimap, target)

    # test error in proc_batch_trimap
    processor = MattorPreprocessor(proc_trimap='wrong_method')
    with pytest.raises(ValueError):
        processor(data)

    # test training is False
    processor = MattorPreprocessor(proc_trimap='as_is')
    inputs = torch.rand(1, 3, 20, 20)
    target = torch.rand(3, 20, 20)
    gt_fg = torch.rand(3, 20, 20)
    gt_bg = torch.rand(3, 20, 20)
    data_sample = EditDataSample(trimap=target)
    data = dict(inputs=inputs, data_samples=[data_sample])

    data = processor(data, training=False)

# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.testing import assert_allclose

from mmagic.models.data_preprocessors import MattorPreprocessor
from mmagic.structures import DataSample


def test_mattor_preprocessor():

    processor = MattorPreprocessor()
    # prepare data
    inputs = torch.rand(1, 3, 20, 20)
    target = torch.rand(3, 20, 20)
    gt_fg = torch.rand(3, 20, 20)
    gt_bg = torch.rand(3, 20, 20)
    data_sample = DataSample(trimap=target, gt_fg=gt_fg, gt_bg=gt_bg)
    data = dict(inputs=inputs, data_samples=[data_sample])
    # process
    data = processor(data, True)
    batch_inputs, batch_data_samples = data['inputs'], data['data_samples']
    assert isinstance(batch_inputs, torch.Tensor)
    assert batch_inputs.shape == (1, 6, 20, 20)
    assert isinstance(batch_data_samples, DataSample)
    assert batch_data_samples.trimap.shape == (1, 3, 20, 20)

    # test proc_batch_trimap
    processor = MattorPreprocessor(proc_trimap='as_is')
    inputs = torch.rand(1, 3, 20, 20)
    target = torch.rand(3, 20, 20)
    data_sample = DataSample(trimap=target)
    data = dict(inputs=inputs, data_samples=[data_sample])
    data = processor(data, True)
    batch_inputs, batch_data_samples = data['inputs'], data['data_samples']
    assert isinstance(batch_inputs, torch.Tensor)
    assert batch_inputs.shape == (1, 6, 20, 20)
    assert isinstance(batch_data_samples, DataSample)
    assert batch_data_samples.trimap.shape == (1, 3, 20, 20)
    assert_allclose(batch_data_samples.trimap[0], target)

    # test error in proc_batch_trimap
    processor = MattorPreprocessor(proc_trimap='wrong_method')
    with pytest.raises(ValueError):
        processor(dict(inputs=inputs, data_samples=[data_sample]))

    # test training is False
    processor = MattorPreprocessor(proc_trimap='as_is')
    inputs = torch.rand(1, 3, 20, 20)
    target = torch.rand(3, 20, 20)
    gt_fg = torch.rand(3, 20, 20)
    gt_bg = torch.rand(3, 20, 20)
    data_sample = DataSample(trimap=target)
    data = dict(inputs=inputs, data_samples=[data_sample])

    data = processor(data, training=False)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

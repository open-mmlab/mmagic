# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock, patch

import pytest
import torch
from mmengine.model import MMDistributedDataParallel

from mmagic.evaluation.metrics.base_gen_metric import (GenerativeMetric,
                                                       GenMetric)


def mock_collect_fn(results, *args, **kwargs):
    return results[0]


@patch('mmagic.evaluation.metrics.base_gen_metric.collect_results',
       mock_collect_fn)
class ToyMetric(GenMetric):

    def process(self, data_batch, data_samples):
        return

    def compute_metrics(self, results):
        return dict(score=1)


@patch('mmagic.evaluation.metrics.base_gen_metric.collect_results',
       mock_collect_fn)
class ToyGenerativeMetric(GenerativeMetric):

    def process(self, data_batch, data_samples):
        return

    def compute_metrics(self, results):
        return dict(score=1)


def test_GenMetric():
    metric = ToyMetric(10, 10)
    assert metric.real_nums_per_device == 10

    metric.real_results = []

    # test get_metric_sampler
    model = MagicMock()
    dataset = MagicMock()
    dataset.__len__.return_value = 10
    dataloader = MagicMock()
    dataloader.batch_size = 4
    dataloader.dataset = dataset
    sampler = metric.get_metric_sampler(model, dataloader, [metric])
    assert sampler.dataset == dataset

    # test prepare
    model = MagicMock(spec=MMDistributedDataParallel)
    preprocessor = MagicMock()
    model.module = MagicMock()
    model.module.data_preprocessor = preprocessor
    metric.prepare(model, dataloader)
    assert metric.data_preprocessor == preprocessor

    # test raise error with dataset is length than real_nums
    dataset.__len__.return_value = 5
    with pytest.raises(AssertionError):
        metric.get_metric_sampler(model, dataloader, [metric])


def test_GenerativeMetric():
    metric = ToyGenerativeMetric(11, need_cond_input=True)
    assert metric.need_cond_input
    assert metric.real_nums == 0
    assert metric.fake_nums == 11

    # NOTE: only test whether returned sampler is correct in this UT
    def side_effect(index):
        return {'gt_label': [i for i in range(index, index + 3)]}

    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=2)
    dataset.get_data_info.side_effect = side_effect
    dataloader = MagicMock()
    dataloader.batch_size = 10
    dataloader.dataset = dataset

    model = MagicMock()
    sampler = metric.get_metric_sampler(model, dataloader, [metric])
    assert sampler.dataset == dataset
    assert sampler.batch_size == 10
    assert sampler.max_length == 11
    assert sampler.sample_model == 'ema'

    # index passed to `side_effect` can only be 0 or 1
    assert len(sampler) == 2

    iterator = iter(sampler)
    output = next(iterator)
    assert output['inputs'] == dict(
        sample_model='ema', num_batches=10, sample_kwargs={})
    assert len(output['data_samples']) == 10

    target_label_list = [
        torch.FloatTensor([0, 1, 2]),
        torch.FloatTensor([1, 2, 3])
    ]
    # check if all cond in target label list
    for data in output['data_samples']:
        label = data.gt_label.label
        assert any([(label == tar).all() for tar in target_label_list])

    # test with sample kwargs
    sample_kwargs = dict(
        num_inference_steps=250, show_progress=True, classifier_scale=1.)
    metric = ToyGenerativeMetric(
        11, need_cond_input=True, sample_kwargs=sample_kwargs)
    sampler = metric.get_metric_sampler(model, dataloader, [metric])
    iterator = iter(sampler)
    output = next(iterator)
    assert output['inputs'] == dict(
        sample_model='ema', num_batches=10, sample_kwargs=sample_kwargs)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

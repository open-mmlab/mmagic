# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock, patch

from mmengine.model import MMDistributedDataParallel

from mmedit.evaluation.metrics.base_gen_metric import GenMetric


def mock_collect_fn(results, *args, **kwargs):
    return results[0]


@patch('mmedit.evaluation.metrics.base_gen_metric.collect_results',
       mock_collect_fn)
class ToyMetric(GenMetric):

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

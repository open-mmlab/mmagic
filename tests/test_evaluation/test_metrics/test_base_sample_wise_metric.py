# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest.mock import MagicMock

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from mmagic.datasets import BasicImageDataset
from mmagic.evaluation.metrics.base_sample_wise_metric import \
    BaseSampleWiseMetric


class BaseSampleWiseMetricMock(BaseSampleWiseMetric):

    metric = 'metric'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_image(self, *args, **kwargs):
        return 0


def test_compute_metrics():
    metric = BaseSampleWiseMetricMock()

    results = []
    key = 'metric'
    total = 0
    n = 0
    for i in range(10):
        results.append({'batch_size': i, key: i})
        total += i * i
        n += i
    average = metric.compute_metrics(results)
    assert average[key] == total / n


def test_process():
    metric = BaseSampleWiseMetricMock()

    mask = np.ones((32, 32, 3)) * 2
    mask[:16] *= 0
    gt = np.ones((32, 32, 3)) * 2
    data_sample = dict(gt_img=gt, mask=mask)
    data_batch = [dict(data_samples=data_sample)]
    predictions = [dict(pred_img=np.ones((32, 32, 3)))]

    data_batch.append(
        dict(
            data_samples=dict(
                gt_img=torch.from_numpy(gt), mask=torch.from_numpy(mask))))

    predictions.append({
        k: torch.from_numpy(deepcopy(v))
        for (k, v) in predictions[0].items()
    })

    for d, p in zip(data_batch, predictions):
        d['output'] = p

    predictions = data_batch
    metric.process(data_batch, predictions)
    assert len(metric.results) == 2
    assert metric.results[0]['metric'] == 0


def test_prepare():
    data_root = 'tests/data/image/'
    dataset = BasicImageDataset(
        metainfo=dict(dataset_type='sr_annotation_dataset', task_name='sisr'),
        data_root=data_root,
        data_prefix=dict(img='lq', gt='gt'),
        filename_tmpl=dict(img='{}_x4'),
        pipeline=[])
    dataloader = DataLoader(dataset)

    metric = BaseSampleWiseMetricMock()
    model = MagicMock()
    model.data_preprocessor = MagicMock()

    metric.prepare(model, dataloader)
    assert metric.SAMPLER_MODE == 'normal'
    assert metric.sample_model == 'orig'
    assert metric.size == 1
    assert metric.data_preprocessor == model.data_preprocessor

    metric.get_metric_sampler(None, dataloader, [])
    assert dataloader == dataloader


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

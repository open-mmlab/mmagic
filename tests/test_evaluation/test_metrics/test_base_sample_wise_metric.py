# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
import torch

from mmedit.evaluation.metrics import base_sample_wise_metric


def test_compute_metrics():
    metric = base_sample_wise_metric.BaseSampleWiseMetric()
    metric.metric = 'metric'
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
    metric = base_sample_wise_metric.BaseSampleWiseMetric()
    metric.metric = 'metric'

    mask = np.ones((32, 32, 3)) * 2
    mask[:16] *= 0
    gt = np.ones((32, 32, 3)) * 2
    data_sample = dict(gt_img=gt, mask=mask, gt_channel_order='bgr')
    data_batch = [dict(data_samples=data_sample)]
    predictions = [dict(pred_img=np.ones((32, 32, 3)))]

    data_batch.append(
        dict(
            data_samples=dict(
                gt_img=torch.from_numpy(gt),
                mask=torch.from_numpy(mask),
                img_channel_order='bgr')))
    predictions.append({
        k: torch.from_numpy(deepcopy(v))
        for (k, v) in predictions[0].items()
    })
    metric.process(data_batch, predictions)
    assert len(metric.results) == 2
    assert metric.results[0]['metric'] == 0

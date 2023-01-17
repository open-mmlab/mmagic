# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import mmcv
import numpy as np
import pytest
import torch

from mmedit.evaluation.metrics import NIQE


def test_niqe():
    img = mmcv.imread('tests/data/image/gt/baboon.png')

    predictions = [dict(pred_img=img)]
    predictions.append({
        k: torch.from_numpy(deepcopy(v))
        for (k, v) in predictions[0].items()
    })

    data_batch = [
        dict(
            data_samples=dict(
                gt_img=torch.from_numpy(img), gt_channel_order='bgr'))
    ]
    data_batch.append(
        dict(
            data_samples=dict(
                gt_img=torch.from_numpy(img), img_channel_order='bgr')))

    data_samples = [d_['data_samples'] for d_ in data_batch]
    for d, p in zip(data_samples, predictions):
        d['output'] = p

    niqe_ = NIQE()
    niqe_.process(data_batch, data_samples)
    result = niqe_.compute_metrics(niqe_.results)
    assert 'NIQE' in result
    np.testing.assert_almost_equal(result['NIQE'], 5.731541051885604)

    niqe_ = NIQE(key='gt_img', is_predicted=False)
    niqe_.process(data_batch, data_samples)
    result = niqe_.compute_metrics(niqe_.results)
    assert 'NIQE' in result
    np.testing.assert_almost_equal(result['NIQE'], 5.731541051885604)

    with pytest.raises(AssertionError):
        niqe_ = NIQE(convert_to='a')

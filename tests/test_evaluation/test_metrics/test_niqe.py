# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import mmcv
import numpy as np
import pytest
import torch

from mmedit.metrics import NIQE, niqe


def test_niqe():
    img = mmcv.imread('tests/data/image/gt/baboon.png')

    data_batch = [dict(gt_img=img)]
    predictions = [dict(pred_img=img)]

    data_batch.append(
        {k: torch.from_numpy(deepcopy(v))
         for (k, v) in data_batch[0].items()})
    predictions.append({
        k: torch.from_numpy(deepcopy(v))
        for (k, v) in predictions[0].items()
    })

    niqe_ = NIQE()
    niqe_.process(data_batch, predictions)
    result = niqe_.compute_metrics(niqe_.results)
    assert 'NIQE' in result
    np.testing.assert_almost_equal(result['NIQE'], 5.731541051885604)

    niqe_ = NIQE(key='gt_img', is_predicted=False)
    niqe_.process(data_batch, predictions)
    result = niqe_.compute_metrics(niqe_.results)
    assert 'NIQE' in result
    np.testing.assert_almost_equal(result['NIQE'], 5.731541051885604)

    with pytest.raises(AssertionError):
        niqe_ = NIQE(convert_to='a')


def test_calculate_niqe():
    img = mmcv.imread('tests/data/image/gt/baboon.png')

    result = niqe(img[:, :, 0], crop_border=0, input_order='HW')
    np.testing.assert_almost_equal(result, 5.62525, decimal=5)
    result = niqe(img, crop_border=0, input_order='HWC', convert_to='y')
    np.testing.assert_almost_equal(result, 5.72957, decimal=5)
    result = niqe(img, crop_border=0, input_order='HWC', convert_to='gray')
    np.testing.assert_almost_equal(result, 5.73154, decimal=5)
    result = niqe(
        img.transpose(2, 0, 1),
        crop_border=0,
        input_order='CHW',
        convert_to='y')
    np.testing.assert_almost_equal(result, 5.72957, decimal=5)
    result = niqe(
        img.transpose(2, 0, 1),
        crop_border=0,
        input_order='CHW',
        convert_to='gray')
    np.testing.assert_almost_equal(result, 5.73154, decimal=5)

    result = niqe(img[:, :, 0], crop_border=6, input_order='HW')
    np.testing.assert_almost_equal(result, 5.82981, decimal=5)
    result = niqe(img, crop_border=6, input_order='HWC', convert_to='y')
    np.testing.assert_almost_equal(result, 6.10074, decimal=5)
    result = niqe(
        img.transpose(2, 0, 1),
        crop_border=6,
        input_order='CHW',
        convert_to='y')
    np.testing.assert_almost_equal(result, 6.10074, decimal=5)

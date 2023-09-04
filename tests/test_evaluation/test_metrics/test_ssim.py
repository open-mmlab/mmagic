# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
import pytest
import torch

from mmagic.evaluation.metrics import SSIM, ssim


def test_ssim():

    mask = np.ones((3, 32, 32)) * 2
    mask[:16] *= 0
    gt_img = np.ones((3, 32, 32)) * 2
    predictions = [dict(pred_img=np.ones((3, 32, 32)))]

    data_batch = [
        dict(
            data_samples=dict(
                gt_img=gt_img, mask=mask, gt_channel_order='bgr'))
    ]
    data_batch.append(
        dict(
            data_samples=dict(
                gt_img=torch.from_numpy(gt_img),
                mask=torch.from_numpy(mask),
                img_channel_order='bgr')))
    predictions.append({
        k: torch.from_numpy(deepcopy(v))
        for (k, v) in predictions[0].items()
    })

    data_samples = [d_['data_samples'] for d_ in data_batch]
    for d, p in zip(data_samples, predictions):
        d['output'] = p

    ssim_ = SSIM()
    ssim_.process(data_batch, data_samples)
    result = ssim_.compute_metrics(ssim_.results)
    assert 'SSIM' in result
    np.testing.assert_almost_equal(result['SSIM'], 0.913062377743969)


def test_calculate_ssim():
    img_hw_1 = np.ones((32, 32))
    img_hwc_1 = np.ones((32, 32, 3))
    img_chw_1 = np.ones((3, 32, 32))
    img_hw_2 = np.ones((32, 32)) * 2
    img_hwc_2 = np.ones((32, 32, 3)) * 2
    img_chw_2 = np.ones((3, 32, 32)) * 2

    with pytest.raises(ValueError):
        ssim(img_hw_1, img_hw_2, crop_border=0, input_order='HH')

    with pytest.raises(ValueError):
        ssim(img_hw_1, img_hw_2, crop_border=0, input_order='ABC')

    ssim_result = ssim(img_hw_1, img_hw_2, crop_border=0)
    np.testing.assert_almost_equal(ssim_result, 0.9130623)
    ssim_result = ssim(img_hwc_1, img_hwc_2, crop_border=0, input_order='HWC')
    np.testing.assert_almost_equal(ssim_result, 0.9130623)
    ssim_result = ssim(img_chw_1, img_chw_2, crop_border=0, input_order='CHW')
    np.testing.assert_almost_equal(ssim_result, 0.9130623)

    ssim_result = ssim(img_hw_1, img_hw_2, crop_border=2)
    np.testing.assert_almost_equal(ssim_result, 0.9130623)
    ssim_result = ssim(img_hwc_1, img_hwc_2, crop_border=3, input_order='HWC')
    np.testing.assert_almost_equal(ssim_result, 0.9130623)
    ssim_result = ssim(img_chw_1, img_chw_2, crop_border=4, input_order='CHW')
    np.testing.assert_almost_equal(ssim_result, 0.9130623)

    ssim_result = ssim(img_hwc_1, img_hwc_2, crop_border=0, convert_to=None)
    np.testing.assert_almost_equal(ssim_result, 0.9130623)
    ssim_result = ssim(img_hwc_1, img_hwc_2, crop_border=0, convert_to='Y')
    np.testing.assert_almost_equal(ssim_result, 0.9987801)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

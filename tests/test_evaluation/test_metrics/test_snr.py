# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
import pytest
import torch

from mmagic.evaluation.metrics import SNR, snr


class TestPixelMetrics:

    @classmethod
    def setup_class(cls):

        mask = np.ones((32, 32, 3)) * 2
        mask[:16] *= 0
        gt = np.ones((32, 32, 3)) * 2
        data_sample = dict(gt_img=gt, mask=mask, gt_channel_order='bgr')
        cls.data_batch = [dict(data_samples=data_sample)]
        cls.predictions = [dict(pred_img=np.ones((32, 32, 3)))]

        cls.data_batch.append(
            dict(
                data_samples=dict(
                    gt_img=torch.from_numpy(gt),
                    mask=torch.from_numpy(mask),
                    img_channel_order='bgr')))
        cls.predictions.append({
            k: torch.from_numpy(deepcopy(v))
            for (k, v) in cls.predictions[0].items()
        })

        for d, p in zip(cls.data_batch, cls.predictions):
            d['output'] = p
        cls.predictions = cls.data_batch

    def test_snr(self):

        snr_ = SNR()
        snr_.process(self.data_batch, self.predictions)
        result = snr_.compute_metrics(snr_.results)
        assert 'SNR' in result
        np.testing.assert_almost_equal(result['SNR'], 6.0206001996994)


def test_snr():
    img_hw_1 = np.ones((32, 32)) * 2
    img_hwc_1 = np.ones((32, 32, 3)) * 2
    img_chw_1 = np.ones((3, 32, 32)) * 2
    img_hw_2 = np.ones((32, 32))
    img_hwc_2 = np.ones((32, 32, 3))
    img_chw_2 = np.ones((3, 32, 32))

    with pytest.raises(ValueError):
        snr(img_hw_1, img_hw_2, crop_border=0, input_order='HH')

    with pytest.raises(ValueError):
        snr(img_hw_1, img_hw_2, crop_border=0, convert_to='ABC')

    snr_result = snr(img_hw_1, img_hw_2, crop_border=0)
    np.testing.assert_almost_equal(snr_result, 6.020600199699402)
    snr_result = snr(img_hwc_1, img_hwc_2, crop_border=0, input_order='HWC')
    np.testing.assert_almost_equal(snr_result, 6.020600199699402)
    print(snr_result)
    snr_result = snr(img_chw_1, img_chw_2, crop_border=0, input_order='CHW')
    np.testing.assert_almost_equal(snr_result, 6.020600199699402)
    print(snr_result)

    snr_result = snr(img_hw_1, img_hw_2, crop_border=2)
    np.testing.assert_almost_equal(snr_result, 6.020600199699402)
    snr_result = snr(img_hwc_1, img_hwc_2, crop_border=3, input_order='HWC')
    np.testing.assert_almost_equal(snr_result, 6.020600199699402)
    snr_result = snr(img_chw_1, img_chw_2, crop_border=4, input_order='CHW')
    np.testing.assert_almost_equal(snr_result, 6.020600199699402)

    snr_result = snr(img_hwc_1, img_hwc_2, crop_border=0, convert_to=None)
    np.testing.assert_almost_equal(snr_result, 6.020600199699402)
    snr_result = snr(img_hwc_1, img_hwc_2, crop_border=0, convert_to='Y')
    np.testing.assert_almost_equal(snr_result, 26.290040016174316)

    # test float inf
    snr_result = snr(img_hw_1, img_hw_1, crop_border=0)
    assert snr_result == float('inf')

    # test uint8
    img_hw_1 = np.ones((32, 32), dtype=np.uint8)
    img_hw_2 = np.zeros((32, 32), dtype=np.uint8)
    snr_result = snr(img_hw_1, img_hw_2, crop_border=0)
    assert snr_result == 0


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

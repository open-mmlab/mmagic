# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
import pytest
import torch

from mmedit.metrics import MAE, MSE, PSNR, SNR, psnr
from mmedit.metrics.utils import reorder_image


class TestPixelMetrics:

    @classmethod
    def setup_class(cls):

        mask = np.ones((32, 32, 3)) * 2
        mask[:16] *= 0
        cls.data_batch = [dict(gt_img=np.ones((32, 32, 3)) * 2, mask=mask)]
        cls.predictions = [dict(pred_img=np.ones((32, 32, 3)))]

        cls.data_batch.append({
            k: torch.from_numpy(deepcopy(v))
            for (k, v) in cls.data_batch[0].items()
        })
        cls.predictions.append({
            k: torch.from_numpy(deepcopy(v))
            for (k, v) in cls.predictions[0].items()
        })

    def test_mae(self):

        # Single MAE
        mae = MAE()
        mae.process(self.data_batch, self.predictions)
        result = mae.compute_metrics(mae.results)
        assert 'SingleMAE' in result
        np.testing.assert_almost_equal(result['SingleMAE'], 0.003921568627)

        # Masked MAE
        mae = MAE(mask_key='mask', prefix='MaskedMAE')
        mae.process(self.data_batch, self.predictions)
        result = mae.compute_metrics(mae.results)
        assert 'MaskedMAE' in result
        np.testing.assert_almost_equal(result['MaskedMAE'], 0.003921568627)

    def test_mse(self):

        # Single MSE
        mae = MSE()
        mae.process(self.data_batch, self.predictions)
        result = mae.compute_metrics(mae.results)
        assert 'SingleMSE' in result
        np.testing.assert_almost_equal(result['SingleMSE'], 0.000015378700496)

        # Masked MSE
        mae = MSE(mask_key='mask', prefix='MaskedMSE')
        mae.process(self.data_batch, self.predictions)
        result = mae.compute_metrics(mae.results)
        assert 'MaskedMSE' in result
        np.testing.assert_almost_equal(result['MaskedMSE'], 0.000015378700496)

    def test_psnr(self):

        psnr_ = PSNR()
        psnr_.process(self.data_batch, self.predictions)
        result = psnr_.compute_metrics(psnr_.results)
        assert 'PSNR' in result
        np.testing.assert_almost_equal(result['PSNR'], 48.1308036)

    def test_snr(self):

        snr_ = SNR()
        snr_.process(self.data_batch, self.predictions)
        result = snr_.compute_metrics(snr_.results)
        assert 'SNR' in result
        np.testing.assert_almost_equal(result['SNR'], 6.0205999132796)


def test_reorder_image():
    img_hw = np.ones((32, 32))
    img_hwc = np.ones((32, 32, 3))
    img_chw = np.ones((3, 32, 32))

    with pytest.raises(ValueError):
        reorder_image(img_hw, 'HH')

    output = reorder_image(img_hw)
    assert output.shape == (32, 32, 1)

    output = reorder_image(img_hwc)
    assert output.shape == (32, 32, 3)

    output = reorder_image(img_chw, input_order='CHW')
    assert output.shape == (32, 32, 3)


def test_psnr():
    img_hw_1 = np.ones((32, 32))
    img_hwc_1 = np.ones((32, 32, 3))
    img_chw_1 = np.ones((3, 32, 32))
    img_hw_2 = np.ones((32, 32)) * 2
    img_hwc_2 = np.ones((32, 32, 3)) * 2
    img_chw_2 = np.ones((3, 32, 32)) * 2

    with pytest.raises(ValueError):
        psnr(img_hw_1, img_hw_2, crop_border=0, input_order='HH')

    with pytest.raises(ValueError):
        psnr(img_hw_1, img_hw_2, crop_border=0, convert_to='ABC')

    psnr_result = psnr(img_hw_1, img_hw_2, crop_border=0)
    np.testing.assert_almost_equal(psnr_result, 48.1308036)
    psnr_result = psnr(img_hwc_1, img_hwc_2, crop_border=0, input_order='HWC')
    np.testing.assert_almost_equal(psnr_result, 48.1308036)
    psnr_result = psnr(img_chw_1, img_chw_2, crop_border=0, input_order='CHW')
    np.testing.assert_almost_equal(psnr_result, 48.1308036)

    psnr_result = psnr(img_hw_1, img_hw_2, crop_border=2)
    np.testing.assert_almost_equal(psnr_result, 48.1308036)
    psnr_result = psnr(img_hwc_1, img_hwc_2, crop_border=3, input_order='HWC')
    np.testing.assert_almost_equal(psnr_result, 48.1308036)
    psnr_result = psnr(img_chw_1, img_chw_2, crop_border=4, input_order='CHW')
    np.testing.assert_almost_equal(psnr_result, 48.1308036)

    psnr_result = psnr(img_hwc_1, img_hwc_2, crop_border=0, convert_to=None)
    np.testing.assert_almost_equal(psnr_result, 48.1308036)
    psnr_result = psnr(img_hwc_1, img_hwc_2, crop_border=0, convert_to='Y')
    np.testing.assert_almost_equal(psnr_result, 49.4527218)

    # test float inf
    psnr_result = psnr(img_hw_1, img_hw_1, crop_border=0)
    assert psnr_result == float('inf')

    # test uint8
    img_hw_1 = np.zeros((32, 32), dtype=np.uint8)
    img_hw_2 = np.ones((32, 32), dtype=np.uint8) * 255
    psnr_result = psnr(img_hw_1, img_hw_2, crop_border=0)
    assert psnr_result == 0


def test_snr():
    img_hw_1 = np.ones((32, 32))
    img_hwc_1 = np.ones((32, 32, 3))
    img_chw_1 = np.ones((3, 32, 32))
    img_hw_2 = np.ones((32, 32)) * 2
    img_hwc_2 = np.ones((32, 32, 3)) * 2
    img_chw_2 = np.ones((3, 32, 32)) * 2

    with pytest.raises(ValueError):
        psnr(img_hw_1, img_hw_2, crop_border=0, input_order='HH')

    with pytest.raises(ValueError):
        psnr(img_hw_1, img_hw_2, crop_border=0, convert_to='ABC')

    psnr_result = psnr(img_hw_1, img_hw_2, crop_border=0)
    np.testing.assert_almost_equal(psnr_result, 48.1308036)
    psnr_result = psnr(img_hwc_1, img_hwc_2, crop_border=0, input_order='HWC')
    np.testing.assert_almost_equal(psnr_result, 48.1308036)
    psnr_result = psnr(img_chw_1, img_chw_2, crop_border=0, input_order='CHW')
    np.testing.assert_almost_equal(psnr_result, 48.1308036)

    psnr_result = psnr(img_hw_1, img_hw_2, crop_border=2)
    np.testing.assert_almost_equal(psnr_result, 48.1308036)
    psnr_result = psnr(img_hwc_1, img_hwc_2, crop_border=3, input_order='HWC')
    np.testing.assert_almost_equal(psnr_result, 48.1308036)
    psnr_result = psnr(img_chw_1, img_chw_2, crop_border=4, input_order='CHW')
    np.testing.assert_almost_equal(psnr_result, 48.1308036)

    psnr_result = psnr(img_hwc_1, img_hwc_2, crop_border=0, convert_to=None)
    np.testing.assert_almost_equal(psnr_result, 48.1308036)
    psnr_result = psnr(img_hwc_1, img_hwc_2, crop_border=0, convert_to='Y')
    np.testing.assert_almost_equal(psnr_result, 49.4527218)

    # test float inf
    psnr_result = psnr(img_hw_1, img_hw_1, crop_border=0)
    assert psnr_result == float('inf')

    # test uint8
    img_hw_1 = np.zeros((32, 32), dtype=np.uint8)
    img_hw_2 = np.ones((32, 32), dtype=np.uint8) * 255
    psnr_result = psnr(img_hw_1, img_hw_2, crop_border=0)
    assert psnr_result == 0


t = TestPixelMetrics()
t.setup_class()
t.test_mae()
t.test_mse()
t.test_psnr()
t.test_snr()

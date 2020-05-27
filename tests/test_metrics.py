import numpy as np
import pytest
from mmedit.core.evaluation.metrics import (connectivity, gradient_error, mse,
                                            psnr, reorder_image, sad, ssim)


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


def test_calculate_psnr():
    img_hw_1 = np.ones((32, 32))
    img_hwc_1 = np.ones((32, 32, 3))
    img_chw_1 = np.ones((3, 32, 32))
    img_hw_2 = np.ones((32, 32)) * 2
    img_hwc_2 = np.ones((32, 32, 3)) * 2
    img_chw_2 = np.ones((3, 32, 32)) * 2

    with pytest.raises(ValueError):
        psnr(img_hw_1, img_hw_2, crop_border=0, input_order='HH')

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

    # test float inf
    psnr_result = psnr(img_hw_1, img_hw_1, crop_border=0)
    assert psnr_result == float('inf')


def test_calculate_ssim():
    img_hw_1 = np.ones((32, 32))
    img_hwc_1 = np.ones((32, 32, 3))
    img_chw_1 = np.ones((3, 32, 32))
    img_hw_2 = np.ones((32, 32)) * 2
    img_hwc_2 = np.ones((32, 32, 3)) * 2
    img_chw_2 = np.ones((3, 32, 32)) * 2

    with pytest.raises(ValueError):
        ssim(img_hw_1, img_hw_2, crop_border=0, input_order='HH')

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


def test_sad():
    alpha = np.ones((32, 32)) * 255
    pred_alpha = np.zeros((32, 32))
    trimap = np.zeros((32, 32))
    trimap[:16, :16] = 128
    trimap[16:, 16:] = 255

    with pytest.raises(AssertionError):
        # pred_alpha should be masked by trimap before evaluation
        sad(alpha, trimap, pred_alpha)

    with pytest.raises(ValueError):
        # input should all be two dimentional
        sad(alpha[..., None], trimap, pred_alpha)

    # mask pred_alpha
    pred_alpha[trimap == 0] = 0
    pred_alpha[trimap == 255] = 255

    sad_result = sad(alpha, trimap, pred_alpha)
    np.testing.assert_almost_equal(sad_result, 0.768)


def test_mse():
    alpha = np.ones((32, 32)) * 255
    pred_alpha = np.zeros((32, 32))
    trimap = np.zeros((32, 32))
    trimap[:16, :16] = 128
    trimap[16:, 16:] = 255

    with pytest.raises(AssertionError):
        # pred_alpha should be masked by trimap before evaluation
        mse(alpha, trimap, pred_alpha)

    with pytest.raises(ValueError):
        # input should all be two dimentional
        mse(alpha[..., None], trimap, pred_alpha)

    # mask pred_alpha
    pred_alpha[trimap == 0] = 0
    pred_alpha[trimap == 255] = 255

    mse_result = mse(alpha, trimap, pred_alpha)
    np.testing.assert_almost_equal(mse_result, 3.0)


def test_gradient_error():
    """Test gradient error for evaluating predicted alpha matte."""
    alpha = np.ones((32, 32)) * 255
    pred_alpha = np.zeros((32, 32))
    trimap = np.zeros((32, 32))
    trimap[:16, :16] = 128
    trimap[16:, 16:] = 255

    with pytest.raises(ValueError):
        # pred_alpha should be masked by trimap before evaluation
        gradient_error(alpha, trimap, pred_alpha)

    with pytest.raises(ValueError):
        # input should all be two dimentional
        gradient_error(alpha[..., None], trimap, pred_alpha)

    # mask pred_alpha
    pred_alpha[trimap == 0] = 0
    pred_alpha[trimap == 255] = 255

    gradient_result = gradient_error(alpha, trimap, pred_alpha)
    np.testing.assert_almost_equal(gradient_result, 0.0028887)


def test_connectivity():
    """Test connectivity error for evaluating predicted alpha matte."""
    alpha = np.ones((32, 32)) * 255
    pred_alpha = np.zeros((32, 32))
    trimap = np.zeros((32, 32))
    trimap[:16, :16] = 128
    trimap[16:, 16:] = 255

    with pytest.raises(ValueError):
        # pred_alpha should be masked by trimap before evaluation
        connectivity(alpha, trimap, pred_alpha)

    with pytest.raises(ValueError):
        # input should all be two dimentional
        connectivity(alpha[..., None], trimap, pred_alpha)

    # mask pred_alpha
    pred_alpha[trimap == 0] = 0
    pred_alpha[trimap == 255] = 255

    connectivity_result = connectivity(alpha, trimap, pred_alpha)
    np.testing.assert_almost_equal(connectivity_result, 0.256)

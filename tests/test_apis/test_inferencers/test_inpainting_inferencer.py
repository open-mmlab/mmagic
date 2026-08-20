# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pytest
import torch

from mmagic.apis.inferencers.inpainting_inferencer import InpaintingInferencer
from mmagic.utils import register_all_modules

register_all_modules()


def test_inpainting_inferencer():
    data_root = osp.join(osp.dirname(__file__), '../../')
    masked_img_path = data_root + 'data/inpainting/celeba_test.png'
    mask_path = data_root + 'data/inpainting/bbox_mask.png'
    cfg = osp.join(
        osp.dirname(__file__),
        '..',
        '..',
        '..',
        'configs',
        'aot_gan',
        'aot-gan_smpgan_4xb4_places-512x512.py',
    )
    result_out_dir = osp.join(
        osp.dirname(__file__), '..', '..', 'data/out', 'inpainting_result.png')

    inferencer_instance = \
        InpaintingInferencer(cfg, None)
    inferencer_instance(img=masked_img_path, mask=mask_path)

    rgb_img = mmcv.imread(masked_img_path, flag='color')[:, :, ::-1].copy()
    img = rgb_img[:, :, ::-1]
    mask = mmcv.imread(mask_path, flag='unchanged')
    inference_result = inferencer_instance(
        img=img, mask=mask, result_out_dir=result_out_dir)
    result_img = inference_result[1]
    assert result_img.shape == (256, 256, 3)


@pytest.mark.parametrize('img_as_array, mask_as_array', [
    (True, True),
    (True, False),
    (False, True),
])
def test_inpainting_preprocess_array_inputs(img_as_array, mask_as_array):
    data_root = osp.join(osp.dirname(__file__), '../../')
    img_path = data_root + 'data/inpainting/celeba_test.png'
    mask_path = data_root + 'data/inpainting/bbox_mask.png'
    rgb_img = mmcv.imread(img_path, flag='color')[:, :, ::-1].copy()
    img = rgb_img[:, :, ::-1]
    mask = mmcv.imread(mask_path, flag='unchanged')
    original_img = img.copy()
    original_mask = mask.copy()

    inferencer = object.__new__(InpaintingInferencer)
    expected = inferencer.preprocess(img=img_path, mask=mask_path)
    actual = inferencer.preprocess(
        img=img if img_as_array else img_path,
        mask=mask if mask_as_array else mask_path)

    torch.testing.assert_close(actual['inputs'][0], expected['inputs'][0])
    actual_sample = actual['data_samples'][0]
    expected_sample = expected['data_samples'][0]
    torch.testing.assert_close(actual_sample.gt_img, expected_sample.gt_img)
    torch.testing.assert_close(actual_sample.mask, expected_sample.mask)

    for key in [
            'ori_gt_shape', 'gt_channel_order', 'gt_color_type',
            'ori_img_shape', 'img_channel_order', 'img_color_type'
    ]:
        assert actual_sample.metainfo[key] == expected_sample.metainfo[key]

    np.testing.assert_array_equal(img, original_img)
    np.testing.assert_array_equal(mask, original_mask)


def test_inpainting_preprocess_array_shapes():
    img = np.zeros((4, 4), dtype=np.uint8)
    mask = np.zeros((4, 4, 3), dtype=np.uint8)
    mask[0, 0, 0] = 255
    mask[1, 1, 1] = 255

    inferencer = object.__new__(InpaintingInferencer)
    result = inferencer.preprocess(img=img, mask=mask)
    data_sample = result['data_samples'][0]

    assert result['inputs'][0].shape == (1, 4, 4)
    assert data_sample.gt_img.shape == (1, 4, 4)
    assert data_sample.mask.shape == (1, 4, 4)
    assert data_sample.gt_color_type == 'grayscale'
    assert data_sample.mask[0, 0, 0] == 1
    assert data_sample.mask[0, 1, 1] == 0


@pytest.mark.parametrize('img, mask, match', [
    (np.zeros((4, 4, 2)), np.zeros((4, 4)), 'Image array'),
    (np.zeros((4, 4, 3, 1)), np.zeros((4, 4)), 'Image array'),
    (np.zeros((4, 4, 3)), np.zeros((4, 4, 0)), 'Mask array'),
    (np.zeros((4, 4, 3)), np.zeros((4, 4, 1, 1)), 'Mask array'),
])
def test_inpainting_preprocess_invalid_array_shapes(img, mask, match):
    inferencer = object.__new__(InpaintingInferencer)
    with pytest.raises(ValueError, match=match):
        inferencer.preprocess(img=img, mask=mask)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

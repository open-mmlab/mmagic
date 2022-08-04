# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import mmcv
import numpy as np
import pytest

from mmedit.transforms import GetSpatialDiscountMask, LoadMask


def test_dct_mask():
    mask = np.zeros((64, 64, 1))
    mask[20:40, 20:40] = 1.
    mask_bbox = [20, 20, 20, 20]
    results = dict(mask=mask, mask_bbox=mask_bbox)

    dct_mask = GetSpatialDiscountMask()
    results = dct_mask(results)
    assert 'discount_mask' in results
    assert results['discount_mask'].shape == (64, 64, 1)

    mask_height = mask_width = 20
    gamma = 0.99
    beta = 1.5
    mask_values = np.ones((mask_width, mask_height, 1))
    for i in range(mask_width):
        for j in range(mask_height):
            mask_values[i,
                        j] = max(gamma**(min(i, mask_width - i - 1) * beta),
                                 gamma**(min(j, mask_height - j - 1) * beta))
    dct_mask_test = np.zeros_like(mask)
    dct_mask_test[20:40, 20:40, ...] = mask_values

    np.testing.assert_almost_equal(dct_mask_test, results['discount_mask'])
    repr_str = dct_mask.__class__.__name__ + (f'(gamma={dct_mask.gamma}, '
                                              f'beta={dct_mask.beta})')
    assert repr_str == repr(dct_mask)


class TestInpaintLoading:

    @classmethod
    def setup_class(cls):
        cls.img_path = Path(__file__).parent.parent.parent.joinpath(
            'data/image/test.png')
        cls.results = dict(img_info=dict(filename=cls.img_path))

    def test_load_mask(self):

        # test mask mode: set
        mask_config = dict(
            mask_list_file='tests/data/inpainting/mask_list.txt',
            prefix='tests/data/inpainting/',
            io_backend='disk',
            color_type='unchanged',
            file_client_kwargs=dict())

        set_loader = LoadMask('set', mask_config)
        class_name = set_loader.__class__.__name__
        assert repr(set_loader) == class_name + "(mask_mode='set')"
        for _ in range(2):
            results = dict()
            results = set_loader(results)
            gt_mask = mmcv.imread(
                'tests/data/inpainting/mask/test.png', flag='unchanged')
            assert np.array_equal(results['mask'], gt_mask[..., 0:1] / 255.)

        mask_config = dict(
            mask_list_file='tests/data/inpainting/mask_list_single_ch.txt',
            prefix='tests/data/inpainting/',
            io_backend='disk',
            color_type='unchanged',
            file_client_kwargs=dict())

        # test mask mode: set with input as single channel image
        set_loader = LoadMask('set', mask_config)
        results = dict()
        results = set_loader(results)
        gt_mask = mmcv.imread(
            'tests/data/inpainting/mask/test_single_ch.png', flag='unchanged')
        gt_mask = np.expand_dims(gt_mask, axis=2)
        assert np.array_equal(results['mask'], gt_mask[..., 0:1] / 255.)

        # test mask mode: ff
        mask_config = dict(
            img_shape=(256, 256),
            num_vertices=(4, 12),
            mean_angle=1.2,
            angle_range=0.4,
            brush_width=(12, 40))

        ff_loader = LoadMask('ff', mask_config)
        results = dict()
        results = ff_loader(results)
        assert results['mask'].shape == (256, 256, 1)

        # test mask mode: irregular holes
        mask_config = dict(
            img_shape=(256, 256),
            num_vertices=(4, 12),
            max_angle=4.,
            length_range=(10, 100),
            brush_width=(10, 40),
            area_ratio_range=(0.15, 0.5))

        irregular_loader = LoadMask('irregular', mask_config)
        results = dict()
        results = irregular_loader(results)
        assert results['mask'].shape == (256, 256, 1)

        # test mask mode: bbox
        mask_config = dict(img_shape=(256, 256), max_bbox_shape=128)

        bbox_loader = LoadMask('bbox', mask_config)
        results = dict()
        results = bbox_loader(results)
        assert results['mask'].shape == (256, 256, 1)

        # test mask mode: file
        mask_loader = LoadMask('file')
        mask = mask_loader(
            dict(mask_path='tests/data/inpainting/mask/test_single_ch.png'))
        assert mask['mask'].shape == (256, 256, 1)

        with pytest.raises(NotImplementedError):
            loader = LoadMask('xxxx', mask_config)
            results = loader(results)

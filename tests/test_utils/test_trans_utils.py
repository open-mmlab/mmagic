# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import numpy as np
import pytest

from mmagic.datasets.transforms import (CropAroundCenter, CropAroundFg,
                                        CropAroundUnknown, LoadImageFromFile)
from mmagic.utils import (adjust_gamma, bbox2mask, brush_stroke_mask,
                          get_irregular_mask, random_bbox)

dtype_range = {
    np.bool_: (False, True),
    np.bool8: (False, True),
    np.float16: (-1, 1),
    np.float32: (-1, 1),
    np.float64: (-1, 1)
}


class TestCrop:

    @classmethod
    def setup_class(cls):
        """Check the dimension of gray scale images read by
        LoadImageFromFile."""
        image_loader = LoadImageFromFile(key='img')
        path_alpha = Path(
            __file__
        ).parent.parent / 'data' / 'matting_dataset' / 'alpha' / 'GT05.jpg'  # noqa: E501
        result = image_loader({'img_path': path_alpha})
        if result['img'].ndim == 3:
            cls.ext_dim = (1, )
        elif result['img'].ndim == 2:
            cls.ext_dim = ()
        else:
            raise ValueError('invalid ndim')

    @classmethod
    def check_ndim(cls, result_img_shape):
        if cls.ext_dim:
            return len(result_img_shape) == 3
        else:
            return not (len(result_img_shape.ndim) == 3
                        and result_img_shape[-1] == 1)

    @classmethod
    def check_crop(cls, result_img_shape, result_bbox):
        crop_w = result_bbox[2] - result_bbox[0]
        """Check if the result_bbox is in correspond to result_img_shape."""
        crop_h = result_bbox[3] - result_bbox[1]
        crop_shape = (crop_h, crop_w)
        return result_img_shape[:2] == crop_shape

    @staticmethod
    def check_crop_around_semi(alpha):
        return ((alpha > 0) & (alpha < 255)).any()

    @staticmethod
    def check_keys_contain(result_keys, target_keys):
        """Check if all elements in target_keys is in result_keys."""
        return set(target_keys).issubset(set(result_keys))

    def test_crop_around_center(self):

        with pytest.raises(TypeError):
            CropAroundCenter(320.)
        with pytest.raises(AssertionError):
            CropAroundCenter((320, 320, 320))

        target_keys = ['fg', 'bg', 'alpha', 'trimap', 'crop_bbox']

        fg = np.random.rand(240, 320, 3)
        bg = np.random.rand(240, 320, 3)
        trimap = np.random.rand(240, 320, *self.ext_dim)
        alpha = np.random.rand(240, 320, *self.ext_dim)

        # make sure there would be semi-transparent area
        trimap[128, 128] = 128
        results = dict(fg=fg, bg=bg, trimap=trimap, alpha=alpha)
        crop_around_center = CropAroundCenter(
            crop_size=330)  # this will trigger rescale
        crop_around_center_results = crop_around_center(results)
        assert self.check_keys_contain(crop_around_center_results.keys(),
                                       target_keys)
        assert self.check_ndim(crop_around_center_results['alpha'].shape)
        assert self.check_ndim(crop_around_center_results['trimap'].shape)
        assert self.check_crop(crop_around_center_results['alpha'].shape,
                               crop_around_center_results['crop_bbox'])
        assert self.check_crop_around_semi(crop_around_center_results['alpha'])

        # make sure there would be semi-transparent area
        trimap[:, :] = 128
        results = dict(fg=fg, bg=bg, trimap=trimap, alpha=alpha)
        crop_around_center = CropAroundCenter(crop_size=200)
        crop_around_center_results = crop_around_center(results)
        assert self.check_keys_contain(crop_around_center_results.keys(),
                                       target_keys)
        assert self.check_ndim(crop_around_center_results['alpha'].shape)
        assert self.check_ndim(crop_around_center_results['trimap'].shape)
        assert self.check_crop(crop_around_center_results['alpha'].shape,
                               crop_around_center_results['crop_bbox'])
        assert self.check_crop_around_semi(crop_around_center_results['alpha'])

        repr_str = crop_around_center.__class__.__name__ + (
            f'(crop_size={(200, 200)})')
        assert repr(crop_around_center) == repr_str

    def test_crop_around_fg(self):
        with pytest.raises(ValueError):
            # keys must contain 'seg'
            CropAroundFg(['fg', 'bg'])
        with pytest.raises(TypeError):
            # bd_ratio_range must be a tuple of 2 float
            CropAroundFg(['seg', 'merged'], bd_ratio_range=0.1)

        keys = ['bg', 'merged', 'seg']
        target_keys = ['bg', 'merged', 'seg', 'crop_bbox']

        bg = np.random.rand(60, 60, 3)
        merged = np.random.rand(60, 60, 3)
        seg = np.random.rand(60, 60)
        results = dict(bg=bg, merged=merged, seg=seg)

        crop_around_fg = CropAroundFg(keys)
        crop_around_fg_results = crop_around_fg(results)
        assert self.check_keys_contain(crop_around_fg_results.keys(),
                                       target_keys)
        assert self.check_crop(crop_around_fg_results['seg'].shape,
                               crop_around_fg_results['crop_bbox'])

        crop_around_fg = CropAroundFg(keys, test_mode=True)
        crop_around_fg_results = crop_around_fg(results)
        result_img_shape = crop_around_fg_results['seg'].shape
        assert self.check_keys_contain(crop_around_fg_results.keys(),
                                       target_keys)
        assert self.check_crop(result_img_shape,
                               crop_around_fg_results['crop_bbox'])
        # it should be a square in test mode
        assert result_img_shape[0] == result_img_shape[1]

    def test_crop_around_unknown(self):
        with pytest.raises(ValueError):
            # keys must contain 'alpha'
            CropAroundUnknown(['fg', 'bg'], [320])
        with pytest.raises(TypeError):
            # crop_size must be a list
            CropAroundUnknown(['alpha'], 320)
        with pytest.raises(TypeError):
            # crop_size must be a list of int
            CropAroundUnknown(['alpha'], [320.])
        with pytest.raises(ValueError):
            # unknown_source must be either 'alpha' or 'trimap'
            CropAroundUnknown(['alpha', 'fg'], [320], unknown_source='fg')
        with pytest.raises(ValueError):
            # if unknown_source is 'trimap', then keys must contain it
            CropAroundUnknown(['alpha', 'fg'], [320], unknown_source='trimap')

        keys = ['fg', 'bg', 'merged', 'alpha', 'trimap', 'ori_merged']
        target_keys = [
            'fg', 'bg', 'merged', 'alpha', 'trimap', 'ori_merged', 'crop_bbox'
        ]

        # test cropping using trimap to decide unknown area
        fg = np.random.rand(240, 320, 3)
        bg = np.random.rand(240, 320, 3)
        merged = np.random.rand(240, 320, 3)
        ori_merged = merged.copy()
        alpha = np.zeros((240, 320, *self.ext_dim))
        # make sure there would be unknown area
        alpha[:16, -16:] = 128
        trimap = np.zeros_like(alpha)
        trimap[alpha > 0] = 128
        trimap[alpha == 255] = 255
        results = dict(
            fg=fg,
            bg=bg,
            merged=merged,
            ori_merged=ori_merged,
            alpha=alpha,
            trimap=trimap)
        crop_around_semi_trans = CropAroundUnknown(
            keys, crop_sizes=[320], unknown_source='trimap')
        crop_around_semi_trans_results = crop_around_semi_trans(results)
        assert self.check_keys_contain(crop_around_semi_trans_results.keys(),
                                       target_keys)
        assert self.check_ndim(crop_around_semi_trans_results['alpha'].shape)
        assert self.check_ndim(crop_around_semi_trans_results['trimap'].shape)
        assert self.check_crop(crop_around_semi_trans_results['alpha'].shape,
                               crop_around_semi_trans_results['crop_bbox'])
        assert self.check_crop_around_semi(
            crop_around_semi_trans_results['alpha'])

        keys = ['fg', 'bg', 'merged', 'alpha', 'ori_merged']
        target_keys = [
            'fg', 'bg', 'merged', 'alpha', 'ori_merged', 'crop_bbox'
        ]

        # test cropping using alpha to decide unknown area
        fg = np.random.rand(240, 320, 3)
        bg = np.random.rand(240, 320, 3)
        merged = np.random.rand(240, 320, 3)
        ori_merged = merged.copy()
        alpha = np.random.rand(240, 320, *self.ext_dim)
        # make sure there would be unknown area
        alpha[120:160, 120:160] = 128
        results = dict(
            fg=fg, bg=bg, merged=merged, ori_merged=ori_merged, alpha=alpha)
        crop_around_semi_trans = CropAroundUnknown(
            keys, crop_sizes=[160], unknown_source='alpha')
        crop_around_semi_trans_results = crop_around_semi_trans(results)
        assert self.check_keys_contain(crop_around_semi_trans_results.keys(),
                                       target_keys)
        assert self.check_ndim(crop_around_semi_trans_results['alpha'].shape)
        assert self.check_crop(crop_around_semi_trans_results['alpha'].shape,
                               crop_around_semi_trans_results['crop_bbox'])
        assert self.check_crop_around_semi(
            crop_around_semi_trans_results['alpha'])

        # test cropping when there is no unknown area
        fg = np.random.rand(240, 320, 3)
        bg = np.random.rand(240, 320, 3)
        merged = np.random.rand(240, 320, 3)
        ori_merged = merged.copy()
        alpha = np.zeros((240, 320, *self.ext_dim))
        results = dict(
            fg=fg, bg=bg, merged=merged, ori_merged=ori_merged, alpha=alpha)
        crop_around_semi_trans = CropAroundUnknown(
            keys, crop_sizes=[240], unknown_source='alpha')
        crop_around_semi_trans_results = crop_around_semi_trans(results)
        assert self.check_keys_contain(crop_around_semi_trans_results.keys(),
                                       target_keys)
        assert self.check_ndim(crop_around_semi_trans_results['alpha'].shape)
        assert self.check_crop(crop_around_semi_trans_results['alpha'].shape,
                               crop_around_semi_trans_results['crop_bbox'])

        repr_str = (
            crop_around_semi_trans.__class__.__name__ +
            f"(keys={keys}, crop_sizes={[(240, 240)]}, unknown_source='alpha',"
            " interpolations=['bilinear', 'bilinear', 'bilinear', 'bilinear', "
            "'bilinear'])")
        assert crop_around_semi_trans.__repr__() == repr_str


def test_adjust_gamma():
    """Test Gamma Correction.

    Adpted from
    # https://github.com/scikit-image/scikit-image/blob/7e4840bd9439d1dfb6beaf549998452c99f97fdd/skimage/exposure/tests/test_exposure.py#L534  # noqa
    """
    # Check that the shape is maintained.
    img = np.ones([1, 1])
    result = adjust_gamma(img, 1.5)
    assert img.shape == result.shape

    # Same image should be returned for gamma equal to one.
    image = np.random.uniform(0, 255, (8, 8))
    result = adjust_gamma(image, 1)
    np.testing.assert_array_equal(result, image)

    # White image should be returned for gamma equal to zero.
    image = np.random.uniform(0, 255, (8, 8))
    result = adjust_gamma(image, 0)
    dtype = image.dtype.type
    np.testing.assert_array_equal(result, dtype_range[dtype][1])

    # Verifying the output with expected results for gamma
    # correction with gamma equal to half.
    image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
    expected = np.array([[0, 31, 45, 55, 63, 71, 78, 84],
                         [90, 95, 100, 105, 110, 115, 119, 123],
                         [127, 131, 135, 139, 142, 146, 149, 153],
                         [156, 159, 162, 165, 168, 171, 174, 177],
                         [180, 183, 186, 188, 191, 194, 196, 199],
                         [201, 204, 206, 209, 211, 214, 216, 218],
                         [221, 223, 225, 228, 230, 232, 234, 236],
                         [238, 241, 243, 245, 247, 249, 251, 253]],
                        dtype=np.uint8)

    result = adjust_gamma(image, 0.5)
    np.testing.assert_array_equal(result, expected)

    # Verifying the output with expected results for gamma
    # correction with gamma equal to two.
    image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
    expected = np.array([[0, 0, 0, 0, 1, 1, 2, 3], [4, 5, 6, 7, 9, 10, 12, 14],
                         [16, 18, 20, 22, 25, 27, 30, 33],
                         [36, 39, 42, 45, 49, 52, 56, 60],
                         [64, 68, 72, 76, 81, 85, 90, 95],
                         [100, 105, 110, 116, 121, 127, 132, 138],
                         [144, 150, 156, 163, 169, 176, 182, 189],
                         [196, 203, 211, 218, 225, 233, 241, 249]],
                        dtype=np.uint8)

    result = adjust_gamma(image, 2)
    np.testing.assert_array_equal(result, expected)

    # Test invalid image input
    image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
    with pytest.raises(ValueError):
        adjust_gamma(image, -1)


def test_bbox_mask():
    # default config for random bbox mask
    cfg = dict(
        img_shape=(256, 256),
        max_bbox_shape=100,
        max_bbox_delta=10,
        min_margin=10)

    bbox = random_bbox(**cfg)
    mask_bbox = bbox2mask(cfg['img_shape'], bbox)
    assert mask_bbox.shape == (256, 256, 1)
    zero_area = np.sum((mask_bbox == 0).astype(np.uint8))
    ones_area = np.sum((mask_bbox == 1).astype(np.uint8))
    assert zero_area + ones_area == 256 * 256
    assert mask_bbox.dtype == np.uint8

    with pytest.raises(ValueError):
        cfg_ = cfg.copy()
        cfg_['max_bbox_shape'] = 300
        bbox = random_bbox(**cfg_)

    with pytest.raises(ValueError):
        cfg_ = cfg.copy()
        cfg_['max_bbox_delta'] = 300
        bbox = random_bbox(**cfg_)

    with pytest.raises(ValueError):
        cfg_ = cfg.copy()
        cfg_['max_bbox_shape'] = 254
        bbox = random_bbox(**cfg_)

    cfg_ = cfg.copy()
    cfg_['max_bbox_delta'] = 1
    bbox = random_bbox(**cfg_)
    mask_bbox = bbox2mask(cfg['img_shape'], bbox)
    assert mask_bbox.shape == (256, 256, 1)


def test_free_form_mask():
    img_shape = (256, 256, 3)
    for _ in range(10):
        mask = brush_stroke_mask(img_shape)
        assert mask.shape == (256, 256, 1)

    img_shape = (256, 256, 3)
    mask = brush_stroke_mask(img_shape, num_vertices=8)
    assert mask.shape == (256, 256, 1)
    zero_area = np.sum((mask == 0).astype(np.uint8))
    ones_area = np.sum((mask == 1).astype(np.uint8))
    assert zero_area + ones_area == 256 * 256
    assert mask.dtype == np.uint8

    img_shape = (256, 256, 3)
    mask = brush_stroke_mask(img_shape, brush_width=10)
    assert mask.shape == (256, 256, 1)

    with pytest.raises(TypeError):
        mask = brush_stroke_mask(img_shape, num_vertices=dict())

    with pytest.raises(TypeError):
        mask = brush_stroke_mask(img_shape, brush_width=dict())


def test_irregular_mask():
    img_shape = (256, 256)
    for _ in range(10):
        mask = get_irregular_mask(img_shape)
        assert mask.shape == (256, 256, 1)
        assert 0.15 < (np.sum(mask) / (img_shape[0] * img_shape[1])) < 0.50
        zero_area = np.sum((mask == 0).astype(np.uint8))
        ones_area = np.sum((mask == 1).astype(np.uint8))
        assert zero_area + ones_area == 256 * 256
        assert mask.dtype == np.uint8

    with pytest.raises(TypeError):
        mask = get_irregular_mask(img_shape, brush_width=dict())

    with pytest.raises(TypeError):
        mask = get_irregular_mask(img_shape, length_range=dict())

    with pytest.raises(TypeError):
        mask = get_irregular_mask(img_shape, num_vertices=dict())

    mask = get_irregular_mask(img_shape, brush_width=10)
    assert mask.shape == (256, 256, 1)

    mask = get_irregular_mask(img_shape, length_range=10)
    assert mask.shape == (256, 256, 1)

    mask = get_irregular_mask(img_shape, num_vertices=10)
    assert mask.shape == (256, 256, 1)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmedit.transforms import CropAroundCenter, CropAroundFg, CropAroundUnknown


class TestCrop:

    @staticmethod
    def check_crop(result_img_shape, result_bbox):
        crop_w = result_bbox[2] - result_bbox[0]
        """Check if the result_bbox is in correspond to result_img_shape."""
        crop_h = result_bbox[3] - result_bbox[1]
        crop_shape = (crop_h, crop_w)
        return result_img_shape == crop_shape

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
        trimap = np.random.rand(240, 320)
        alpha = np.random.rand(240, 320)

        # make sure there would be semi-transparent area
        trimap[128, 128] = 128
        results = dict(fg=fg, bg=bg, trimap=trimap, alpha=alpha)
        crop_around_center = CropAroundCenter(crop_size=320)
        crop_around_center_results = crop_around_center(results)
        assert self.check_keys_contain(crop_around_center_results.keys(),
                                       target_keys)
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
        alpha = np.zeros((240, 320))
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
        alpha = np.random.rand(240, 320)
        # make sure there would be unknown area
        alpha[120:160, 120:160] = 128
        results = dict(
            fg=fg, bg=bg, merged=merged, ori_merged=ori_merged, alpha=alpha)
        crop_around_semi_trans = CropAroundUnknown(
            keys, crop_sizes=[160], unknown_source='alpha')
        crop_around_semi_trans_results = crop_around_semi_trans(results)
        assert self.check_keys_contain(crop_around_semi_trans_results.keys(),
                                       target_keys)
        assert self.check_crop(crop_around_semi_trans_results['alpha'].shape,
                               crop_around_semi_trans_results['crop_bbox'])
        assert self.check_crop_around_semi(
            crop_around_semi_trans_results['alpha'])

        # test cropping when there is no unknown area
        fg = np.random.rand(240, 320, 3)
        bg = np.random.rand(240, 320, 3)
        merged = np.random.rand(240, 320, 3)
        ori_merged = merged.copy()
        alpha = np.zeros((240, 320))
        results = dict(
            fg=fg, bg=bg, merged=merged, ori_merged=ori_merged, alpha=alpha)
        crop_around_semi_trans = CropAroundUnknown(
            keys, crop_sizes=[240], unknown_source='alpha')
        crop_around_semi_trans_results = crop_around_semi_trans(results)
        assert self.check_keys_contain(crop_around_semi_trans_results.keys(),
                                       target_keys)
        assert self.check_crop(crop_around_semi_trans_results['alpha'].shape,
                               crop_around_semi_trans_results['crop_bbox'])

        repr_str = (
            crop_around_semi_trans.__class__.__name__ +
            f"(keys={keys}, crop_sizes={[(240, 240)]}, unknown_source='alpha',"
            " interpolations=['bilinear', 'bilinear', 'bilinear', 'bilinear', "
            "'bilinear'])")
        assert crop_around_semi_trans.__repr__() == repr_str

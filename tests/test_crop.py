import copy

import numpy as np
import pytest
from mmedit.datasets.pipelines import (Crop, CropAroundCenter,
                                       CropAroundSemiTransparent, ModCrop,
                                       PairedRandomCrop)


class TestAugmentations(object):

    @classmethod
    def setup_class(cls):
        cls.results = dict()
        cls.img_gt = np.random.rand(256, 128, 3).astype(np.float32)
        cls.img_lq = np.random.rand(64, 32, 3).astype(np.float32)

        cls.results = dict(
            lq=cls.img_lq,
            gt=cls.img_gt,
            scale=4,
            lq_path='fake_lq_path',
            gt_path='fake_gt_path')

        cls.results['img'] = np.random.rand(256, 256, 3).astype(np.float32)

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

    def test_crop(self):
        with pytest.raises(TypeError):
            Crop(['img'], (0.23, 0.1))

        # test center crop
        results = copy.deepcopy(self.results)
        center_crop = Crop(['img'], crop_size=(128, 128), random_crop=False)
        results = center_crop(results)
        assert results['img_crop_bbox'] == [64, 64, 128, 128]
        assert np.array_equal(self.results['img'][64:192, 64:192, :],
                              results['img'])

        # test random crop
        results = copy.deepcopy(self.results)
        random_crop = Crop(['img'], crop_size=(128, 128), random_crop=True)
        results = random_crop(results)
        assert 0 <= results['img_crop_bbox'][0] <= 128
        assert 0 <= results['img_crop_bbox'][1] <= 128
        assert results['img_crop_bbox'][2] == 128
        assert results['img_crop_bbox'][3] == 128

        # test random crop for lager size than the original shape
        results = copy.deepcopy(self.results)
        random_crop = Crop(['img'], crop_size=(512, 512), random_crop=True)
        results = random_crop(results)
        assert np.array_equal(self.results['img'], results['img'])
        assert str(random_crop) == random_crop.__class__.__name__ + \
            'keys={}, crop_size={}, random_crop={}'.\
            format(['img'], (512, 512), True)

    def test_crop_around_center(self):

        with pytest.raises(TypeError):
            CropAroundCenter(320.)
        with pytest.raises(AssertionError):
            CropAroundCenter((320, 320, 320))

        target_keys = ['fg', 'bg', 'alpha', 'trimap', 'img_shape', 'crop_bbox']

        img_shape = (240, 320)
        fg = np.random.rand(240, 320, 3)
        bg = np.random.rand(240, 320, 3)
        trimap = np.random.rand(240, 320)
        alpha = np.random.rand(240, 320)

        # make sure there would be semi-transparent area
        trimap[128, 128] = 128
        results = dict(
            fg=fg, bg=bg, trimap=trimap, alpha=alpha, img_shape=img_shape)
        crop_around_center = CropAroundCenter(crop_size=320)
        crop_around_center_results = crop_around_center(results)
        assert self.check_keys_contain(crop_around_center_results.keys(),
                                       target_keys)
        assert self.check_crop(crop_around_center_results['img_shape'],
                               crop_around_center_results['crop_bbox'])
        assert self.check_crop_around_semi(crop_around_center_results['alpha'])

        # make sure there would be semi-transparent area
        trimap[:, :] = 128
        results = dict(
            fg=fg, bg=bg, trimap=trimap, alpha=alpha, img_shape=img_shape)
        crop_around_center = CropAroundCenter(crop_size=200)
        crop_around_center_results = crop_around_center(results)
        assert self.check_keys_contain(crop_around_center_results.keys(),
                                       target_keys)
        assert self.check_crop(crop_around_center_results['img_shape'],
                               crop_around_center_results['crop_bbox'])
        assert self.check_crop_around_semi(crop_around_center_results['alpha'])

        repr_str = crop_around_center.__class__.__name__ + (
            f'(crop_size={(200, 200)})')
        assert repr(crop_around_center) == repr_str

    def test_crop_around_semi_transparent(self):
        with pytest.raises(TypeError):
            CropAroundSemiTransparent(320)
        with pytest.raises(TypeError):
            CropAroundSemiTransparent([320.])

        target_keys = [
            'fg', 'bg', 'merged', 'alpha', 'ori_merged', 'img_shape',
            'crop_bbox'
        ]

        fg = np.random.rand(240, 320, 3)
        bg = np.random.rand(240, 320, 3)
        merged = np.random.rand(240, 320, 3)
        alpha = np.random.rand(240, 320)
        # make sure there would be semi-transparent area
        alpha[128, 128] = 128
        results = dict(fg=fg, bg=bg, merged=merged, alpha=alpha)
        crop_around_semi_trans = CropAroundSemiTransparent(crop_sizes=[320])
        crop_around_semi_trans_results = crop_around_semi_trans(results)
        assert self.check_keys_contain(crop_around_semi_trans_results.keys(),
                                       target_keys)
        assert self.check_crop(crop_around_semi_trans_results['img_shape'],
                               crop_around_semi_trans_results['crop_bbox'])
        assert self.check_crop_around_semi(
            crop_around_semi_trans_results['alpha'])

        repr_str = crop_around_semi_trans.__class__.__name__ +\
            '(crop_sizes={}, interpolation={})'.format(
                [(320, 320)], 'bilinear')
        assert crop_around_semi_trans.__repr__() == repr_str

    def test_modcrop(self):
        # color image
        results = dict(gt=np.random.randn(257, 258, 3), scale=4)
        modcrop = ModCrop()
        results = modcrop(results)
        assert results['gt'].shape == (256, 256, 3)

        # gray image
        results = dict(gt=np.random.randn(257, 258), scale=4)
        results = modcrop(results)
        assert results['gt'].shape == (256, 256)

        # Wrong img ndim
        with pytest.raises(ValueError):
            results = dict(gt=np.random.randn(1, 257, 258, 3), scale=4)
            results = modcrop(results)

    def test_paired_random_crop(self):
        results = self.results.copy()
        pairedrandomcrop = PairedRandomCrop(128)
        results = pairedrandomcrop(results)
        assert results['gt'].shape == (128, 128, 3)
        assert results['lq'].shape == (32, 32, 3)

        # Scale mismatches. GT (h, w) is not {scale} multiplication of LQ's.
        with pytest.raises(ValueError):
            results = dict(
                gt=np.random.randn(128, 128, 3),
                lq=np.random.randn(64, 64, 3),
                scale=4,
                gt_path='fake_gt_path',
                lq_path='fake_lq_path')
            results = pairedrandomcrop(results)

        # LQ (h, w) is smaller than patch size.
        with pytest.raises(ValueError):
            results = dict(
                gt=np.random.randn(32, 32, 3),
                lq=np.random.randn(8, 8, 3),
                scale=4,
                gt_path='fake_gt_path',
                lq_path='fake_lq_path')
            results = pairedrandomcrop(results)

        assert repr(pairedrandomcrop) == (
            pairedrandomcrop.__class__.__name__ + f'(gt_patch_size=128)')

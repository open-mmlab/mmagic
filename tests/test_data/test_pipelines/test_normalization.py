# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmedit.datasets.pipelines import Normalize, RescaleToZeroOne


class TestAugmentations:

    @staticmethod
    def assert_img_equal(img, ref_img, ratio_thr=0.999):
        """Check if img and ref_img are matched approximately."""
        assert img.shape == ref_img.shape
        assert img.dtype == ref_img.dtype
        area = ref_img.shape[-1] * ref_img.shape[-2]
        diff = np.abs(img.astype('int32') - ref_img.astype('int32'))
        assert np.sum(diff <= 1) / float(area) > ratio_thr

    @staticmethod
    def check_keys_contain(result_keys, target_keys):
        """Check if all elements in target_keys is in result_keys."""
        return set(target_keys).issubset(set(result_keys))

    def check_normalize(self, origin_img, result_img, norm_cfg):
        """Check if the origin_img are normalized correctly into result_img in
        a given norm_cfg."""
        target_img = result_img.copy()
        target_img *= norm_cfg['std'][None, None, :]
        target_img += norm_cfg['mean'][None, None, :]
        if norm_cfg['to_rgb']:
            target_img = target_img[:, ::-1, ...].copy()
        self.assert_img_equal(origin_img, target_img)

    def test_normalize(self):
        with pytest.raises(TypeError):
            Normalize(['alpha'], dict(mean=[123.675, 116.28, 103.53]),
                      [58.395, 57.12, 57.375])

        with pytest.raises(TypeError):
            Normalize(['alpha'], [123.675, 116.28, 103.53],
                      dict(std=[58.395, 57.12, 57.375]))

        target_keys = ['merged', 'img_norm_cfg']

        merged = np.random.rand(240, 320, 3).astype(np.float32)
        results = dict(merged=merged)
        config = dict(
            keys=['merged'],
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=False)
        normalize = Normalize(**config)
        normalize_results = normalize(results)
        assert self.check_keys_contain(normalize_results.keys(), target_keys)
        self.check_normalize(merged, normalize_results['merged'],
                             normalize_results['img_norm_cfg'])

        merged = np.random.rand(240, 320, 3).astype(np.float32)
        results = dict(merged=merged)
        config = dict(
            keys=['merged'],
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)
        normalize = Normalize(**config)
        normalize_results = normalize(results)
        assert self.check_keys_contain(normalize_results.keys(), target_keys)
        self.check_normalize(merged, normalize_results['merged'],
                             normalize_results['img_norm_cfg'])

        assert normalize.__repr__() == (
            normalize.__class__.__name__ +
            f"(keys={ ['merged']}, mean={np.array([123.675, 116.28, 103.53])},"
            f' std={np.array([58.395, 57.12, 57.375])}, to_rgb=True)')

        # input is an image list
        merged = np.random.rand(240, 320, 3).astype(np.float32)
        merged_2 = np.random.rand(240, 320, 3).astype(np.float32)
        results = dict(merged=[merged, merged_2])
        config = dict(
            keys=['merged'],
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=False)
        normalize = Normalize(**config)
        normalize_results = normalize(results)
        assert self.check_keys_contain(normalize_results.keys(), target_keys)
        self.check_normalize(merged, normalize_results['merged'][0],
                             normalize_results['img_norm_cfg'])
        self.check_normalize(merged_2, normalize_results['merged'][1],
                             normalize_results['img_norm_cfg'])

        merged = np.random.rand(240, 320, 3).astype(np.float32)
        merged_2 = np.random.rand(240, 320, 3).astype(np.float32)
        results = dict(merged=[merged, merged_2])
        config = dict(
            keys=['merged'],
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)
        normalize = Normalize(**config)
        normalize_results = normalize(results)
        assert self.check_keys_contain(normalize_results.keys(), target_keys)
        self.check_normalize(merged, normalize_results['merged'][0],
                             normalize_results['img_norm_cfg'])
        self.check_normalize(merged_2, normalize_results['merged'][1],
                             normalize_results['img_norm_cfg'])

    def test_rescale_to_zero_one(self):
        target_keys = ['alpha']

        alpha = np.random.rand(240, 320).astype(np.float32)
        results = dict(alpha=alpha)
        rescale_to_zero_one = RescaleToZeroOne(keys=['alpha'])
        rescale_to_zero_one_results = rescale_to_zero_one(results)
        assert self.check_keys_contain(rescale_to_zero_one_results.keys(),
                                       target_keys)
        assert rescale_to_zero_one_results['alpha'].shape == (240, 320)
        np.testing.assert_almost_equal(rescale_to_zero_one_results['alpha'],
                                       alpha / 255.)
        assert repr(rescale_to_zero_one) == (
            rescale_to_zero_one.__class__.__name__ + f"(keys={['alpha']})")

        # input is image list
        alpha = np.random.rand(240, 320).astype(np.float32)
        alpha_2 = np.random.rand(240, 320).astype(np.float32)
        results = dict(alpha=[alpha, alpha_2])
        rescale_to_zero_one = RescaleToZeroOne(keys=['alpha'])
        rescale_to_zero_one_results = rescale_to_zero_one(results)
        assert rescale_to_zero_one_results['alpha'][0].shape == (240, 320)
        assert rescale_to_zero_one_results['alpha'][1].shape == (240, 320)
        np.testing.assert_almost_equal(rescale_to_zero_one_results['alpha'][0],
                                       alpha / 255.)
        np.testing.assert_almost_equal(rescale_to_zero_one_results['alpha'][1],
                                       alpha_2 / 255.)

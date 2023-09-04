# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmagic.datasets.transforms import MirrorSequence, TemporalReverse


class TestAugmentations:

    @classmethod
    def setup_class(cls):

        cls.results = dict()
        cls.gt = np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8)
        cls.img = np.random.randint(0, 256, (64, 32, 3), dtype=np.uint8)

        cls.results = dict(
            img=cls.img,
            gt=cls.gt,
            scale=4,
            img_path='fake_img_path',
            gt_path='fake_gt_path')

        cls.results['ori_img'] = np.random.randint(
            0, 256, (256, 256, 3), dtype=np.uint8)
        cls.results['mask'] = np.random.randint(
            0, 256, (256, 256, 1), dtype=np.uint8)
        # cls.results['img_tensor'] = torch.rand((3, 256, 256))
        # cls.results['mask_tensor'] = torch.zeros((1, 256, 256))
        # cls.results['mask_tensor'][:, 50:150, 40:140] = 1.

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

    def test_mirror_sequence(self):
        imgs = [np.random.rand(4, 4, 3) for _ in range(0, 5)]
        gts = [np.random.rand(16, 16, 3) for _ in range(0, 5)]

        target_keys = ['img', 'gt']
        mirror_sequence = MirrorSequence(keys=['img', 'gt'])
        results = dict(img=imgs, gt=gts)
        results = mirror_sequence(results)

        assert self.check_keys_contain(results.keys(), target_keys)
        for i in range(0, 5):
            np.testing.assert_almost_equal(results['img'][i],
                                           results['img'][-i - 1])
            np.testing.assert_almost_equal(results['gt'][i],
                                           results['gt'][-i - 1])

        assert repr(mirror_sequence) == mirror_sequence.__class__.__name__ + (
            "(keys=['img', 'gt'])")

        # each key should contain a list of nparray
        with pytest.raises(TypeError):
            results = dict(img=0, gt=gts)
            mirror_sequence(results)

    def test_temporal_reverse(self):
        img_lq1 = np.random.rand(4, 4, 3).astype(np.float32)
        img_lq2 = np.random.rand(4, 4, 3).astype(np.float32)
        img_gt = np.random.rand(8, 8, 3).astype(np.float32)
        results = dict(lq=[img_lq1, img_lq2], gt=[img_gt])

        target_keys = ['lq', 'gt', 'reverse']
        temporal_reverse = TemporalReverse(keys=['lq', 'gt'], reverse_ratio=1)
        results = temporal_reverse(results)
        assert self.check_keys_contain(results.keys(), target_keys)
        np.testing.assert_almost_equal(results['lq'][0], img_lq2)
        np.testing.assert_almost_equal(results['lq'][1], img_lq1)
        np.testing.assert_almost_equal(results['gt'][0], img_gt)

        assert repr(
            temporal_reverse) == temporal_reverse.__class__.__name__ + (
                f"(keys={['lq', 'gt']}, reverse_ratio=1)")

        results = dict(lq=[img_lq1, img_lq2], gt=[img_gt])
        temporal_reverse = TemporalReverse(keys=['lq', 'gt'], reverse_ratio=0)
        results = temporal_reverse(results)
        assert self.check_keys_contain(results.keys(), target_keys)
        np.testing.assert_almost_equal(results['lq'][0], img_lq1)
        np.testing.assert_almost_equal(results['lq'][1], img_lq2)
        np.testing.assert_almost_equal(results['gt'][0], img_gt)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

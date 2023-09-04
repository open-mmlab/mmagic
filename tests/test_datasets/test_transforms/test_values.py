# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmagic.datasets.transforms import CopyValues, SetValues


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

    def test_copy_value(self):

        with pytest.raises(AssertionError):
            CopyValues(src_keys='gt', dst_keys='img')
        with pytest.raises(ValueError):
            CopyValues(src_keys=['gt', 'gt'], dst_keys=['img'])

        results = {}
        results['gt'] = np.zeros((1)).astype(np.float32)

        copy_ = CopyValues(src_keys=['gt'], dst_keys=['img'])
        assert np.array_equal(copy_(results)['img'], results['gt'])
        assert repr(copy_) == copy_.__class__.__name__ + ("(src_keys=['gt'])"
                                                          "(dst_keys=['img'])")

    def test_set_value(self):

        with pytest.raises(AssertionError):
            CopyValues(src_keys='gt', dst_keys='img')
        with pytest.raises(ValueError):
            CopyValues(src_keys=['gt', 'gt'], dst_keys=['img'])

        results = {}
        results['gt'] = np.zeros((1)).astype(np.float32)
        dictionary = dict(a='b')

        set_values = SetValues(dictionary=dictionary)
        new_results = set_values(results)
        for key in dictionary.keys():
            assert new_results[key] == dictionary[key]
        assert repr(set_values) == (
            set_values.__class__.__name__ + f'(dictionary={dictionary})')


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

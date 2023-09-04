# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import pytest

from mmagic.datasets.transforms import (CenterCropLongEdge, Flip, NumpyPad,
                                        RandomCropLongEdge, RandomRotation,
                                        RandomTransposeHW, Resize)


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
    def check_flip(origin_img, result_img, flip_direction):
        """Check if the origin_img are flipped correctly into result_img in
        different flip_directions.

        Args:
            origin_img (np.ndarray): Original image.
            result_img (np.ndarray): Result image.
            flip_direction (str): Direction of flip.

        Returns:
            bool: Whether origin_img == result_img.
        """

        if flip_direction == 'horizontal':
            diff = result_img[:, ::-1] - origin_img
        else:
            diff = result_img[::-1, :] - origin_img

        return diff.max() < 1e-8

    @staticmethod
    def check_keys_contain(result_keys, target_keys):
        """Check if all elements in target_keys is in result_keys."""

        return set(target_keys).issubset(set(result_keys))

    @staticmethod
    def check_transposehw(origin_img, result_img):
        """Check if the origin_imgs are transposed correctly."""

        h, w, c = origin_img.shape
        for i in range(c):
            for j in range(h):
                for k in range(w):
                    if result_img[k, j, i] != origin_img[j, k, i]:  # noqa:E501
                        return False
        return True

    def test_flip(self):
        results = copy.deepcopy(self.results)

        trans = Flip(keys='img', flip_ratio=0)
        assert trans.keys == ['img']

        with pytest.raises(ValueError):
            Flip(keys=['img', 'gt'], direction='vertically')

        # horizontal
        np.random.seed(1)
        target_keys = ['img', 'gt', 'flip_infos']
        flip = Flip(keys=['img', 'gt'], flip_ratio=1, direction='horizontal')
        assert 'flip_infos' not in results
        results = flip(results)
        assert results['flip_infos'] == [
            dict(
                keys=['img', 'gt'], direction='horizontal', ratio=1, flip=True)
        ]
        assert self.check_keys_contain(results.keys(), target_keys)
        assert results['img'].shape == self.img.shape
        assert results['gt'].shape == self.gt.shape
        assert self.check_flip(self.img, results['img'],
                               results['flip_infos'][-1]['direction'])
        assert self.check_flip(self.gt, results['gt'],
                               results['flip_infos'][-1]['direction'])
        results = flip(results)
        assert results['flip_infos'] == [
            dict(
                keys=['img', 'gt'], direction='horizontal', ratio=1,
                flip=True),
            dict(
                keys=['img', 'gt'], direction='horizontal', ratio=1,
                flip=True),
        ]

        # vertical
        results = copy.deepcopy(self.results)
        flip = Flip(keys=['img', 'gt'], flip_ratio=1, direction='vertical')
        results = flip(results)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert results['img'].shape == self.img.shape
        assert results['gt'].shape == self.gt.shape
        assert self.check_flip(self.img, results['img'],
                               results['flip_infos'][-1]['direction'])
        assert self.check_flip(self.gt, results['gt'],
                               results['flip_infos'][-1]['direction'])
        assert repr(flip) == flip.__class__.__name__ + (
            f"(keys={['img', 'gt']}, flip_ratio=1, "
            f"direction={results['flip_infos'][-1]['direction']})")

        # flip a list
        # horizontal
        flip = Flip(keys=['img', 'gt'], flip_ratio=1, direction='horizontal')
        results = dict(
            img=[self.img, np.copy(self.img)],
            gt=[self.gt, np.copy(self.gt)],
            scale=4,
            img_path='fake_img_path',
            gt_path='fake_gt_path')
        flip_rlt = flip(copy.deepcopy(results))
        assert self.check_keys_contain(flip_rlt.keys(), target_keys)
        assert self.check_flip(self.img, flip_rlt['img'][0],
                               flip_rlt['flip_infos'][-1]['direction'])
        assert self.check_flip(self.gt, flip_rlt['gt'][0],
                               flip_rlt['flip_infos'][-1]['direction'])
        np.testing.assert_almost_equal(flip_rlt['gt'][0], flip_rlt['gt'][1])
        np.testing.assert_almost_equal(flip_rlt['img'][0], flip_rlt['img'][1])

        # vertical
        flip = Flip(keys=['img', 'gt'], flip_ratio=1, direction='vertical')
        flip_rlt = flip(copy.deepcopy(results))
        assert self.check_keys_contain(flip_rlt.keys(), target_keys)
        assert self.check_flip(self.img, flip_rlt['img'][0],
                               flip_rlt['flip_infos'][-1]['direction'])
        assert self.check_flip(self.gt, flip_rlt['gt'][0],
                               flip_rlt['flip_infos'][-1]['direction'])
        np.testing.assert_almost_equal(flip_rlt['gt'][0], flip_rlt['gt'][1])
        np.testing.assert_almost_equal(flip_rlt['img'][0], flip_rlt['img'][1])

        # no flip
        flip = Flip(keys=['img', 'gt'], flip_ratio=0, direction='vertical')
        results = flip(copy.deepcopy(results))
        assert self.check_keys_contain(results.keys(), target_keys)
        np.testing.assert_almost_equal(results['gt'][0], self.gt)
        np.testing.assert_almost_equal(results['img'][0], self.img)
        np.testing.assert_almost_equal(results['gt'][0], results['gt'][1])
        np.testing.assert_almost_equal(results['img'][0], results['img'][1])

    def test_random_rotation(self):
        with pytest.raises(ValueError):
            RandomRotation(None, degrees=-10.0)
        with pytest.raises(TypeError):
            RandomRotation(None, degrees=('0.0', '45.0'))

        target_keys = ['degrees']
        results = copy.deepcopy(self.results)

        random_rotation = RandomRotation(['ori_img'], degrees=(0, 45))
        random_rotation_results = random_rotation(results)
        assert self.check_keys_contain(random_rotation_results.keys(),
                                       target_keys)
        assert random_rotation_results['ori_img'].shape == (256, 256, 3)
        assert random_rotation_results['degrees'] == (0, 45)
        assert repr(random_rotation) == random_rotation.__class__.__name__ + (
            "(keys=['ori_img'], degrees=(0, 45))")

        # test single degree integer
        random_rotation = RandomRotation(['ori_img'], degrees=45)
        random_rotation_results = random_rotation(results)
        assert self.check_keys_contain(random_rotation_results.keys(),
                                       target_keys)
        assert random_rotation_results['ori_img'].shape == (256, 256, 3)
        assert random_rotation_results['degrees'] == (-45, 45)

        # test image dim == 2
        grey_scale_ori_img = np.random.rand(256, 256).astype(np.float32)
        results = dict(ori_img=grey_scale_ori_img.copy())
        random_rotation = RandomRotation(['ori_img'], degrees=(0, 45))
        random_rotation_results = random_rotation(results)
        assert self.check_keys_contain(random_rotation_results.keys(),
                                       target_keys)
        assert random_rotation_results['ori_img'].shape == (256, 256, 1)

    def test_random_transposehw(self):

        trans = RandomTransposeHW(keys='img', transpose_ratio=1)
        assert trans.keys == ['img']

        results = self.results.copy()
        target_keys = ['img', 'gt', 'transpose']
        transposehw = RandomTransposeHW(keys=['img', 'gt'], transpose_ratio=1)
        results = transposehw(results)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert self.check_transposehw(self.img, results['img'])
        assert self.check_transposehw(self.gt, results['gt'])
        assert results['img'].shape == (32, 64, 3)
        assert results['gt'].shape == (128, 256, 3)

        assert repr(transposehw) == transposehw.__class__.__name__ + (
            f"(keys={['img', 'gt']}, transpose_ratio=1)")

        # for image list
        ori_results = dict(
            img=[self.img, np.copy(self.img)],
            gt=[self.gt, np.copy(self.gt)],
            scale=4,
            img_path='fake_img_path',
            gt_path='fake_gt_path')
        target_keys = ['img', 'gt', 'transpose']
        transposehw = RandomTransposeHW(keys=['img', 'gt'], transpose_ratio=1)
        results = transposehw(ori_results.copy())
        assert self.check_keys_contain(results.keys(), target_keys)
        assert self.check_transposehw(self.img, results['img'][0])
        assert self.check_transposehw(self.gt, results['gt'][1])
        np.testing.assert_almost_equal(results['gt'][0], results['gt'][1])
        np.testing.assert_almost_equal(results['img'][0], results['img'][1])

        # no transpose
        target_keys = ['img', 'gt', 'transpose']
        transposehw = RandomTransposeHW(keys=['img', 'gt'], transpose_ratio=0)
        results = transposehw(ori_results.copy())
        assert self.check_keys_contain(results.keys(), target_keys)
        np.testing.assert_almost_equal(results['gt'][0], self.gt)
        np.testing.assert_almost_equal(results['img'][0], self.img)
        np.testing.assert_almost_equal(results['gt'][0], results['gt'][1])
        np.testing.assert_almost_equal(results['img'][0], results['img'][1])

    def test_resize(self):
        with pytest.raises(AssertionError):
            Resize([], scale=0.5)
        with pytest.raises(AssertionError):
            Resize(['gt_img'], size_factor=32, scale=0.5)
        with pytest.raises(AssertionError):
            Resize(['gt_img'], size_factor=32, keep_ratio=True)
        with pytest.raises(AssertionError):
            Resize(['gt_img'], max_size=32, size_factor=None)
        with pytest.raises(ValueError):
            Resize(['gt_img'], scale=-0.5)
        with pytest.raises(TypeError):
            Resize(['gt_img'], (0.4, 0.2))
        with pytest.raises(TypeError):
            Resize(['gt_img'], dict(test=None))

        target_keys = ['alpha']

        alpha = np.random.rand(240, 320).astype(np.float32)
        results = dict(alpha=alpha)
        resize = Resize(keys=['alpha'], size_factor=32, max_size=None)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert resize_results['alpha'].shape == (224, 320, 1)
        resize = Resize(keys=['alpha'], size_factor=32, max_size=320)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert resize_results['alpha'].shape == (224, 320, 1)

        resize = Resize(keys=['alpha'], size_factor=32, max_size=200)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert resize_results['alpha'].shape == (192, 192, 1)

        resize = Resize(['gt_img'], (-1, 200))
        assert resize.scale == (np.inf, 200)

        results = dict(gt_img=self.results['ori_img'].copy())
        resize_keep_ratio = Resize(['gt_img'], scale=0.5, keep_ratio=True)
        results = resize_keep_ratio(results)
        assert results['gt_img'].shape[:2] == (128, 128)
        assert results['scale_factor'] == 0.5

        results = dict(gt_img=self.results['ori_img'].copy())
        resize_keep_ratio = Resize(['gt_img'],
                                   scale=(128, 128),
                                   keep_ratio=False)
        results = resize_keep_ratio(results)
        assert results['gt_img'].shape[:2] == (128, 128)

        # test input with shape (256, 256)
        results = dict(
            gt_img=self.results['ori_img'][..., 0].copy(), alpha=alpha)
        resize = Resize(['gt_img', 'alpha'],
                        scale=(128, 128),
                        keep_ratio=False,
                        output_keys=['img', 'beta'])
        results = resize(results)
        assert results['gt_img'].shape == (256, 256)
        assert results['img'].shape == (128, 128, 1)
        assert results['alpha'].shape == (240, 320)
        assert results['beta'].shape == (128, 128, 1)

        name_ = str(resize_keep_ratio)
        assert name_ == resize_keep_ratio.__class__.__name__ + (
            "(keys=['gt_img'], output_keys=['gt_img'], "
            'scale=(128, 128), '
            f'keep_ratio={False}, size_factor=None, '
            'max_size=None, interpolation=bilinear)')

        # test input with shape (256, 256) + out keys and metainfo copy
        results = dict(
            gt_img=self.results['ori_img'][..., 0].copy(),
            alpha=alpha,
            ori_alpha_shape=[3, 3],
            gt_img_channel_order='rgb',
            alpha_color_type='grayscale')
        resize = Resize(['gt_img', 'alpha'],
                        scale=(128, 128),
                        keep_ratio=False,
                        output_keys=['img', 'beta'])
        results = resize(results)
        assert results['gt_img'].shape == (256, 256)
        assert results['img'].shape == (128, 128, 1)
        assert results['alpha'].shape == (240, 320)
        assert results['beta'].shape == (128, 128, 1)
        assert results['ori_beta_shape'] == [3, 3]
        assert results['img_channel_order'] == 'rgb'
        assert results['beta_color_type'] == 'grayscale'

        name_ = str(resize_keep_ratio)
        assert name_ == resize_keep_ratio.__class__.__name__ + (
            "(keys=['gt_img'], output_keys=['gt_img'], "
            'scale=(128, 128), '
            f'keep_ratio={False}, size_factor=None, '
            'max_size=None, interpolation=bilinear)')


def test_random_long_edge_crop():
    results = dict(img=np.random.rand(256, 128, 3).astype(np.float32))
    crop = RandomCropLongEdge(['img'])
    results = crop(results)
    assert results['img'].shape == (128, 128, 3)

    repr_str = crop.__class__.__name__
    repr_str += (f'(keys={crop.keys})')

    assert str(crop) == repr_str


def test_center_long_edge_crop():
    results = dict(img=np.random.rand(256, 128, 3).astype(np.float32))
    crop = CenterCropLongEdge(['img'])
    results = crop(results)
    assert results['img'].shape == (128, 128, 3)


def test_numpy_pad():
    results = dict(img=np.zeros((5, 5, 1)))

    pad = NumpyPad(['img'], ((2, 2), (0, 0), (0, 0)))
    results = pad(results)
    assert results['img'].shape == (9, 5, 1)

    repr_str = pad.__class__.__name__
    repr_str += (
        f'(keys={pad.keys}, padding={pad.padding}, kwargs={pad.kwargs})')

    assert str(pad) == repr_str


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

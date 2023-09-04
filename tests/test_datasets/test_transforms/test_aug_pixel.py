# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import pytest

from mmagic.datasets.transforms import (BinarizeImage, Clip, ColorJitter,
                                        RandomAffine, RandomMaskDilation,
                                        UnsharpMasking)


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
    def check_keys_contain(result_keys, target_keys):
        """Check if all elements in target_keys is in result_keys."""

        return set(target_keys).issubset(set(result_keys))

    def test_binarize(self):

        mask_ = np.zeros((5, 5, 1))
        mask_[2, 2, :] = 0.6
        gt_mask = mask_.copy()
        gt_mask[2, 2, :] = 1.
        results = dict(mask=mask_.copy())
        binarize = BinarizeImage(['mask'], 0.5, dtype=np.float32)
        results = binarize(results)
        assert np.array_equal(results['mask'], gt_mask.astype(np.float32))

        results = dict(mask=mask_.copy())
        binarize = BinarizeImage(['mask'], 0.5)
        results = binarize(results)
        assert np.array_equal(results['mask'], gt_mask.astype(np.uint8))
        assert repr(binarize) == (
            binarize.__class__.__name__ + f"(keys={['mask']}, binary_thr=0.5, "
            f'a_min=0, a_max=1, dtype={np.uint8})')

        results = dict(mask=mask_.copy())
        binarize = BinarizeImage(['mask'], 0.5, a_max=3, a_min=1)
        results = binarize(results)
        assert np.array_equal(results['mask'],
                              gt_mask.astype(np.uint8) * 2 + 1)
        assert repr(binarize) == (
            binarize.__class__.__name__ + f"(keys={['mask']}, binary_thr=0.5, "
            f'a_min=1, a_max=3, dtype={np.uint8})')

    def test_clip(self):
        results = {}

        # clip
        results['gt'] = (1.2 - 0.1) * self.results['gt']
        model = Clip(keys=['gt'])
        assert np.array_equal(model(results)['gt'], results['gt'].clip(0, 255))

    def test_color_jitter(self):

        results = copy.deepcopy(self.results)
        results['gt'] = (results['gt'] * 255).astype(np.uint8)
        results['img'] = [results['gt'], results['gt']]

        target_keys = ['gt', 'img']

        color_jitter = ColorJitter(
            keys=['gt', 'img'],
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5)
        color_jitter_results = color_jitter(results)

        assert self.check_keys_contain(color_jitter_results.keys(),
                                       target_keys)
        assert color_jitter_results['gt'].shape == self.gt.shape
        color_jitter = ColorJitter(
            keys=['gt', 'img'],
            channel_order='bgr',
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5)
        color_jitter_results = color_jitter(results)
        assert self.check_keys_contain(color_jitter_results.keys(),
                                       target_keys)
        assert color_jitter_results['gt'].shape == self.gt.shape
        assert np.abs(color_jitter_results['gt'] - self.gt.shape).mean() > 0

        assert repr(color_jitter) == color_jitter.__class__.__name__ + (
            f'(keys={color_jitter.keys}, '
            f'channel_order={color_jitter.channel_order}, '
            f'brightness={color_jitter._transform.brightness}, '
            f'contrast={color_jitter._transform.contrast}, '
            f'saturation={color_jitter._transform.saturation}, '
            f'hue={color_jitter._transform.hue})')

        with pytest.raises(AssertionError):
            color_jitter = ColorJitter(
                keys=['gt', 'img'],
                channel_order='bgr',
                to_rgb=True,
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.5)

    def test_random_affine(self):
        with pytest.raises(AssertionError):
            RandomAffine(None, -1)

        with pytest.raises(AssertionError):
            RandomAffine(None, 0, translate='Not a tuple')

        with pytest.raises(AssertionError):
            RandomAffine(None, 0, translate=(0, 0, 0))

        with pytest.raises(AssertionError):
            RandomAffine(None, 0, translate=(0, 2))

        with pytest.raises(AssertionError):
            RandomAffine(None, 0, scale='Not a tuple')

        with pytest.raises(AssertionError):
            RandomAffine(None, 0, scale=(0.8, 1., 1.2))

        with pytest.raises(AssertionError):
            RandomAffine(None, 0, scale=(-0.8, 1.))

        with pytest.raises(AssertionError):
            RandomAffine(None, 0, shear=-1)

        with pytest.raises(AssertionError):
            RandomAffine(None, 0, shear=(0, 1, 2))

        with pytest.raises(AssertionError):
            RandomAffine(None, 0, flip_ratio='Not a float')

        target_keys = ['fg', 'alpha']

        # Test identical transformation
        alpha = np.random.rand(4, 4).astype(np.float32)
        fg = np.random.rand(4, 4).astype(np.float32)
        results = dict(alpha=alpha, fg=fg)
        random_affine = RandomAffine(['fg', 'alpha'],
                                     degrees=0,
                                     flip_ratio=0.0)
        random_affine_results = random_affine(results)
        assert np.allclose(alpha, random_affine_results['alpha'])
        assert np.allclose(fg, random_affine_results['fg'])

        # Test flip in both direction
        fg = np.random.rand(4, 4).astype(np.float32)
        alpha = np.random.rand(4, 4).astype(np.float32)
        results = dict(alpha=alpha, fg=fg)
        random_affine = RandomAffine(['fg', 'alpha'],
                                     degrees=0,
                                     flip_ratio=1.0)
        random_affine_results = random_affine(results)
        assert np.allclose(alpha[::-1, ::-1], random_affine_results['alpha'])
        assert np.allclose(fg[::-1, ::-1], random_affine_results['fg'])

        # test random affine with different valid setting combinations
        # only shape are tested
        alpha = np.random.rand(240, 320, 1).astype(np.float32)
        fg = np.random.rand(240, 320, 3).astype(np.float32)
        results = dict(alpha=alpha, fg=fg)
        random_affine = RandomAffine(['fg', 'alpha'],
                                     degrees=30,
                                     translate=(0, 1),
                                     shear=(10, 20),
                                     flip_ratio=0.5)
        random_affine_results = random_affine(results)
        assert self.check_keys_contain(random_affine_results.keys(),
                                       target_keys)
        assert random_affine_results['fg'].shape == (240, 320, 3)
        assert random_affine_results['alpha'].shape == (240, 320, 1)

        alpha = np.random.rand(240, 320, 1).astype(np.float32)
        fg = np.random.rand(240, 320, 3).astype(np.float32)
        results = dict(alpha=alpha, fg=fg)
        random_affine = RandomAffine(['fg', 'alpha'],
                                     degrees=(-30, 30),
                                     scale=(0.8, 1.25),
                                     shear=10,
                                     flip_ratio=0.5)
        random_affine_results = random_affine(results)
        assert self.check_keys_contain(random_affine_results.keys(),
                                       target_keys)
        assert random_affine_results['fg'].shape == (240, 320, 3)
        assert random_affine_results['alpha'].shape == (240, 320, 1)

        alpha = np.random.rand(240, 320, 1).astype(np.float32)
        fg = np.random.rand(240, 320, 3).astype(np.float32)
        results = dict(alpha=alpha, fg=fg)
        random_affine = RandomAffine(['fg', 'alpha'], degrees=30)
        random_affine_results = random_affine(results)
        assert self.check_keys_contain(random_affine_results.keys(),
                                       target_keys)
        assert random_affine_results['fg'].shape == (240, 320, 3)
        assert random_affine_results['alpha'].shape == (240, 320, 1)

        assert repr(random_affine) == random_affine.__class__.__name__ + (
            f'(keys={target_keys}, degrees={(-30, 30)}, '
            f'translate={None}, scale={None}, '
            f'shear={None}, flip_ratio={0})')

    def test_random_dilation(self):
        mask = np.zeros((3, 3, 1), dtype=np.float32)
        mask[1, 1] = 1
        gt_mask = np.ones_like(mask)
        results = dict(mask=mask.copy())
        dilation = RandomMaskDilation(['mask'],
                                      binary_thr=0.5,
                                      kernel_min=3,
                                      kernel_max=3)
        results = dilation(results)
        assert np.array_equal(results['mask'], gt_mask)
        assert results['mask_dilate_kernel_size'] == 3
        assert str(dilation) == (
            dilation.__class__.__name__ +
            f"(keys={['mask']}, kernel_min=3, kernel_max=3)")

    def test_unsharp_masking(self):
        results = {}

        unsharp_masking = UnsharpMasking(
            kernel_size=15, sigma=0, weight=0.5, threshold=10, keys=['gt'])

        # single image
        results['gt'] = np.zeros((8, 8, 3)).astype(np.float32)
        results = unsharp_masking(results)
        assert isinstance(results['gt_unsharp'], np.ndarray)

        # sequence of images
        results['gt'] = [np.zeros((8, 8, 3)).astype(np.float32)] * 2
        results = unsharp_masking(results)
        assert isinstance(results['gt_unsharp'], list)

        assert repr(unsharp_masking) == unsharp_masking.__class__.__name__ + (
            "(keys=['gt'], kernel_size=15, sigma=0, weight=0.5, threshold=10)")

        # kernel_size must be odd
        with pytest.raises(ValueError):
            unsharp_masking = UnsharpMasking(
                kernel_size=10, sigma=0, weight=0.5, threshold=10, keys=['gt'])


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

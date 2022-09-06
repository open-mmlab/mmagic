# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import numpy as np
import pytest
import torch

# yapf: disable
from mmedit.datasets.pipelines import (BinarizeImage, ColorJitter, CopyValues,
                                       Flip, GenerateFrameIndices,
                                       GenerateFrameIndiceswithPadding,
                                       GenerateSegmentIndices, MirrorSequence,
                                       Pad, Quantize, RandomAffine,
                                       RandomJitter, RandomMaskDilation,
                                       RandomTransposeHW, Resize,
                                       TemporalReverse, UnsharpMasking)
from mmedit.datasets.pipelines.augmentation import RandomRotation


class TestAugmentations:

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
        cls.results['mask'] = np.random.rand(256, 256, 1).astype(np.float32)
        cls.results['img_tensor'] = torch.rand((3, 256, 256))
        cls.results['mask_tensor'] = torch.zeros((1, 256, 256))
        cls.results['mask_tensor'][:, 50:150, 40:140] = 1.

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

    @staticmethod
    def check_flip(origin_img, result_img, flip_type):
        """Check if the origin_img are flipped correctly into result_img in
        different flip_types."""
        h, w, c = origin_img.shape
        if flip_type == 'horizontal':
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        if result_img[i, j, k] != origin_img[i, w - 1 - j, k]:
                            return False
        else:
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        if result_img[i, j, k] != origin_img[h - 1 - i, j, k]:
                            return False
        return True

    def test_binarize(self):
        mask_ = np.zeros((5, 5, 1))
        mask_[2, 2, :] = 0.6
        gt_mask = mask_.copy()
        gt_mask[2, 2, :] = 1.
        results = dict(mask=mask_.copy())
        binarize = BinarizeImage(['mask'], 0.5, to_int=False)
        results = binarize(results)
        assert np.array_equal(results['mask'], gt_mask.astype(np.float32))

        results = dict(mask=mask_.copy())
        binarize = BinarizeImage(['mask'], 0.5, to_int=True)
        results = binarize(results)
        assert np.array_equal(results['mask'], gt_mask.astype(np.int32))
        assert str(binarize) == (
            binarize.__class__.__name__ +
            f"(keys={['mask']}, binary_thr=0.5, to_int=True)")

    def test_flip(self):
        results = copy.deepcopy(self.results)

        with pytest.raises(ValueError):
            Flip(keys=['lq', 'gt'], direction='vertically')

        # horizontal
        np.random.seed(1)
        target_keys = ['lq', 'gt', 'flip', 'flip_direction']
        flip = Flip(keys=['lq', 'gt'], flip_ratio=1, direction='horizontal')
        results = flip(results)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert self.check_flip(self.img_lq, results['lq'],
                               results['flip_direction'])
        assert self.check_flip(self.img_gt, results['gt'],
                               results['flip_direction'])
        assert results['lq'].shape == self.img_lq.shape
        assert results['gt'].shape == self.img_gt.shape

        # vertical
        results = copy.deepcopy(self.results)
        flip = Flip(keys=['lq', 'gt'], flip_ratio=1, direction='vertical')
        results = flip(results)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert self.check_flip(self.img_lq, results['lq'],
                               results['flip_direction'])
        assert self.check_flip(self.img_gt, results['gt'],
                               results['flip_direction'])
        assert results['lq'].shape == self.img_lq.shape
        assert results['gt'].shape == self.img_gt.shape
        assert repr(flip) == flip.__class__.__name__ + (
            f"(keys={['lq', 'gt']}, flip_ratio=1, "
            f"direction={results['flip_direction']})")

        # flip a list
        # horizontal
        flip = Flip(keys=['lq', 'gt'], flip_ratio=1, direction='horizontal')
        results = dict(
            lq=[self.img_lq, np.copy(self.img_lq)],
            gt=[self.img_gt, np.copy(self.img_gt)],
            scale=4,
            lq_path='fake_lq_path',
            gt_path='fake_gt_path')
        flip_rlt = flip(copy.deepcopy(results))
        assert self.check_keys_contain(flip_rlt.keys(), target_keys)
        assert self.check_flip(self.img_lq, flip_rlt['lq'][0],
                               flip_rlt['flip_direction'])
        assert self.check_flip(self.img_gt, flip_rlt['gt'][0],
                               flip_rlt['flip_direction'])
        np.testing.assert_almost_equal(flip_rlt['gt'][0], flip_rlt['gt'][1])
        np.testing.assert_almost_equal(flip_rlt['lq'][0], flip_rlt['lq'][1])

        # vertical
        flip = Flip(keys=['lq', 'gt'], flip_ratio=1, direction='vertical')
        flip_rlt = flip(copy.deepcopy(results))
        assert self.check_keys_contain(flip_rlt.keys(), target_keys)
        assert self.check_flip(self.img_lq, flip_rlt['lq'][0],
                               flip_rlt['flip_direction'])
        assert self.check_flip(self.img_gt, flip_rlt['gt'][0],
                               flip_rlt['flip_direction'])
        np.testing.assert_almost_equal(flip_rlt['gt'][0], flip_rlt['gt'][1])
        np.testing.assert_almost_equal(flip_rlt['lq'][0], flip_rlt['lq'][1])

        # no flip
        flip = Flip(keys=['lq', 'gt'], flip_ratio=0, direction='vertical')
        results = flip(copy.deepcopy(results))
        assert self.check_keys_contain(results.keys(), target_keys)
        np.testing.assert_almost_equal(results['gt'][0], self.img_gt)
        np.testing.assert_almost_equal(results['lq'][0], self.img_lq)
        np.testing.assert_almost_equal(results['gt'][0], results['gt'][1])
        np.testing.assert_almost_equal(results['lq'][0], results['lq'][1])

    def test_pad(self):
        target_keys = ['alpha']

        alpha = np.random.rand(319, 321).astype(np.float32)
        results = dict(alpha=alpha)
        pad = Pad(keys=['alpha'], ds_factor=32, mode='constant')
        pad_results = pad(results)
        assert self.check_keys_contain(pad_results.keys(), target_keys)
        assert pad_results['alpha'].shape == (320, 352)
        assert self.check_pad(alpha, results['alpha'], 'constant')

        alpha = np.random.rand(319, 321).astype(np.float32)
        results = dict(alpha=alpha)
        pad = Pad(keys=['alpha'], ds_factor=32, mode='reflect')
        pad_results = pad(results)
        assert self.check_keys_contain(pad_results.keys(), target_keys)
        assert pad_results['alpha'].shape == (320, 352)
        assert self.check_pad(alpha, results['alpha'], 'reflect')

        alpha = np.random.rand(320, 320).astype(np.float32)
        results = dict(alpha=alpha)
        pad = Pad(keys=['alpha'], ds_factor=32, mode='reflect')
        pad_results = pad(results)
        assert self.check_keys_contain(pad_results.keys(), target_keys)
        assert pad_results['alpha'].shape == (320, 320)
        assert self.check_pad(alpha, results['alpha'], 'reflect')

        assert repr(pad) == pad.__class__.__name__ + (
            f"(keys={['alpha']}, ds_factor=32, mode={'reflect'})")

    @staticmethod
    def check_pad(origin_img, result_img, mode, ds_factor=32):
        """Check if the origin_img is padded correctly.

        Supported modes for checking are 'constant' (with 'constant_values' of
        0) and 'reflect'. Supported images should be 2 dimensional.
        """
        if mode not in ['constant', 'reflect']:
            raise NotImplementedError(
                f'Pad checking of mode {mode} is not implemented.')
        assert len(origin_img.shape) == 2, 'Image should be 2 dimensional.'

        h, w = origin_img.shape
        new_h = ds_factor * (h - 1) // ds_factor + 1
        new_w = ds_factor * (w - 1) // ds_factor + 1

        # check the bottom rectangle
        for i in range(h, new_h):
            for j in range(0, w):
                target = origin_img[h - i, j] if mode == 'reflect' else 0
                if result_img[i, j] != target:
                    return False

        # check the right rectangle
        for i in range(0, h):
            for j in range(w, new_w):
                target = origin_img[i, w - j] if mode == 'reflect' else 0
                if result_img[i, j] != target:
                    return False

        # check the bottom right rectangle
        for i in range(h, new_h):
            for j in range(w, new_w):
                target = origin_img[h - i, w - j] if mode == 'reflect' else 0
                if result_img[i, j] != target:
                    return False

        return True

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
                                     degrees=0, flip_ratio=0.0)
        random_affine_results = random_affine(results)
        assert np.allclose(alpha, random_affine_results['alpha'])
        assert np.allclose(fg, random_affine_results['fg'])

        # Test flip in both direction
        alpha = np.random.rand(4, 4).astype(np.float32)
        fg = np.random.rand(4, 4).astype(np.float32)
        results = dict(alpha=alpha, fg=fg)
        random_affine = RandomAffine(['fg', 'alpha'],
                                     degrees=0, flip_ratio=1.0)
        random_affine_results = random_affine(results)
        assert np.allclose(alpha[::-1, ::-1], random_affine_results['alpha'])
        assert np.allclose(fg[::-1, ::-1], random_affine_results['fg'])

        # test random affine with different valid setting combinations
        # only shape are tested
        alpha = np.random.rand(240, 320).astype(np.float32)
        fg = np.random.rand(240, 320).astype(np.float32)
        results = dict(alpha=alpha, fg=fg)
        random_affine = RandomAffine(['fg', 'alpha'],
                                     degrees=30,
                                     translate=(0, 1),
                                     shear=(10, 20),
                                     flip_ratio=0.5)
        random_affine_results = random_affine(results)
        assert self.check_keys_contain(random_affine_results.keys(),
                                       target_keys)
        assert random_affine_results['fg'].shape == (240, 320)
        assert random_affine_results['alpha'].shape == (240, 320)

        alpha = np.random.rand(240, 320).astype(np.float32)
        fg = np.random.rand(240, 320).astype(np.float32)
        results = dict(alpha=alpha, fg=fg)
        random_affine = RandomAffine(['fg', 'alpha'],
                                     degrees=(-30, 30),
                                     scale=(0.8, 1.25),
                                     shear=10,
                                     flip_ratio=0.5)
        random_affine_results = random_affine(results)
        assert self.check_keys_contain(random_affine_results.keys(),
                                       target_keys)
        assert random_affine_results['fg'].shape == (240, 320)
        assert random_affine_results['alpha'].shape == (240, 320)

        alpha = np.random.rand(240, 320).astype(np.float32)
        fg = np.random.rand(240, 320).astype(np.float32)
        results = dict(alpha=alpha, fg=fg)
        random_affine = RandomAffine(['fg', 'alpha'], degrees=30)
        random_affine_results = random_affine(results)
        assert self.check_keys_contain(random_affine_results.keys(),
                                       target_keys)
        assert random_affine_results['fg'].shape == (240, 320)
        assert random_affine_results['alpha'].shape == (240, 320)

        assert repr(random_affine) == random_affine.__class__.__name__ + (
            f'(keys={target_keys}, degrees={(-30, 30)}, '
            f'translate={None}, scale={None}, '
            f'shear={None}, flip_ratio={0})')

    def test_random_jitter(self):
        with pytest.raises(AssertionError):
            RandomJitter(-40)

        with pytest.raises(AssertionError):
            RandomJitter((-40, 40, 40))

        target_keys = ['fg']

        fg = np.random.rand(240, 320, 3).astype(np.float32)
        alpha = np.random.rand(240, 320).astype(np.float32)
        results = dict(fg=fg.copy(), alpha=alpha)
        random_jitter = RandomJitter(40)
        random_jitter_results = random_jitter(results)
        assert self.check_keys_contain(random_jitter_results.keys(),
                                       target_keys)
        assert random_jitter_results['fg'].shape == (240, 320, 3)

        fg = np.random.rand(240, 320, 3).astype(np.float32)
        alpha = np.random.rand(240, 320).astype(np.float32)
        results = dict(fg=fg.copy(), alpha=alpha)
        random_jitter = RandomJitter((-50, 50))
        random_jitter_results = random_jitter(results)
        assert self.check_keys_contain(random_jitter_results.keys(),
                                       target_keys)
        assert random_jitter_results['fg'].shape == (240, 320, 3)

        assert repr(random_jitter) == random_jitter.__class__.__name__ + (
            'hue_range=(-50, 50)')

    def test_color_jitter(self):

        results = copy.deepcopy(self.results)
        results['gt'] = (results['gt'] * 255).astype(np.uint8)
        results['lq'] = [results['gt'], results['gt']]

        target_keys = ['gt', 'lq']

        color_jitter = ColorJitter(
            keys=['gt', 'lq'],
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5)
        color_jitter_results = color_jitter(results)
        assert self.check_keys_contain(color_jitter_results.keys(),
                                       target_keys)
        assert color_jitter_results['gt'].shape == self.img_gt.shape
        color_jitter = ColorJitter(
            keys=['gt', 'lq'],
            channel_order='bgr',
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5)
        color_jitter_results = color_jitter(results)
        assert self.check_keys_contain(color_jitter_results.keys(),
                                       target_keys)
        assert color_jitter_results['gt'].shape == self.img_gt.shape
        assert np.abs(color_jitter_results['gt']-self.img_gt.shape).mean() > 0

        assert repr(color_jitter) == color_jitter.__class__.__name__ + (
            f'(keys={color_jitter.keys}, '
            f'channel_order={color_jitter.channel_order}, '
            f'brightness={color_jitter.transform.brightness}, '
            f'contrast={color_jitter.transform.contrast}, '
            f'saturation={color_jitter.transform.saturation}, '
            f'hue={color_jitter.transform.hue})')

        with pytest.raises(AssertionError):
            color_jitter = ColorJitter(
                keys=['gt', 'lq'],
                channel_order='bgr',
                to_rgb=True,
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.5)

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

    def test_transposehw(self):
        results = self.results.copy()
        target_keys = ['lq', 'gt', 'transpose']
        transposehw = RandomTransposeHW(keys=['lq', 'gt'], transpose_ratio=1)
        results = transposehw(results)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert self.check_transposehw(self.img_lq, results['lq'])
        assert self.check_transposehw(self.img_gt, results['gt'])
        assert results['lq'].shape == (32, 64, 3)
        assert results['gt'].shape == (128, 256, 3)

        assert repr(transposehw) == transposehw.__class__.__name__ + (
            f"(keys={['lq', 'gt']}, transpose_ratio=1)")

        # for image list
        ori_results = dict(
            lq=[self.img_lq, np.copy(self.img_lq)],
            gt=[self.img_gt, np.copy(self.img_gt)],
            scale=4,
            lq_path='fake_lq_path',
            gt_path='fake_gt_path')
        target_keys = ['lq', 'gt', 'transpose']
        transposehw = RandomTransposeHW(keys=['lq', 'gt'], transpose_ratio=1)
        results = transposehw(ori_results.copy())
        assert self.check_keys_contain(results.keys(), target_keys)
        assert self.check_transposehw(self.img_lq, results['lq'][0])
        assert self.check_transposehw(self.img_gt, results['gt'][1])
        np.testing.assert_almost_equal(results['gt'][0], results['gt'][1])
        np.testing.assert_almost_equal(results['lq'][0], results['lq'][1])

        # no transpose
        target_keys = ['lq', 'gt', 'transpose']
        transposehw = RandomTransposeHW(keys=['lq', 'gt'], transpose_ratio=0)
        results = transposehw(ori_results.copy())
        assert self.check_keys_contain(results.keys(), target_keys)
        np.testing.assert_almost_equal(results['gt'][0], self.img_gt)
        np.testing.assert_almost_equal(results['lq'][0], self.img_lq)
        np.testing.assert_almost_equal(results['gt'][0], results['gt'][1])
        np.testing.assert_almost_equal(results['lq'][0], results['lq'][1])

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

        results = dict(gt_img=self.results['img'].copy())
        resize_keep_ratio = Resize(['gt_img'], scale=0.5, keep_ratio=True)
        results = resize_keep_ratio(results)
        assert results['gt_img'].shape[:2] == (128, 128)
        assert results['scale_factor'] == 0.5

        results = dict(gt_img=self.results['img'].copy())
        resize_keep_ratio = Resize(['gt_img'],
                                   scale=(128, 128),
                                   keep_ratio=False)
        results = resize_keep_ratio(results)
        assert results['gt_img'].shape[:2] == (128, 128)

        # test input with shape (256, 256)
        results = dict(gt_img=self.results['img'][..., 0].copy(), alpha=alpha)
        resize = Resize(['gt_img', 'alpha'],
                        scale=(128, 128),
                        keep_ratio=False,
                        output_keys=['lq_img', 'beta'])
        results = resize(results)
        assert results['gt_img'].shape == (256, 256)
        assert results['lq_img'].shape == (128, 128, 1)
        assert results['alpha'].shape == (240, 320)
        assert results['beta'].shape == (128, 128, 1)

        name_ = str(resize_keep_ratio)
        assert name_ == resize_keep_ratio.__class__.__name__ + (
            "(keys=['gt_img'], output_keys=['gt_img'], "
            'scale=(128, 128), '
            f'keep_ratio={False}, size_factor=None, '
            'max_size=None, interpolation=bilinear)')

    def test_random_rotation(self):
        with pytest.raises(ValueError):
            RandomRotation(None, degrees=-10.0)
        with pytest.raises(TypeError):
            RandomRotation(None, degrees=('0.0', '45.0'))

        target_keys = ['degrees']
        results = copy.deepcopy(self.results)

        random_rotation = RandomRotation(['img'], degrees=(0, 45))
        random_rotation_results = random_rotation(results)
        assert self.check_keys_contain(
            random_rotation_results.keys(), target_keys)
        assert random_rotation_results['img'].shape == (256, 256, 3)
        assert random_rotation_results['degrees'] == (0, 45)
        assert repr(random_rotation) == random_rotation.__class__.__name__ + (
            "(keys=['img'], degrees=(0, 45))")

        # test single degree integer
        random_rotation = RandomRotation(['img'], degrees=45)
        random_rotation_results = random_rotation(results)
        assert self.check_keys_contain(
            random_rotation_results.keys(), target_keys)
        assert random_rotation_results['img'].shape == (256, 256, 3)
        assert random_rotation_results['degrees'] == (-45, 45)

        # test image dim == 2
        grey_scale_img = np.random.rand(256, 256).astype(np.float32)
        results = dict(img=grey_scale_img.copy())
        random_rotation = RandomRotation(['img'], degrees=(0, 45))
        random_rotation_results = random_rotation(results)
        assert self.check_keys_contain(
            random_rotation_results.keys(), target_keys)
        assert random_rotation_results['img'].shape == (256, 256, 1)

    def test_frame_index_generation_with_padding(self):
        with pytest.raises(ValueError):
            # Wrong padding mode
            GenerateFrameIndiceswithPadding(padding='fake')

        results = dict(
            lq_path='fake_lq_root',
            gt_path='fake_gt_root',
            key=osp.join('000', '00000000'),
            max_frame_num=100,
            num_input_frames=5)
        target_keys = ['lq_path', 'gt_path', 'key']
        replicate_idx = [0, 0, 0, 1, 2]
        reflection_idx = [2, 1, 0, 1, 2]
        reflection_circle_idx = [4, 3, 0, 1, 2]
        circle_idx = [3, 4, 0, 1, 2]

        # replicate
        lq_paths = [osp.join('fake_lq_root', '000',
                             f'{v:08d}.png') for v in replicate_idx]
        gt_paths = [osp.join('fake_gt_root', '000', '00000000.png')]
        frame_index_generator = GenerateFrameIndiceswithPadding(
            padding='replicate')
        rlt = frame_index_generator(copy.deepcopy(results))
        assert self.check_keys_contain(rlt.keys(), target_keys)
        assert rlt['lq_path'] == lq_paths
        assert rlt['gt_path'] == gt_paths
        # reflection
        lq_paths = [osp.join('fake_lq_root', '000',
                             f'{v:08d}.png') for v in reflection_idx]
        frame_index_generator = GenerateFrameIndiceswithPadding(
            padding='reflection')
        rlt = frame_index_generator(copy.deepcopy(results))
        assert rlt['lq_path'] == lq_paths
        assert rlt['gt_path'] == gt_paths
        # reflection_circle
        lq_paths = [
            osp.join('fake_lq_root', '000',
                     f'{v:08d}.png') for v in reflection_circle_idx
        ]
        frame_index_generator = GenerateFrameIndiceswithPadding(
            padding='reflection_circle')
        rlt = frame_index_generator(copy.deepcopy(results))
        assert rlt['lq_path'] == lq_paths
        assert rlt['gt_path'] == gt_paths
        # circle
        lq_paths = [osp.join('fake_lq_root', '000',
                             f'{v:08d}.png') for v in circle_idx]
        frame_index_generator = GenerateFrameIndiceswithPadding(
            padding='circle')
        rlt = frame_index_generator(copy.deepcopy(results))
        assert rlt['lq_path'] == lq_paths
        assert rlt['gt_path'] == gt_paths

        results = dict(
            lq_path='fake_lq_root',
            gt_path='fake_gt_root',
            key=osp.join('000', '00000099'),
            max_frame_num=100,
            num_input_frames=5)
        target_keys = ['lq_path', 'gt_path', 'key']
        replicate_idx = [97, 98, 99, 99, 99]
        reflection_idx = [97, 98, 99, 98, 97]
        reflection_circle_idx = [97, 98, 99, 96, 95]
        circle_idx = [97, 98, 99, 95, 96]

        # replicate
        lq_paths = [osp.join('fake_lq_root', '000',
                             f'{v:08d}.png') for v in replicate_idx]
        gt_paths = [osp.join('fake_gt_root', '000', '00000099.png')]
        frame_index_generator = GenerateFrameIndiceswithPadding(
            padding='replicate')
        rlt = frame_index_generator(copy.deepcopy(results))
        assert self.check_keys_contain(rlt.keys(), target_keys)
        assert rlt['lq_path'] == lq_paths
        assert rlt['gt_path'] == gt_paths
        # reflection
        lq_paths = [osp.join('fake_lq_root', '000',
                             f'{v:08d}.png') for v in reflection_idx]
        frame_index_generator = GenerateFrameIndiceswithPadding(
            padding='reflection')
        rlt = frame_index_generator(copy.deepcopy(results))
        assert rlt['lq_path'] == lq_paths
        assert rlt['gt_path'] == gt_paths
        # reflection_circle
        lq_paths = [
            osp.join('fake_lq_root', '000',
                     f'{v:08d}.png') for v in reflection_circle_idx
        ]
        frame_index_generator = GenerateFrameIndiceswithPadding(
            padding='reflection_circle')
        rlt = frame_index_generator(copy.deepcopy(results))
        assert rlt['lq_path'] == lq_paths
        assert rlt['gt_path'] == gt_paths
        # circle
        lq_paths = [osp.join('fake_lq_root', '000',
                             f'{v:08d}.png') for v in circle_idx]
        frame_index_generator = GenerateFrameIndiceswithPadding(
            padding='circle')
        rlt = frame_index_generator(copy.deepcopy(results))
        assert rlt['lq_path'] == lq_paths
        assert rlt['gt_path'] == gt_paths

        name_ = repr(frame_index_generator)
        assert name_ == frame_index_generator.__class__.__name__ + (
            "(padding='circle')")

    def test_frame_index_generator(self):
        results = dict(
            lq_path='fake_lq_root',
            gt_path='fake_gt_root',
            key=osp.join('000', '00000010'),
            num_input_frames=3)
        target_keys = ['lq_path', 'gt_path', 'key', 'interval']
        frame_index_generator = GenerateFrameIndices(
            interval_list=[1], frames_per_clip=99)
        rlt = frame_index_generator(copy.deepcopy(results))
        assert self.check_keys_contain(rlt.keys(), target_keys)

        name_ = repr(frame_index_generator)
        assert name_ == frame_index_generator.__class__.__name__ + (
            '(interval_list=[1], frames_per_clip=99)')

        # index out of range
        frame_index_generator = GenerateFrameIndices(interval_list=[10])
        rlt = frame_index_generator(copy.deepcopy(results))
        assert self.check_keys_contain(rlt.keys(), target_keys)

        # index out of range
        results['key'] = osp.join('000', '00000099')
        frame_index_generator = GenerateFrameIndices(interval_list=[2, 3])
        rlt = frame_index_generator(copy.deepcopy(results))
        assert self.check_keys_contain(rlt.keys(), target_keys)

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

    def test_frame_index_generation_for_recurrent(self):
        results = dict(
            lq_path='fake_lq_root',
            gt_path='fake_gt_root',
            key='000',
            num_input_frames=10,
            sequence_length=100)

        target_keys = [
            'lq_path', 'gt_path', 'key', 'interval', 'num_input_frames',
            'sequence_length'
        ]
        frame_index_generator = GenerateSegmentIndices(interval_list=[1, 5, 9])
        rlt = frame_index_generator(copy.deepcopy(results))
        assert self.check_keys_contain(rlt.keys(), target_keys)

        name_ = repr(frame_index_generator)
        assert name_ == frame_index_generator.__class__.__name__ + (
            '(interval_list=[1, 5, 9])')

        # interval too large
        results = dict(
            lq_path='fake_lq_root',
            gt_path='fake_gt_root',
            key='000',
            num_input_frames=11,
            sequence_length=100)

        frame_index_generator = GenerateSegmentIndices(interval_list=[10])
        with pytest.raises(ValueError):
            frame_index_generator(copy.deepcopy(results))

    def test_mirror_sequence(self):
        lqs = [np.random.rand(4, 4, 3) for _ in range(0, 5)]
        gts = [np.random.rand(16, 16, 3) for _ in range(0, 5)]

        target_keys = ['lq', 'gt']
        mirror_sequence = MirrorSequence(keys=['lq', 'gt'])
        results = dict(lq=lqs, gt=gts)
        results = mirror_sequence(results)

        assert self.check_keys_contain(results.keys(), target_keys)
        for i in range(0, 5):
            np.testing.assert_almost_equal(results['lq'][i],
                                           results['lq'][-i - 1])
            np.testing.assert_almost_equal(results['gt'][i],
                                           results['gt'][-i - 1])

        assert repr(mirror_sequence) == mirror_sequence.__class__.__name__ + (
            "(keys=['lq', 'gt'])")

        # each key should contain a list of nparray
        with pytest.raises(TypeError):
            results = dict(lq=0, gt=gts)
            mirror_sequence(results)

    def test_quantize(self):
        results = {}

        # clip (>1)
        results['gt'] = 1.1 * np.ones((1, 1, 3)).astype(np.float32)
        model = Quantize(keys=['gt'])
        assert np.array_equal(
            model(results)['gt'],
            np.ones((1, 1, 3)).astype(np.float32))

        # clip (<0)
        results['gt'] = -0.1 * np.ones((1, 1, 3)).astype(np.float32)
        model = Quantize(keys=['gt'])
        assert np.array_equal(
            model(results)['gt'],
            np.zeros((1, 1, 3)).astype(np.float32))

        # round
        results['gt'] = (1 / 255. + 1e-8) * np.ones(
            (1, 1, 3)).astype(np.float32)
        model = Quantize(keys=['gt'])
        assert np.array_equal(
            model(results)['gt'], (1 / 255.) * np.ones(
                (1, 1, 3)).astype(np.float32))

    def test_copy_value(self):
        with pytest.raises(AssertionError):
            CopyValues(src_keys='gt', dst_keys='lq')
        with pytest.raises(ValueError):
            CopyValues(src_keys=['gt', 'mask'], dst_keys=['lq'])

        results = {}
        results['gt'] = np.zeros((1)).astype(np.float32)

        copy_ = CopyValues(src_keys=['gt'], dst_keys=['lq'])
        assert np.array_equal(copy_(results)['lq'], results['gt'])
        assert repr(copy_) == copy_.__class__.__name__ + (
            "(src_keys=['gt'])"
            "(dst_keys=['lq'])")

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

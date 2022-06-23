# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import numpy as np
import pytest

from mmedit.transforms import (BinarizeImage, Clip, ColorJitter, CopyValues,
                               Flip, GenerateFrameIndices,
                               GenerateFrameIndiceswithPadding,
                               GenerateSegmentIndices, MirrorSequence,
                               RandomAffine, RandomMaskDilation,
                               RandomRotation, RandomTransposeHW, Resize,
                               SetValues, TemporalReverse, UnsharpMasking)


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
    def check_flip(origin_img, result_img, flip_direction):
        """Check if the origin_img are flipped correctly into result_img
        in different flip_directions

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
        """Check if the origin_imgs are transposed correctly"""

        h, w, c = origin_img.shape
        for i in range(c):
            for j in range(h):
                for k in range(w):
                    if result_img[k, j, i] != origin_img[j, k, i]:  # noqa:E501
                        return False
        return True

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

    def test_flip(self):
        results = copy.deepcopy(self.results)

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

    def test_frame_index_generator(self):
        results = dict(
            img_path='fake_img_root',
            gt_path='fake_gt_root',
            key=osp.join('000', '00000010'),
            num_input_frames=3)
        target_keys = ['img_path', 'gt_path', 'key', 'interval']
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

    def test_frame_index_generation_with_padding(self):
        with pytest.raises(ValueError):
            # Wrong padding mode
            GenerateFrameIndiceswithPadding(padding='fake')

        results = dict(
            img_path='fake_img_root',
            gt_path='fake_gt_root',
            key=osp.join('000', '00000000'),
            max_frame_num=100,
            num_input_frames=5)
        target_keys = ['img_path', 'gt_path', 'key']
        replicate_idx = [0, 0, 0, 1, 2]
        reflection_idx = [2, 1, 0, 1, 2]
        reflection_circle_idx = [4, 3, 0, 1, 2]
        circle_idx = [3, 4, 0, 1, 2]

        # replicate
        img_paths = [
            osp.join('fake_img_root', '000', f'{v:08d}.png')
            for v in replicate_idx
        ]
        gt_paths = [osp.join('fake_gt_root', '000', '00000000.png')]
        frame_index_generator = GenerateFrameIndiceswithPadding(
            padding='replicate')
        rlt = frame_index_generator(copy.deepcopy(results))
        assert self.check_keys_contain(rlt.keys(), target_keys)
        assert rlt['img_path'] == img_paths
        assert rlt['gt_path'] == gt_paths
        # reflection
        img_paths = [
            osp.join('fake_img_root', '000', f'{v:08d}.png')
            for v in reflection_idx
        ]
        frame_index_generator = GenerateFrameIndiceswithPadding(
            padding='reflection')
        rlt = frame_index_generator(copy.deepcopy(results))
        assert rlt['img_path'] == img_paths
        assert rlt['gt_path'] == gt_paths
        # reflection_circle
        img_paths = [
            osp.join('fake_img_root', '000', f'{v:08d}.png')
            for v in reflection_circle_idx
        ]
        frame_index_generator = GenerateFrameIndiceswithPadding(
            padding='reflection_circle')
        rlt = frame_index_generator(copy.deepcopy(results))
        assert rlt['img_path'] == img_paths
        assert rlt['gt_path'] == gt_paths
        # circle
        img_paths = [
            osp.join('fake_img_root', '000', f'{v:08d}.png')
            for v in circle_idx
        ]
        frame_index_generator = GenerateFrameIndiceswithPadding(
            padding='circle')
        rlt = frame_index_generator(copy.deepcopy(results))
        assert rlt['img_path'] == img_paths
        assert rlt['gt_path'] == gt_paths

        results = dict(
            img_path='fake_img_root',
            gt_path='fake_gt_root',
            key=osp.join('000', '00000099'),
            max_frame_num=100,
            num_input_frames=5)
        target_keys = ['img_path', 'gt_path', 'key']
        replicate_idx = [97, 98, 99, 99, 99]
        reflection_idx = [97, 98, 99, 98, 97]
        reflection_circle_idx = [97, 98, 99, 96, 95]
        circle_idx = [97, 98, 99, 95, 96]

        # replicate
        img_paths = [
            osp.join('fake_img_root', '000', f'{v:08d}.png')
            for v in replicate_idx
        ]
        gt_paths = [osp.join('fake_gt_root', '000', '00000099.png')]
        frame_index_generator = GenerateFrameIndiceswithPadding(
            padding='replicate')
        rlt = frame_index_generator(copy.deepcopy(results))
        assert self.check_keys_contain(rlt.keys(), target_keys)
        assert rlt['img_path'] == img_paths
        assert rlt['gt_path'] == gt_paths
        # reflection
        img_paths = [
            osp.join('fake_img_root', '000', f'{v:08d}.png')
            for v in reflection_idx
        ]
        frame_index_generator = GenerateFrameIndiceswithPadding(
            padding='reflection')
        rlt = frame_index_generator(copy.deepcopy(results))
        assert rlt['img_path'] == img_paths
        assert rlt['gt_path'] == gt_paths
        # reflection_circle
        img_paths = [
            osp.join('fake_img_root', '000', f'{v:08d}.png')
            for v in reflection_circle_idx
        ]
        frame_index_generator = GenerateFrameIndiceswithPadding(
            padding='reflection_circle')
        rlt = frame_index_generator(copy.deepcopy(results))
        assert rlt['img_path'] == img_paths
        assert rlt['gt_path'] == gt_paths
        # circle
        img_paths = [
            osp.join('fake_img_root', '000', f'{v:08d}.png')
            for v in circle_idx
        ]
        frame_index_generator = GenerateFrameIndiceswithPadding(
            padding='circle')
        rlt = frame_index_generator(copy.deepcopy(results))
        assert rlt['img_path'] == img_paths
        assert rlt['gt_path'] == gt_paths

        name_ = repr(frame_index_generator)
        assert name_ == frame_index_generator.__class__.__name__ + (
            "(padding='circle')")

    def test_frame_index_generation_for_recurrent(self):
        results = dict(
            img_path='fake_img_root',
            gt_path='fake_gt_root',
            key='000',
            num_input_frames=10,
            sequence_length=100)

        target_keys = [
            'img_path',
            'gt_path',
            'key',
            'interval',
            'num_input_frames',
            'sequence_length',
        ]
        frame_index_generator = GenerateSegmentIndices(interval_list=[1, 5, 9])
        rlt = frame_index_generator(copy.deepcopy(results))
        assert self.check_keys_contain(rlt.keys(), target_keys)

        name_ = repr(frame_index_generator)
        assert name_ == frame_index_generator.__class__.__name__ + (
            '(interval_list=[1, 5, 9])')

        # interval too large
        results = dict(
            img_path='fake_img_root',
            gt_path='fake_gt_root',
            key='000',
            num_input_frames=11,
            sequence_length=100)

        frame_index_generator = GenerateSegmentIndices(interval_list=[10])
        with pytest.raises(ValueError):
            frame_index_generator(copy.deepcopy(results))

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

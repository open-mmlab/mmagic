# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import numpy as np
import pytest

from mmagic.datasets.transforms import (GenerateFrameIndices,
                                        GenerateFrameIndiceswithPadding,
                                        GenerateSegmentIndices)


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
            sequence_length=100,
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
            sequence_length=100,
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

        # num_input_frames is None
        results = dict(
            img_path='fake_img_root',
            gt_path='fake_gt_root',
            key='000',
            num_input_frames=None,
            sequence_length=100)

        frame_index_generator = GenerateSegmentIndices(interval_list=[1])
        rlt = frame_index_generator(copy.deepcopy(results))
        assert len(rlt['img_path']) == 100


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

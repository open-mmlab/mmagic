# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

from mmagic.datasets import GrowScaleImgDataset
from mmagic.utils import register_all_modules

register_all_modules()


class TestGrowScaleImgDataset:

    @classmethod
    def setup_class(cls):
        cls.imgs_root = osp.join(osp.dirname(__file__), '..', 'data/image')
        cls.imgs_roots = {
            '4': cls.imgs_root,
            '8': osp.join(cls.imgs_root, 'img_root'),
            '32': osp.join(cls.imgs_root, 'img_root', 'grass')
        }
        cls.default_pipeline = [dict(type='LoadImageFromFile', key='gt')]
        cls.len_per_stage = 10
        cls.gpu_samples_base = 2

    def test_dynamic_unconditional_img_dataset(self):
        dataset = GrowScaleImgDataset(
            self.imgs_roots,
            self.default_pipeline,
            self.len_per_stage,
            io_backend='local',
            gpu_samples_base=self.gpu_samples_base)
        assert len(dataset) == 10
        img = dataset[2]['gt']
        assert img.ndim == 3
        assert repr(dataset) == (
            f'dataset_name: {dataset.__class__}, '
            f'total {10} images in imgs_root: {self.imgs_root}')
        assert dataset.samples_per_gpu == 2

        dataset.update_annotations(8)
        assert len(dataset) == 10
        img = dataset[2]['gt']
        assert img.ndim == 3
        assert repr(dataset) == (f'dataset_name: {dataset.__class__}, '
                                 f'total {10} images in imgs_root:'
                                 f' {osp.join(self.imgs_root, "img_root")}')
        assert dataset.samples_per_gpu == 2

        dataset = GrowScaleImgDataset(
            self.imgs_roots,
            self.default_pipeline,
            20,
            gpu_samples_base=self.gpu_samples_base,
            gpu_samples_per_scale={
                '4': 10,
                '16': 13
            })
        assert len(dataset) == 20
        img = dataset[2]['gt']
        assert img.ndim == 3
        assert repr(dataset) == (
            f'dataset_name: {dataset.__class__}, '
            f'total {20} images in imgs_root: {self.imgs_root}')
        assert dataset.samples_per_gpu == 10

        dataset.update_annotations(8)
        assert len(dataset) == 20
        img = dataset[2]['gt']
        assert img.ndim == 3
        assert repr(dataset) == (f'dataset_name: {dataset.__class__}, '
                                 f'total {20} images in imgs_root:'
                                 f' {osp.join(self.imgs_root, "img_root")}')
        assert dataset.samples_per_gpu == 2

        dataset = GrowScaleImgDataset(
            self.imgs_roots, self.default_pipeline, 5, test_mode=True)
        assert len(dataset) == 5
        img = dataset[2]['gt']
        assert img.ndim == 3
        assert repr(dataset) == (
            f'dataset_name: {dataset.__class__}, '
            f'total {5} images in imgs_root: {self.imgs_root}')

        dataset.update_annotations(24)
        assert len(dataset) == 5
        img = dataset[2]['gt']
        assert img.ndim == 3
        _path_str = osp.join(self.imgs_root, 'img_root', 'grass')
        assert repr(dataset) == (f'dataset_name: {dataset.__class__}, '
                                 f'total {5} images in imgs_root: {_path_str}')

        with pytest.raises(AssertionError):
            _ = GrowScaleImgDataset(
                self.imgs_root,
                self.default_pipeline,
                10,
                gpu_samples_per_scale=10)

        with pytest.raises(AssertionError):
            _ = GrowScaleImgDataset(10, self.default_pipeline, 10.)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

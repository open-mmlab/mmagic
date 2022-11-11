# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmedit.datasets import PairedMultipathDataset
from mmedit.utils import register_all_modules

register_all_modules()


class TestPairedMultipathDataset(object):

    @classmethod
    def setup_class(cls):
        cls.imgs_root = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/paired')
        cls.default_pipeline = [
            dict(
                type='LoadPairedImageFromFile',
                key='pair',
                domain_a='a',
                domain_b='b'),
            dict(
                type='TransformBroadcaster',
                mapping={'img': ['img_a', 'img_b']},
                auto_remap=True,
                share_random_params=True,
                transforms=[
                    dict(
                        type='Resize',
                        scale=(286, 286),
                        interpolation='bicubic'),
                    dict(type='FixedCrop', keys=['img'], crop_size=(256, 256))
                ]),
            dict(type='Flip', direction='horizontal', keys=['img_a', 'img_b']),
            dict(type='PackEditInputs', keys=['img_a', 'img_b'])
        ]

    def test_paired_image_dataset(self):
        dataset = PairedMultipathDataset(datasets=[
            dict(
                type='PairedImageDataset',
                data_root=self.imgs_root,
                pipeline=self.default_pipeline),
            dict(
                type='PairedImageDataset',
                data_root=self.imgs_root,
                pipeline=self.default_pipeline),
        ])
        assert len(dataset) == 4
        print(dataset)
        img = dataset[0]['inputs']['img_a']
        assert img.ndim == 3
        img = dataset[0]['inputs']['img_b']
        assert img.ndim == 3
        img = dataset[1]['inputs']['img_a']
        assert img.ndim == 3
        img = dataset[1]['inputs']['img_b']
        assert img.ndim == 3


class TestPairedMultipathDatasetFix(object):

    @classmethod
    def setup_class(cls):
        cls.imgs_root = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/paired')
        cls.default_pipeline = [
            dict(
                type='LoadPairedImageFromFile',
                key='pair',
                domain_a='a',
                domain_b='b'),
            dict(
                type='TransformBroadcaster',
                mapping={'img': ['img_a', 'img_b']},
                auto_remap=True,
                share_random_params=True,
                transforms=[
                    dict(
                        type='Resize',
                        scale=(286, 286),
                        interpolation='bicubic'),
                    dict(type='FixedCrop', keys=['img'], crop_size=(256, 256))
                ]),
            dict(type='Flip', direction='horizontal', keys=['img_a', 'img_b']),
            dict(type='PackEditInputs', keys=['img_a', 'img_b'])
        ]

    def test_paired_image_dataset(self):
        dataset = PairedMultipathDataset(
            fix_length=1,
            datasets=[
                dict(
                    type='PairedImageDataset',
                    data_root=self.imgs_root,
                    pipeline=self.default_pipeline),
                dict(
                    type='PairedImageDataset',
                    data_root=self.imgs_root,
                    pipeline=self.default_pipeline),
            ])
        assert len(dataset) == 2
        print(dataset)
        img = dataset[0]['inputs']['img_a']
        assert img.ndim == 3
        img = dataset[0]['inputs']['img_b']
        assert img.ndim == 3
        img = dataset[1]['inputs']['img_a']
        assert img.ndim == 3
        img = dataset[1]['inputs']['img_b']
        assert img.ndim == 3

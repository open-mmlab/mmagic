# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmagic.datasets import PairedImageDataset
from mmagic.utils import register_all_modules

register_all_modules()


class TestPairedImageDataset(object):

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
            dict(type='PackInputs', keys=['img_a', 'img_b'])
        ]

    def test_paired_image_dataset(self):
        dataset = PairedImageDataset(
            self.imgs_root, pipeline=self.default_pipeline)
        assert len(dataset) == 2
        img = dataset[0]['inputs']['img_a']
        assert img.ndim == 3
        img = dataset[0]['inputs']['img_b']
        assert img.ndim == 3

        dataset = PairedImageDataset(
            self.imgs_root, pipeline=self.default_pipeline, io_backend='local')
        assert len(dataset) == 2
        img = dataset[0]['inputs']['img_a']
        assert img.ndim == 3
        img = dataset[0]['inputs']['img_b']
        assert img.ndim == 3


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

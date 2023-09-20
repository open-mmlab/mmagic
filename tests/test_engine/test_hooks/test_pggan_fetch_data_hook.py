# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase
from unittest.mock import MagicMock

import torch
from mmengine.dataset import DefaultSampler, InfiniteSampler, pseudo_collate
from mmengine.runner import IterBasedTrainLoop
from torch.utils.data.dataloader import DataLoader

from mmagic.engine import PGGANFetchDataHook
from mmagic.registry import DATASETS, MODELS
from mmagic.utils import register_all_modules

register_all_modules()


class TestPGGANFetchDataHook(TestCase):

    pggan_cfg = dict(
        type='ProgressiveGrowingGAN',
        data_preprocessor=dict(type='DataPreprocessor'),
        noise_size=512,
        generator=dict(type='PGGANGenerator', out_scale=8),
        discriminator=dict(type='PGGANDiscriminator', in_scale=8),
        nkimgs_per_scale={
            '4': 600,
            '8': 1200,
            '16': 1200,
            '32': 1200,
            '64': 1200,
            '128': 12000
        },
        transition_kimgs=600,
        ema_config=dict(interval=1))

    imgs_root = osp.join(osp.dirname(__file__), '..', '..', 'data/image')
    grow_scale_dataset_cfg = dict(
        type='GrowScaleImgDataset',
        data_roots={
            '4': imgs_root,
            '8': osp.join(imgs_root, 'img_root'),
            '32': osp.join(imgs_root, 'img_root', 'grass')
        },
        gpu_samples_base=4,
        gpu_samples_per_scale={
            '4': 64,
            '8': 32,
            '16': 16,
            '32': 8,
            '64': 4
        },
        len_per_stage=10,
        pipeline=[dict(type='LoadImageFromFile', key='img')])

    def test_before_train_iter(self):
        runner = MagicMock()
        model = MODELS.build(self.pggan_cfg)
        dataset = DATASETS.build(self.grow_scale_dataset_cfg)

        # test default sampler
        default_sampler = DefaultSampler(dataset)
        dataloader = DataLoader(
            batch_size=64,
            dataset=dataset,
            sampler=default_sampler,
            collate_fn=pseudo_collate)

        runner.train_loop = MagicMock(spec=IterBasedTrainLoop)
        runner.train_loop.dataloader = dataloader
        runner.model = model

        hooks = PGGANFetchDataHook()
        hooks.before_train_iter(runner, 0, None)

        for scale, target_bz in self.grow_scale_dataset_cfg[
                'gpu_samples_per_scale'].items():

            model._next_scale_int = torch.tensor(int(scale), dtype=torch.int32)
            hooks.before_train_iter(runner, 0, None)
            self.assertEqual(runner.train_loop.dataloader.batch_size,
                             target_bz)
            # check attribute of default sampler
            sampler = runner.train_loop.dataloader.sampler
            self.assertEqual(sampler.seed, default_sampler.seed)
            self.assertEqual(sampler.shuffle, default_sampler.shuffle)
            self.assertEqual(sampler.round_up, default_sampler.round_up)

        # set `_next_scale_int` as int
        delattr(model, '_next_scale_int')
        setattr(model, '_next_scale_int', 128)
        hooks.before_train_iter(runner, 0, None)
        self.assertEqual(runner.train_loop.dataloader.batch_size, 4)

        # test InfinitySampler
        infinite_sampler = InfiniteSampler(dataset)
        dataloader = DataLoader(
            batch_size=64,
            dataset=dataset,
            sampler=infinite_sampler,
            collate_fn=pseudo_collate)
        runner.train_loop.dataloader = dataloader
        for scale, target_bz in self.grow_scale_dataset_cfg[
                'gpu_samples_per_scale'].items():

            model._next_scale_int = torch.tensor(int(scale), dtype=torch.int32)
            hooks.before_train_iter(runner, 0, None)
            self.assertEqual(runner.train_loop.dataloader.batch_size,
                             target_bz)
            # check attribute of infinite sampler
            sampler = runner.train_loop.dataloader.sampler
            self.assertEqual(sampler.seed, infinite_sampler.seed)
            self.assertEqual(sampler.shuffle, infinite_sampler.shuffle)

        # test do not update + `IterBasedTrainLoop`
        hooks.before_train_iter(runner, 1, None)

        # test not `IterBasedTrainLoop`
        runner.train_loop = MagicMock()
        runner.train_loop.dataloader = dataloader
        runner.model = model
        model._next_scale_int = 8
        # test update
        hooks.before_train_iter(runner, 0, None)
        # test do not update
        hooks.before_train_iter(runner, 1, None)

        # test invalid sampler type
        dataloader = DataLoader(
            batch_size=64, dataset=dataset, collate_fn=pseudo_collate)
        runner.train_loop.dataloader = dataloader
        model._next_scale_int = 4
        self.assertRaises(ValueError, hooks.before_train_iter, runner, 0, None)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

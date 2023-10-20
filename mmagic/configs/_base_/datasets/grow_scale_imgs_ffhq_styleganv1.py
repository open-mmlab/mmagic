# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler, InfiniteSampler

from mmagic.datasets import BasicImageDataset, GrowScaleImgDataset
from mmagic.datasets.transforms import Flip, LoadImageFromFile, PackInputs

dataset_type = 'GrowScaleImgDataset'

pipeline = [
    dict(type=LoadImageFromFile, key='gt'),
    dict(type=Flip, keys='gt', direction='horizontal'),
    dict(type=PackInputs)
]

train_dataloader = dict(
    num_workers=4,
    batch_size=64,
    dataset=dict(
        type=GrowScaleImgDataset,
        data_roots={
            '1024': './data/ffhq/images',
            '256': './data/ffhq/ffhq_imgs/ffhq_256',
        },
        gpu_samples_base=4,
        # note that this should be changed with total gpu number
        gpu_samples_per_scale={
            '4': 64,
            '8': 32,
            '16': 16,
            '32': 8,
            '64': 4,
            '128': 4,
            '256': 4,
            '512': 4,
            '1024': 4
        },
        len_per_stage=300000,
        pipeline=pipeline),
    sampler=dict(type=InfiniteSampler, shuffle=True))

test_dataloader = dict(
    num_workers=4,
    batch_size=64,
    dataset=dict(
        type=BasicImageDataset,
        data_prefix=dict(gt=''),
        pipeline=pipeline,
        data_root='./data/ffhq/images'),
    sampler=dict(type=DefaultSampler, shuffle=False))

val_dataloader = test_dataloader

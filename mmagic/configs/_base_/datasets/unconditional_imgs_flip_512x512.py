# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset.sampler import DefaultSampler, InfiniteSampler

from mmagic.datasets.basic_image_dataset import BasicImageDataset
from mmagic.datasets.transforms.aug_shape import Flip, Resize
from mmagic.datasets.transforms.formatting import PackInputs
from mmagic.datasets.transforms.loading import LoadImageFromFile

dataset_type = BasicImageDataset

# TODO:
train_pipeline = [
    dict(type=LoadImageFromFile, key='gt'),
    dict(type=Resize, keys='gt', scale=(512, 512)),
    dict(type=Flip, keys=['gt'], direction='horizontal'),  # TODO:
    dict(type=PackInputs)
]

# `batch_size` and `data_root` need to be set.
train_dataloader = dict(
    batch_size=None,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(gt=''),
        data_root=None,  # set by user
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=None,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(gt=''),
        data_root=None,  # set by user
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=False),
    persistent_workers=True)

test_dataloader = dict(
    batch_size=None,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(gt=''),
        data_root=None,  # set by user
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=False),
    persistent_workers=True)

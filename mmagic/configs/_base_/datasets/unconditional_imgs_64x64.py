# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler, InfiniteSampler

from mmagic.datasets.transforms import LoadImageFromFile, PackInputs, Resize

dataset_type = 'BasicImageDataset'

train_pipeline = [
    dict(type=LoadImageFromFile, key='gt'),
    dict(type=Resize, keys='gt', scale=(64, 64)),
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

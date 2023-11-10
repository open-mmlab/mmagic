# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler, InfiniteSampler

from mmagic.datasets import CIFAR10
from mmagic.datasets.transforms import Flip, PackInputs

cifar_pipeline = [
    dict(type=Flip, keys=['gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type=PackInputs)
]
cifar_dataset = dict(
    type=CIFAR10,
    data_root='./data',
    data_prefix='cifar10',
    test_mode=False,
    pipeline=cifar_pipeline)

# test dataset do not use flip
cifar_pipeline_test = [dict(type=PackInputs)]
cifar_dataset_test = dict(
    type=CIFAR10,
    data_root='./data',
    data_prefix='cifar10',
    test_mode=False,
    pipeline=cifar_pipeline_test)

train_dataloader = dict(
    num_workers=2,
    dataset=cifar_dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    persistent_workers=True)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=cifar_dataset_test,
    sampler=dict(type=DefaultSampler, shuffle=False),
    persistent_workers=True)

test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=cifar_dataset_test,
    sampler=dict(type=DefaultSampler, shuffle=False),
    persistent_workers=True)

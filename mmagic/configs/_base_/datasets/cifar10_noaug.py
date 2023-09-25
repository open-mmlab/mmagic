# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset.sampler import DefaultSampler, InfiniteSampler

from mmagic.datasets.cifar10_dataset import CIFAR10
from mmagic.datasets.transforms.formatting import PackInputs

cifar_pipeline = [dict(type=PackInputs)]
cifar_dataset = dict(
    type=CIFAR10,
    data_root='./data',
    data_prefix='cifar10',
    test_mode=False,
    pipeline=cifar_pipeline)

train_dataloader = dict(
    num_workers=2,
    dataset=cifar_dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    persistent_workers=True)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=cifar_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    persistent_workers=True)

test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=cifar_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    persistent_workers=True)

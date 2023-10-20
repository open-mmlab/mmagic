# Copyright (c) OpenMMLab. All rights reserved.
# dataset settings
from mmengine.dataset.sampler import DefaultSampler

from mmagic.datasets.imagenet_dataset import ImageNet
from mmagic.datasets.transforms.aug_shape import Resize
from mmagic.datasets.transforms.crop import CenterCropLongEdge
from mmagic.datasets.transforms.formatting import PackInputs
from mmagic.datasets.transforms.loading import LoadImageFromFile

dataset_type = ImageNet

# different from mmcls, we adopt the setting used in BigGAN.
# Remove `RandomFlip` augmentation and change `RandomCropLongEdge` to
# `CenterCropLongEdge` to eliminate randomness.
# dataset settings
train_pipeline = [
    dict(type=LoadImageFromFile, key='img'),
    dict(type=CenterCropLongEdge),
    dict(type=Resize, scale=(128, 128), backend='pillow'),
    dict(type=PackInputs)
]

test_pipeline = [
    dict(type=LoadImageFromFile, key='img'),
    dict(type=CenterCropLongEdge),
    dict(type=Resize, scale=(128, 128), backend='pillow'),
    dict(type=PackInputs)
]

train_dataloader = dict(
    batch_size=None,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=True),
    persistent_workers=True)

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='./data/imagenet/',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=test_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=False),
    persistent_workers=True)

test_dataloader = val_dataloader

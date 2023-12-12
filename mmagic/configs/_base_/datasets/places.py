# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler, InfiniteSampler

from mmagic.evaluation import MAE, PSNR, SSIM

# Base config for places365 dataset

# dataset settings
dataset_type = 'BasicImageDataset'
data_root = 'data/Places'

train_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(gt='data_large'),
        ann_file='meta/places365_train_challenge.txt',
        # Note that Places365-standard (1.8M images) and
        # Place365-challenge (8M images) use different image lists.
        test_mode=False,
    ))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(gt='val_large'),
        ann_file='meta/places365_val.txt',
        test_mode=True,
    ))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type=MAE, mask_key='mask', scaling=100),
    # By default, compute with pixel value from 0-1
    # scale=2 to align with 1.0
    # scale=100 seems to align with readme
    dict(type=PSNR),
    dict(type=SSIM),
]

test_evaluator = val_evaluator

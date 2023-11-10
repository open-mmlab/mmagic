# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler, InfiniteSampler

from mmagic.evaluation import MAE, PSNR, SSIM

# Base config for CelebA-HQ dataset

# dataset settings
dataset_type = 'BasicImageDataset'
data_root = 'data/CelebA-HQ'

train_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(gt=''),
        ann_file='train_celeba_img_list.txt',
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
        data_prefix=dict(gt=''),
        ann_file='val_celeba_img_list.txt',
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

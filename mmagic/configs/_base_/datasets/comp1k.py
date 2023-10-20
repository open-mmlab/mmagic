# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler, InfiniteSampler

from mmagic.evaluation import SAD, ConnectivityError, GradientError, MattingMSE

# Base config for Composition-1K dataset

# dataset settings
dataset_type = 'AdobeComp1kDataset'
data_root = 'data/adobe_composition-1k'

train_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='training_list.json',
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
        ann_file='test_list.json',
        test_mode=True,
    ))

test_dataloader = val_dataloader

# TODO: matting
val_evaluator = [
    dict(type=SAD),
    dict(type=MattingMSE),
    dict(type=GradientError),
    dict(type=ConnectivityError),
]

test_evaluator = val_evaluator

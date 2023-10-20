# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler, InfiniteSampler

from mmagic.datasets.transforms import Crop, Flip, PackInputs, Resize

dataset_type = 'UnpairedImageDataset'
domain_a = None  # set by user
domain_b = None  # set by user
train_pipeline = [
    dict(type='LoadImageFromFile', key='img_A', color_type='color'),
    dict(type='LoadImageFromFile', key='img_B', color_type='color'),
    dict(
        type='TransformBroadcaster',
        mapping={'img': ['img_A', 'img_B']},
        auto_remap=True,
        share_random_params=True,
        transforms=[
            dict(type=Resize, scale=(286, 286), interpolation='bicubic'),
            dict(
                type=Crop,
                keys=['img'],
                crop_size=(256, 256),
                random_crop=True),
        ]),
    dict(type=Flip, keys=['img_A'], direction='horizontal'),
    dict(type=Flip, keys=['img_B'], direction='horizontal'),
    # NOTE: users should implement their own keyMapper and Pack operation
    # dict(
    #     type='KeyMapper',
    #     mapping={
    #         f'img_{domain_a}': 'img_A',
    #         f'img_{domain_b}': 'img_B'
    #     },
    #     remapping={
    #         f'img_{domain_a}': f'img_{domain_a}',
    #         f'img_{domain_b}': f'img_{domain_b}'
    #     }),
    # dict(
    #     type=PackInputs,
    #     keys=[f'img_{domain_a}', f'img_{domain_b}'],
    #     data_keys=[f'img_{domain_a}', f'img_{domain_b}'])
]

test_pipeline = [
    dict(type='LoadImageFromFile', key='img_A', color_type='color'),
    dict(type='LoadImageFromFile', key='img_B', color_type='color'),
    dict(
        type='TransformBroadcaster',
        mapping={'img': ['img_A', 'img_B']},
        auto_remap=True,
        share_random_params=True,
        transforms=dict(
            type=Resize, scale=(256, 256), interpolation='bicubic'),
    ),
    # NOTE: users should implement their own keyMapper and Pack operation
    # dict(
    #     type='KeyMapper',
    #     mapping={
    #         f'img_{domain_a}': 'img_A',
    #         f'img_{domain_b}': 'img_B'
    #     },
    #     remapping={
    #         f'img_{domain_a}': f'img_{domain_a}',
    #         f'img_{domain_b}': f'img_{domain_b}'
    #     }),
    # dict(
    #     type=PackInputs,
    #     keys=[f'img_{domain_a}', f'img_{domain_b}'],
    #     data_keys=[f'img_{domain_a}', f'img_{domain_b}'])
]

# `batch_size` and `data_root` need to be set.
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=None,  # set by user
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=None,  # set by user
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=False),
    persistent_workers=True)

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=None,  # set by user
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=False),
    persistent_workers=True)

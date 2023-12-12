# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler, InfiniteSampler

from mmagic.datasets.transforms import (FixedCrop, Flip,
                                        LoadPairedImageFromFile, PackInputs,
                                        Resize)

dataset_type = 'PairedImageDataset'
# domain_a = None  # set by user
# domain_b = None  # set by user

train_pipeline = [
    dict(
        type=LoadPairedImageFromFile,
        key='pair',
        domain_a='A',
        domain_b='B',
        color_type='color'),
    dict(
        type=Resize,
        keys=['img_A', 'img_B'],
        scale=(286, 286),
        interpolation='bicubic'),
    dict(type=FixedCrop, keys=['img_A', 'img_B'], crop_size=(256, 256)),
    dict(type=Flip, keys=['img_A', 'img_B'], direction='horizontal'),
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
    dict(
        type=LoadPairedImageFromFile,
        key='pair',
        domain_a='A',
        domain_b='B',
        color_type='color'),
    dict(
        type='TransformBroadcaster',
        mapping={'img': ['img_A', 'img_B']},
        auto_remap=True,
        share_random_params=True,
        transforms=[
            dict(
                type=Resize,
                scale=(256, 256),
                keys='img',
                interpolation='bicubic')
        ]),
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
        pipeline=test_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=False),
    persistent_workers=True)

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=None,  # set by user
        pipeline=test_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=False),
    persistent_workers=True)

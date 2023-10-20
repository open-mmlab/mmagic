# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler

from mmagic.datasets import BasicFramesDataset
from mmagic.datasets.transforms import (GenerateFrameIndiceswithPadding,
                                        GenerateSegmentIndices,
                                        LoadImageFromFile, PackInputs)
from mmagic.engine.runner import MultiTestLoop
from mmagic.evaluation import PSNR, SSIM, Evaluator

# configs for SPMCS-30
SPMC_data_root = 'data/SPMCS'

SPMC_pipeline = [
    dict(type=GenerateFrameIndiceswithPadding, padding='reflection'),
    dict(type=LoadImageFromFile, key='img', channel_order='rgb'),
    dict(type=LoadImageFromFile, key='gt', channel_order='rgb'),
    dict(type=PackInputs)
]

SPMC_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicFramesDataset,
        metainfo=dict(dataset_type='spmcs', task_name='vsr'),
        data_root=SPMC_data_root,
        data_prefix=dict(img='BDx4', gt='GT'),
        ann_file='meta_info_SPMCS_GT.txt',
        depth=2,
        num_input_frames=5,
        pipeline=SPMC_pipeline))

SPMC_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicFramesDataset,
        metainfo=dict(dataset_type='spmcs', task_name='vsr'),
        data_root=SPMC_data_root,
        data_prefix=dict(img='BIx4', gt='GT'),
        ann_file='meta_info_SPMCS_GT.txt',
        depth=2,
        num_input_frames=5,
        pipeline=SPMC_pipeline))

SPMC_bd_evaluator = dict(
    type=Evaluator,
    metrics=[
        dict(type=PSNR, crop_border=8, convert_to='Y', prefix='SPMCS-BDx4-Y'),
        dict(type=SSIM, crop_border=8, convert_to='Y', prefix='SPMCS-BDx4-Y'),
    ])
SPMC_bi_evaluator = dict(
    type=Evaluator,
    metrics=[
        dict(type=PSNR, crop_border=8, convert_to='Y', prefix='SPMCS-BIx4-Y'),
        dict(type=SSIM, crop_border=8, convert_to='Y', prefix='SPMCS-BIx4-Y'),
    ])

# config for vid4
vid4_data_root = 'data/Vid4'

vid4_pipeline = [
    # dict(type=GenerateSegmentIndices, interval_list=[1]),
    dict(type=GenerateFrameIndiceswithPadding, padding='reflection'),
    dict(type=LoadImageFromFile, key='img', channel_order='rgb'),
    dict(type=LoadImageFromFile, key='gt', channel_order='rgb'),
    dict(type=PackInputs)
]
vid4_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicFramesDataset,
        metainfo=dict(dataset_type='vid4', task_name='vsr'),
        data_root=vid4_data_root,
        data_prefix=dict(img='BDx4', gt='GT'),
        ann_file='meta_info_Vid4_GT.txt',
        depth=2,
        num_input_frames=5,
        pipeline=vid4_pipeline))

vid4_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicFramesDataset,
        metainfo=dict(dataset_type='vid4', task_name='vsr'),
        data_root=vid4_data_root,
        data_prefix=dict(img='BIx4', gt='GT'),
        ann_file='meta_info_Vid4_GT.txt',
        depth=2,
        num_input_frames=5,
        pipeline=vid4_pipeline))

vid4_bd_evaluator = dict(
    type=Evaluator,
    metrics=[
        dict(type=PSNR, convert_to='Y', prefix='VID4-BDx4-Y'),
        dict(type=SSIM, convert_to='Y', prefix='VID4-BDx4-Y'),
    ])
vid4_bi_evaluator = dict(
    type=Evaluator,
    metrics=[
        dict(type=PSNR, convert_to='Y', prefix='VID4-BIx4-Y'),
        dict(type=SSIM, convert_to='Y', prefix='VID4-BIx4-Y'),
    ])

# config for test
test_cfg = dict(type=MultiTestLoop)
test_dataloader = [
    SPMC_bd_dataloader,
    SPMC_bi_dataloader,
    vid4_bd_dataloader,
    vid4_bi_dataloader,
]
test_evaluator = [
    SPMC_bd_evaluator,
    SPMC_bi_evaluator,
    vid4_bd_evaluator,
    vid4_bi_evaluator,
]

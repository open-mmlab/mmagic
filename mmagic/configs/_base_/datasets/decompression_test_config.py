# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler

from mmagic.datasets import BasicImageDataset
from mmagic.datasets.transforms import (LoadImageFromFile, PackInputs,
                                        RandomJPEGCompression)
from mmagic.engine.runner import MultiTestLoop
from mmagic.evaluation import PSNR, SSIM

quality = 10
test_pipeline = [
    dict(
        type=LoadImageFromFile,
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type=LoadImageFromFile,
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type=RandomJPEGCompression,
        params=dict(quality=[quality, quality]),
        bgr2rgb=True,
        keys=['img']),
    dict(type=PackInputs)
]

classic5_data_root = 'data/Classic5'
classic5_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='classic5', task_name='CAR'),
        data_root=classic5_data_root,
        data_prefix=dict(img='', gt=''),
        pipeline=test_pipeline))
classic5_evaluator = [
    dict(type=PSNR, prefix='Classic5'),
    dict(type=SSIM, prefix='Classic5'),
]

live1_data_root = 'data/LIVE1'
live1_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='live1', task_name='CAR'),
        data_root=live1_data_root,
        data_prefix=dict(img='', gt=''),
        pipeline=test_pipeline))
live1_evaluator = [
    dict(type=PSNR, prefix='LIVE1'),
    dict(type=SSIM, prefix='LIVE1'),
]

# test config
test_cfg = dict(type=MultiTestLoop)
test_dataloader = [
    classic5_dataloader,
    live1_dataloader,
]
test_evaluator = [
    classic5_evaluator,
    live1_evaluator,
]

# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler

from mmagic.datasets import BasicImageDataset
from mmagic.datasets.transforms import (LoadImageFromFile, PackInputs,
                                        RandomNoise)
from mmagic.engine.runner import MultiTestLoop
from mmagic.evaluation import PSNR, SSIM

sigma = 15
test_pipeline = [
    dict(
        type=LoadImageFromFile,
        key='img',
        color_type='color',
        channel_order='rgb',
        to_y_channel=True,
        imdecode_backend='cv2'),
    dict(
        type=LoadImageFromFile,
        key='gt',
        color_type='color',
        channel_order='rgb',
        to_y_channel=True,
        imdecode_backend='cv2'),
    dict(
        type=RandomNoise,
        params=dict(
            noise_type=['gaussian'],
            noise_prob=[1],
            gaussian_sigma=[sigma, sigma],
            gaussian_gray_noise_prob=1),
        keys=['img']),
    dict(type=PackInputs)
]

data_root = 'data/denoising_gaussian_test'
set12_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='Set12', task_name='denoising'),
        data_root=data_root,
        data_prefix=dict(img='Set12', gt='Set12'),
        pipeline=test_pipeline))
set12_evaluator = [
    dict(type=PSNR, prefix='Set12'),
    dict(type=SSIM, prefix='Set12'),
]

bsd68_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='BSD68', task_name='denoising'),
        data_root=data_root,
        data_prefix=dict(img='BSD68', gt='BSD68'),
        pipeline=test_pipeline))
bsd68_evaluator = [
    dict(type=PSNR, prefix='BSD68'),
    dict(type=SSIM, prefix='BSD68'),
]

urban100_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='Urban100', task_name='denoising'),
        data_root=data_root,
        data_prefix=dict(img='Urban100', gt='Urban100'),
        pipeline=test_pipeline))
urban100_evaluator = [
    dict(type=PSNR, prefix='Urban100'),
    dict(type=SSIM, prefix='Urban100'),
]

# test config
test_cfg = dict(type=MultiTestLoop)
test_dataloader = [
    set12_dataloader,
    bsd68_dataloader,
    urban100_dataloader,
]
test_evaluator = [
    set12_evaluator,
    bsd68_evaluator,
    urban100_evaluator,
]

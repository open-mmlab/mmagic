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
        imdecode_backend='cv2'),
    dict(
        type=LoadImageFromFile,
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type=RandomNoise,
        params=dict(
            noise_type=['gaussian'],
            noise_prob=[1],
            gaussian_sigma=[sigma, sigma],
            gaussian_gray_noise_prob=0),
        keys=['img']),
    dict(type=PackInputs)
]

data_root = 'data/denoising_gaussian_test'
cbsd68_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='CBSD68', task_name='denoising'),
        data_root=data_root,
        data_prefix=dict(img='CBSD68', gt='CBSD68'),
        pipeline=test_pipeline))
cbsd68_evaluator = [
    dict(type=PSNR, prefix='CBSD68'),
    dict(type=SSIM, prefix='CBSD68'),
]

kodak24_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='Kodak24', task_name='denoising'),
        data_root=data_root,
        data_prefix=dict(img='Kodak24', gt='Kodak24'),
        pipeline=test_pipeline))
kodak24_evaluator = [
    dict(type=PSNR, prefix='Kodak24'),
    dict(type=SSIM, prefix='Kodak24'),
]

mcmaster_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='McMaster', task_name='denoising'),
        data_root=data_root,
        data_prefix=dict(img='McMaster', gt='McMaster'),
        pipeline=test_pipeline))
mcmaster_evaluator = [
    dict(type=PSNR, prefix='McMaster'),
    dict(type=SSIM, prefix='McMaster'),
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
    cbsd68_dataloader,
    kodak24_dataloader,
    mcmaster_dataloader,
    urban100_dataloader,
]
test_evaluator = [
    cbsd68_evaluator,
    kodak24_evaluator,
    mcmaster_evaluator,
    urban100_evaluator,
]

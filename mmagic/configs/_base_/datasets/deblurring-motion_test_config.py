# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler

from mmagic.datasets import BasicImageDataset
from mmagic.datasets.transforms import LoadImageFromFile, PackInputs
from mmagic.engine.runner import MultiTestLoop
from mmagic.evaluation import PSNR, SSIM

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
    dict(type=PackInputs)
]

gopro_data_root = 'data/gopro/test'
gopro_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='GoPro', task_name='deblurring'),
        data_root=gopro_data_root,
        data_prefix=dict(img='blur', gt='sharp'),
        pipeline=test_pipeline))
gopro_evaluator = [
    dict(type=PSNR, prefix='GoPro'),
    dict(type=SSIM, prefix='GoPro'),
]

hide_data_root = 'data/HIDE'
hide_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='HIDE', task_name='deblurring'),
        data_root=hide_data_root,
        data_prefix=dict(img='input', gt='target'),
        pipeline=test_pipeline))
hide_evaluator = [
    dict(type=PSNR, prefix='HIDE'),
    dict(type=SSIM, prefix='HIDE'),
]

realblurj_data_root = 'data/RealBlur_J'
realblurj_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='RealBlur_J', task_name='deblurring'),
        data_root=realblurj_data_root,
        data_prefix=dict(img='input', gt='target'),
        pipeline=test_pipeline))
realblurj_evaluator = [
    dict(type=PSNR, convert_to='Y', prefix='RealBlurJ'),
    dict(type=SSIM, convert_to='Y', prefix='RealBlurJ'),
]

realblurr_data_root = 'data/RealBlur_R'
realblurr_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='RealBlur_R', task_name='deblurring'),
        data_root=realblurr_data_root,
        data_prefix=dict(img='input', gt='target'),
        pipeline=test_pipeline))
realblurr_evaluator = [
    dict(type=PSNR, convert_to='Y', prefix='RealBlurR'),
    dict(type=SSIM, convert_to='Y', prefix='RealBlurR'),
]

# test config
test_cfg = dict(type=MultiTestLoop)
test_dataloader = [
    gopro_dataloader,
    hide_dataloader,
    realblurj_dataloader,
    realblurr_dataloader,
]
test_evaluator = [
    gopro_evaluator,
    hide_evaluator,
    realblurj_evaluator,
    realblurr_evaluator,
]

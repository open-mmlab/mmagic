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

rain100h_data_root = 'data/Rain100H'
rain100h_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='Rain100H', task_name='deraining'),
        data_root=rain100h_data_root,
        data_prefix=dict(img='input', gt='target'),
        pipeline=test_pipeline))
rain100h_evaluator = [
    dict(type=PSNR, convert_to='Y', prefix='Rain100H'),
    dict(type=SSIM, convert_to='Y', prefix='Rain100H'),
]

rain100l_data_root = 'data/Rain100L'
rain100l_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='Rain100L', task_name='deraining'),
        data_root=rain100l_data_root,
        data_prefix=dict(img='input', gt='target'),
        pipeline=test_pipeline))
rain100l_evaluator = [
    dict(type=PSNR, convert_to='Y', prefix='Rain100L'),
    dict(type=SSIM, convert_to='Y', prefix='Rain100L'),
]

test100_data_root = 'data/Test100'
test100_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='Test100', task_name='deraining'),
        data_root=test100_data_root,
        data_prefix=dict(img='input', gt='target'),
        pipeline=test_pipeline))
test100_evaluator = [
    dict(type=PSNR, convert_to='Y', prefix='Test100'),
    dict(type=SSIM, convert_to='Y', prefix='Test100'),
]

test1200_data_root = 'data/Test1200'
test1200_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='Test1200', task_name='deraining'),
        data_root=test1200_data_root,
        data_prefix=dict(img='input', gt='target'),
        pipeline=test_pipeline))
test1200_evaluator = [
    dict(type=PSNR, convert_to='Y', prefix='Test1200'),
    dict(type=SSIM, convert_to='Y', prefix='Test1200'),
]

test2800_data_root = 'data/Test2800'
test2800_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='Test2800', task_name='deraining'),
        data_root=test2800_data_root,
        data_prefix=dict(img='input', gt='target'),
        pipeline=test_pipeline))
test2800_evaluator = [
    dict(type=PSNR, convert_to='Y', prefix='Test2800'),
    dict(type=SSIM, convert_to='Y', prefix='Test2800'),
]

# test config
test_cfg = dict(type=MultiTestLoop)
test_dataloader = [
    rain100h_dataloader,
    rain100l_dataloader,
    test100_dataloader,
    test1200_dataloader,
    test2800_dataloader,
]
test_evaluator = [
    rain100h_evaluator,
    rain100l_evaluator,
    test100_evaluator,
    test1200_evaluator,
    test2800_evaluator,
]

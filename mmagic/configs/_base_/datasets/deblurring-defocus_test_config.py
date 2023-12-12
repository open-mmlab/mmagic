# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler

from mmagic.datasets import BasicImageDataset
from mmagic.datasets.transforms import LoadImageFromFile, PackInputs
from mmagic.engine.runner import MultiTestLoop
from mmagic.evaluation import MAE, PSNR, SSIM

test_pipeline = [
    dict(
        type=LoadImageFromFile,
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type=LoadImageFromFile,
        key='imgL',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type=LoadImageFromFile,
        key='imgR',
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

dpdd_data_root = 'data/DPDD'

dpdd_indoor_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='DPDD-Indoor', task_name='deblurring'),
        data_root=dpdd_data_root,
        data_prefix=dict(
            img='inputC', imgL='inputL', imgR='inputR', gt='target'),
        ann_file='indoor_labels.txt',
        pipeline=test_pipeline))
dpdd_indoor_evaluator = [
    dict(type=MAE, prefix='DPDD-Indoor'),
    dict(type=PSNR, prefix='DPDD-Indoor'),
    dict(type=SSIM, prefix='DPDD-Indoor'),
]

dpdd_outdoor_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='DPDD-Outdoor', task_name='deblurring'),
        data_root=dpdd_data_root,
        data_prefix=dict(
            img='inputC', imgL='inputL', imgR='inputR', gt='target'),
        ann_file='outdoor_labels.txt',
        pipeline=test_pipeline))
dpdd_outdoor_evaluator = [
    dict(type=MAE, prefix='DPDD-Outdoor'),
    dict(type=PSNR, prefix='DPDD-Outdoor'),
    dict(type=SSIM, prefix='DPDD-Outdoor'),
]

dpdd_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BasicImageDataset,
        metainfo=dict(dataset_type='DPDD-Combined', task_name='deblurring'),
        data_root=dpdd_data_root,
        data_prefix=dict(
            img='inputC', imgL='inputL', imgR='inputR', gt='target'),
        pipeline=test_pipeline))
dpdd_evaluator = [
    dict(type=MAE, prefix='DPDD-Combined'),
    dict(type=PSNR, prefix='DPDD-Combined'),
    dict(type=SSIM, prefix='DPDD-Combined'),
]

# test config
test_cfg = dict(type=MultiTestLoop)
test_dataloader = [
    dpdd_indoor_dataloader,
    dpdd_outdoor_dataloader,
    dpdd_dataloader,
]
test_evaluator = [
    dpdd_indoor_evaluator,
    dpdd_outdoor_evaluator,
    dpdd_evaluator,
]

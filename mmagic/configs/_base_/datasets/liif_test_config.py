# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler

from mmagic.datasets import BasicImageDataset
from mmagic.datasets.transforms import (GenerateCoordinateAndCell,
                                        LoadImageFromFile, PackInputs,
                                        RandomDownSampling)
from mmagic.engine.runner import MultiTestLoop
from mmagic.evaluation import PSNR, SSIM

scale_test_list = [2, 3, 4, 6, 18, 30]

test_pipelines = [[
    dict(
        type=LoadImageFromFile,
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type=RandomDownSampling, scale_min=scale_test, scale_max=scale_test),
    dict(type=GenerateCoordinateAndCell, scale=scale_test, reshape_gt=False),
    dict(type=PackInputs)
] for scale_test in scale_test_list]

# test config for Set5
set5_dataloaders = [
    dict(
        num_workers=4,
        persistent_workers=False,
        drop_last=False,
        sampler=dict(type=DefaultSampler, shuffle=False),
        dataset=dict(
            type=BasicImageDataset,
            metainfo=dict(dataset_type='set5', task_name='sisr'),
            data_root='data/Set5',
            data_prefix=dict(img='LRbicx4', gt='GTmod12'),
            pipeline=test_pipeline)) for test_pipeline in test_pipelines
]
set5_evaluators = [[
    dict(type=PSNR, crop_border=scale, prefix=f'Set5x{scale}'),
    dict(type=SSIM, crop_border=scale, prefix=f'Set5x{scale}'),
] for scale in scale_test_list]

# test config for Set14
set14_dataloaders = [
    dict(
        num_workers=4,
        persistent_workers=False,
        drop_last=False,
        sampler=dict(type=DefaultSampler, shuffle=False),
        dataset=dict(
            type=BasicImageDataset,
            metainfo=dict(dataset_type='set14', task_name='sisr'),
            data_root='data/Set14',
            data_prefix=dict(img='LRbicx4', gt='GTmod12'),
            pipeline=test_pipeline)) for test_pipeline in test_pipelines
]
set14_evaluators = [[
    dict(type=PSNR, crop_border=scale, prefix=f'Set14x{scale}'),
    dict(type=SSIM, crop_border=scale, prefix=f'Set14x{scale}'),
] for scale in scale_test_list]

# test config for DIV2K
div2k_dataloaders = [
    dict(
        num_workers=4,
        persistent_workers=False,
        drop_last=False,
        sampler=dict(type=DefaultSampler, shuffle=False),
        dataset=dict(
            type=BasicImageDataset,
            ann_file='meta_info_DIV2K100sub_GT.txt',
            metainfo=dict(dataset_type='div2k', task_name='sisr'),
            data_root='data/DIV2K',
            data_prefix=dict(
                img='DIV2K_train_LR_bicubic/X4_sub', gt='DIV2K_train_HR_sub'),
            pipeline=test_pipeline)) for test_pipeline in test_pipelines
]
div2k_evaluators = [[
    dict(type=PSNR, crop_border=scale, prefix=f'DIV2Kx{scale}'),
    dict(type=SSIM, crop_border=scale, prefix=f'DIV2Kx{scale}'),
] for scale in scale_test_list]

# test config
test_cfg = dict(type=MultiTestLoop)
test_dataloader = [
    *set5_dataloaders,
    *set14_dataloaders,
    *div2k_dataloaders,
]
test_evaluator = [
    *set5_evaluators,
    *set14_evaluators,
    *div2k_evaluators,
]

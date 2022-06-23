# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import numpy as np
import torch
from mmengine.dataset import Compose

from mmedit.models.mattors import MattorPreprocessor
from mmedit.registry import register_all_modules
from mmedit.transforms import PackEditInputs

DATA_ROOT = Path(__file__).parent.parent.parent / 'data' / 'matting_dataset'
BG_DIR = Path(
    __file__).parent.parent.parent / 'data' / 'matting_dataset' / 'bg'

dim_train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', color_type='grayscale'),
    dict(type='LoadImageFromFile', key='fg'),
    dict(type='LoadImageFromFile', key='bg'),
    dict(type='LoadImageFromFile', key='merged'),
    dict(
        type='CropAroundUnknown',
        keys=['alpha', 'merged', 'fg', 'bg'],
        crop_sizes=[320, 480, 640]),
    dict(type='Flip', keys=['alpha', 'merged', 'fg', 'bg']),
    dict(
        type='Resize',
        keys=['alpha', 'merged', 'fg', 'bg'],
        scale=(320, 320),
        keep_ratio=False),
    dict(type='GenerateTrimap', kernel_size=(1, 30)),
    dict(type='PackEditInputs'),
]

dim_test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='alpha',
        color_type='grayscale',
        save_original_img=True),
    dict(
        type='LoadImageFromFile',
        key='trimap',
        color_type='grayscale',
        save_original_img=True),
    dict(type='LoadImageFromFile', key='merged'),
    dict(type='PackEditInputs'),
]

indexnet_train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', color_type='grayscale'),
    dict(type='LoadImageFromFile', key='fg'),
    dict(type='LoadImageFromFile', key='bg'),
    dict(type='LoadImageFromFile', key='merged'),
    dict(type='GenerateTrimapWithDistTransform', dist_thr=20),
    dict(
        type='CropAroundUnknown',
        keys=['alpha', 'merged', 'fg', 'bg', 'trimap'],
        crop_sizes=[320, 480, 640],
        interpolations=['bicubic', 'bicubic', 'bicubic', 'bicubic',
                        'nearest']),
    dict(
        type='Resize',
        keys=['trimap'],
        scale=(320, 320),
        keep_ratio=False,
        interpolation='nearest'),
    dict(
        type='Resize',
        keys=['alpha', 'merged', 'fg', 'bg'],
        scale=(320, 320),
        keep_ratio=False,
        interpolation='bicubic'),
    dict(type='Flip', keys=['alpha', 'merged', 'fg', 'bg', 'trimap']),
    dict(type='PackEditInputs'),
]

indexnet_test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='alpha',
        color_type='grayscale',
        save_original_img=True),
    dict(
        type='LoadImageFromFile',
        key='trimap',
        color_type='grayscale',
        save_original_img=True),
    dict(type='LoadImageFromFile', key='merged'),
    dict(type='PackEditInputs'),
]

gca_train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', color_type='grayscale'),
    dict(type='LoadImageFromFile', key='fg'),
    dict(type='RandomLoadResizeBg', bg_dir=BG_DIR),
    dict(
        type='CompositeFg',
        fg_dirs=[
            f'{DATA_ROOT}/fg',
        ],
        alpha_dirs=[
            f'{DATA_ROOT}/alpha',
        ]),
    dict(
        type='RandomAffine',
        keys=['alpha', 'fg'],
        degrees=30,
        scale=(0.8, 1.25),
        shear=10,
        flip_ratio=0.5),
    dict(type='GenerateTrimap', kernel_size=(1, 30)),
    dict(type='CropAroundCenter', crop_size=512),
    dict(type='RandomJitter'),
    dict(type='MergeFgAndBg'),
    dict(type='FormatTrimap', to_onehot=True),
    dict(type='PackEditInputs'),
]

gca_test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='alpha',
        color_type='grayscale',
        save_original_img=True),
    dict(
        type='LoadImageFromFile',
        key='trimap',
        color_type='grayscale',
        save_original_img=True),
    dict(type='LoadImageFromFile', key='merged'),
    dict(type='FormatTrimap', to_onehot=True),
    dict(type='PackEditInputs'),
]


def describe(x):
    if isinstance(x, np.ndarray):
        return (type(x).__name__, x.shape, x.dtype, x.min(), x.max())
    elif isinstance(x, torch.Tensor):
        return (type(x).__name__, x.shape, x.dtype, x.min().item(),
                x.max().item())
    else:
        return (type(x).__name__, x)


def test_mattor_pipelines():
    register_all_modules()
    pack = PackEditInputs()
    for name, pipeline, is_training in (
        ('dim_train', dim_train_pipeline, True),
        ('dim_test', dim_test_pipeline, False),
        ('indexnet_train', indexnet_train_pipeline, True),
        ('indexnet_test', indexnet_test_pipeline, False),
        ('gca_train', gca_train_pipeline, True),
        ('gca_test', gca_test_pipeline, False),
    ):
        print(name)
        # from pprint import pprint  # noqa
        # pprint(pipeline)

        # 1. data_info from Dataset
        data_dict = {
            'alpha_path': f'{DATA_ROOT}/alpha/GT05.jpg',
            'trimap_path': f'{DATA_ROOT}/trimap/GT05.png',
            'bg_path': f'{DATA_ROOT}/bg/GT26r.jpg',
            'fg_path': f'{DATA_ROOT}/fg/GT05.jpg',
            'merged_path': f'{DATA_ROOT}/merged/GT05.jpg',
            'sample_idx': 0,
        }

        # 2. data_dict after pipeline, should be np.ndarray from 0 to 255
        p = Compose(pipeline[:-1])
        p(data_dict)

        for k, v in data_dict.items():
            if k not in ['fg', 'bg', 'alpha', 'trimap', 'merged']:
                continue
            # print(k, *describe(v), sep='\t')

        # 3. PackEditInputs, format to CHW and tensor
        result = pack(data_dict)
        assert 'gt_alpha' in result['data_sample']
        assert 'trimap' in result['data_sample']

        for k, v in result['data_sample'].items():
            if k not in ['gt_fg', 'gt_bg', 'gt_alpha', 'trimap', 'gt_merged']:
                continue
            # print(k, *describe(v.data), sep='\t')

        # 4. data_preprocessor
        if name.startswith('gca'):
            preprocessor = MattorPreprocessor(proc_trimap='as_is')
        else:
            preprocessor = MattorPreprocessor()
        inputs, data_samples = preprocessor([result], is_training)
        print('inputs', *describe(inputs[:, :3, :, :]), sep='\t')
        print('trimap', *describe(inputs[:, 3:, :, :]), sep='\t')
        print('trimap.uniques', inputs[:, 3:, :, :].unique())
        if inputs[:, 3:, :, :].size(1) == 3:
            assert (inputs[:, 3:, :, :].sum(dim=1).max().item() == 1)
        for k, v in data_samples[0].items():
            if k not in ['gt_fg', 'gt_bg', 'gt_alpha', 'gt_merged']:
                continue
            print(k, *describe(v.data), sep='\t')

        print('\n')


if __name__ == '__main__':
    test_mattor_pipelines()

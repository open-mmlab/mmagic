# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch

from mmagic.structures import DataSample
from mmagic.visualization import ConcatImageVisualizer


def test_concatimagevisualizer():
    data_sample = DataSample(
        path_rgb='fake_dir/rgb.png',
        path_bgr='fake_dir/bgr.png',
        tensor3d=torch.ones(3, 32, 32) *
        torch.tensor([[[0.1]], [[0.2]], [[0.3]]]),
        array3d=np.ones(shape=(32, 32, 3)) * [0.4, 0.5, 0.6],
        tensor4d=torch.ones(2, 3, 32, 32) * torch.tensor(
            [[[[0.1]], [[0.2]], [[0.3]]], [[[0.4]], [[0.5]], [[0.6]]]]),
    )

    vis = ConcatImageVisualizer(
        fn_key='path_rgb',
        img_keys=['tensor3d', 'array3d', 'tensor4d'],
        vis_backends=[dict(type='LocalVisBackend')],
        save_dir='work_dirs')

    vis.add_datasample(data_sample=data_sample, step=1)

    vis = ConcatImageVisualizer(
        fn_key='path_bgr',
        img_keys=['tensor3d', 'array3d', 'tensor4d'],
        vis_backends=[dict(type='LocalVisBackend')],
        save_dir='work_dirs',
        bgr2rgb=True)

    vis.add_datasample(data_sample=data_sample, step=2)

    for fn in 'rgb_1.png', 'bgr_2.png':
        img = mmcv.imread(f'work_dirs/vis_data/vis_image/{fn}')
        assert img.shape == (64, 16 * 3 * 2, 3)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

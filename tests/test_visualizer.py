# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch

from mmedit.data_element import EditDataSample, PixelData
from mmedit.visualizer import ConcatImageVisualizer


def test_concatimagevisualizer():
    input = torch.rand(1, 3, 32, 32)
    data_sample = EditDataSample(
        path_rgb='fake_dir/rgb.png',
        path_bgr='fake_dir/bgr.png',
        tensor3d=torch.ones(3, 32, 32) *
        torch.tensor([[[0.1]], [[0.2]], [[0.3]]]),
        array3d=np.ones(shape=(32, 32, 3)) * [0.4, 0.5, 0.6],
        tensor4d=torch.ones(2, 3, 32, 32) * torch.tensor(
            [[[[0.1]], [[0.2]], [[0.3]]], [[[0.4]], [[0.5]], [[0.6]]]]),
        pixdata=PixelData(data=torch.ones(1, 32, 32) * 0.6))
    output = EditDataSample(
        outpixdata=PixelData(data=np.ones(shape=(32, 32)) * 0.8))

    vis = ConcatImageVisualizer(
        fn_key='path_rgb',
        img_keys=['tensor3d', 'array3d', 'pixdata', 'outpixdata', 'tensor4d'],
        vis_backends=[dict(type='LocalVisBackend')],
        save_dir='.')
    vis.add_datasample(
        input=input, data_sample=data_sample, output=output, step=1)

    vis = ConcatImageVisualizer(
        fn_key='path_bgr',
        img_keys=['tensor3d', 'array3d', 'pixdata', 'outpixdata', 'tensor4d'],
        vis_backends=[dict(type='LocalVisBackend')],
        save_dir='.',
        bgr2rgb=True)
    vis.add_datasample(
        input=input, data_sample=data_sample, output=output, step=2)

    for fn in 'rgb_1.png', 'bgr_2.png':
        img = mmcv.imread(f'vis_data/vis_image/{fn}')
        assert img.shape == (64, 160, 3)

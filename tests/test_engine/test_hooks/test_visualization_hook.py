# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

import mmcv
import numpy as np
import torch

from mmedit.engine.hooks import BasicVisualizationHook
from mmedit.structures import EditDataSample, PixelData
from mmedit.visualization import ConcatImageVisualizer


class TestVisualizationHook(TestCase):

    def setUp(self) -> None:
        input = torch.rand(2, 3, 32, 32)
        data_sample = EditDataSample(
            path_rgb='rgb.png',
            tensor3d=torch.ones(3, 32, 32) *
            torch.tensor([[[0.1]], [[0.2]], [[0.3]]]),
            array3d=np.ones(shape=(32, 32, 3)) * [0.4, 0.5, 0.6],
            tensor4d=torch.ones(2, 3, 32, 32) * torch.tensor(
                [[[[0.1]], [[0.2]], [[0.3]]], [[[0.4]], [[0.5]], [[0.6]]]]),
            pixdata=PixelData(data=torch.ones(1, 32, 32) * 0.6))
        self.data_batch = {'inputs': input, 'data_samples': [data_sample] * 2}

        output = EditDataSample(
            outpixdata=PixelData(data=np.ones(shape=(32, 32)) * 0.8))
        self.outputs = [output] * 2

        self.vis = ConcatImageVisualizer(
            fn_key='path_rgb',
            img_keys=[
                'tensor3d', 'array3d', 'pixdata', 'outpixdata', 'tensor4d'
            ],
            vis_backends=[dict(type='LocalVisBackend')],
            save_dir='work_dirs')

    def test_after_iter(self):
        runner = Mock()
        runner.iter = 1
        runner.visualizer = self.vis
        hook = BasicVisualizationHook()
        hook._after_iter(runner, 1, self.data_batch, self.outputs)

        img = mmcv.imread('work_dirs/vis_data/vis_image/rgb_None_1.png')
        assert img.shape == (64, 160, 3)

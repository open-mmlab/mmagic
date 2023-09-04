# Copyright (c) OpenMMLab. All rights reserved.
import platform
import unittest

import pytest
import torch

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
class TestInstColorization:

    def test_inst_colorization(self):
        if not torch.cuda.is_available():
            # RoI pooling only support in GPU
            return unittest.skip('test requires GPU and torch+cuda')

        register_all_modules()
        model_cfg = dict(
            type='InstColorization',
            data_preprocessor=dict(
                type='DataPreprocessor',
                mean=[127.5],
                std=[127.5],
            ),
            image_model=dict(
                type='ColorizationNet',
                input_nc=4,
                output_nc=2,
                norm_type='batch'),
            instance_model=dict(
                type='ColorizationNet',
                input_nc=4,
                output_nc=2,
                norm_type='batch'),
            fusion_model=dict(
                type='FusionNet', input_nc=4, output_nc=2, norm_type='batch'),
            color_data_opt=dict(
                ab_thresh=0,
                p=1.0,
                sample_PS=[
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                ],
                ab_norm=110,
                ab_max=110.,
                ab_quant=10.,
                l_norm=100.,
                l_cent=50.,
                mask_cent=0.5),
            which_direction='AtoB',
            loss=dict(type='HuberLoss', delta=.01))

        model = MODELS.build(model_cfg)

        # test attributes
        assert model.__class__.__name__ == 'InstColorization'

        # prepare data
        inputs = torch.rand(1, 3, 256, 256)
        target_shape = (1, 3, 256, 256)

        data_sample = DataSample(gt_img=inputs)
        metainfo = dict(
            cropped_img=torch.rand(9, 3, 256, 256),
            box_info=torch.tensor([[175, 29, 96, 54, 52, 106],
                                   [14, 191, 84, 61, 51, 111],
                                   [117, 64, 115, 46, 75, 95],
                                   [41, 165, 121, 47, 50, 88],
                                   [46, 136, 94, 45, 74, 117],
                                   [79, 124, 62, 115, 53, 79],
                                   [156, 64, 77, 138, 36, 41],
                                   [200, 48, 114, 131, 8, 11],
                                   [115, 78, 92, 81, 63, 83]]),
            box_info_2x=torch.tensor([[87, 15, 48, 27, 26, 53],
                                      [7, 96, 42, 31, 25, 55],
                                      [58, 32, 57, 23, 38, 48],
                                      [20, 83, 60, 24, 25, 44],
                                      [23, 68, 47, 23, 37, 58],
                                      [39, 62, 31, 58, 27, 39],
                                      [78, 32, 38, 69, 18, 21],
                                      [100, 24, 57, 66, 4, 5],
                                      [57, 39, 46, 41, 32, 41]]),
            box_info_4x=torch.tensor([[43, 8, 24, 14, 13, 26],
                                      [3, 48, 21, 16, 13, 27],
                                      [29, 16, 28, 12, 19, 24],
                                      [10, 42, 30, 12, 12, 22],
                                      [11, 34, 23, 12, 19, 29],
                                      [19, 31, 15, 29, 14, 20],
                                      [39, 16, 19, 35, 9, 10],
                                      [50, 12, 28, 33, 2, 3],
                                      [28, 20, 23, 21, 16, 20]]),
            box_info_8x=torch.tensor([[21, 4, 12, 7, 7, 13],
                                      [1, 24, 10, 8, 7, 14],
                                      [14, 8, 14, 6, 10, 12],
                                      [5, 21, 15, 6, 6, 11],
                                      [5, 17, 11, 6, 10, 15],
                                      [9, 16, 7, 15, 7, 10],
                                      [19, 8, 9, 18, 5, 5],
                                      [25, 6, 14, 17, 1, 1],
                                      [14, 10, 11, 11, 8, 10]]),
            empty_box=False)
        data_sample.set_metainfo(metainfo=metainfo)

        data = dict(
            inputs=inputs, data_samples=DataSample.stack([data_sample]))

        res = model(mode='tensor', **data)

        assert torch.is_tensor(res)
        assert res.shape == target_shape


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

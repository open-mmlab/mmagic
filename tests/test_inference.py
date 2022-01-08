# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmedit.apis import (init_model, restoration_video_inference,
                         video_interpolation_inference)
from mmedit.models.registry import MODELS


def test_restoration_video_inference():
    if torch.cuda.is_available():
        # recurrent framework (BasicVSR)
        model = init_model(
            './configs/restorers/basicvsr/basicvsr_reds4.py',
            None,
            device='cuda')
        img_dir = './tests/data/vimeo90k/00001/0266'
        window_size = 0
        start_idx = 1
        filename_tmpl = 'im{}.png'

        output = restoration_video_inference(model, img_dir, window_size,
                                             start_idx, filename_tmpl)
        assert output.shape == (1, 7, 3, 256, 448)

        # sliding-window framework (EDVR)
        window_size = 5
        model = init_model(
            './configs/restorers/edvr/edvrm_wotsa_x4_g8_600k_reds.py',
            None,
            device='cuda')
        output = restoration_video_inference(model, img_dir, window_size,
                                             start_idx, filename_tmpl)
        assert output.shape == (1, 7, 3, 256, 448)

        # without demo_pipeline
        model.cfg.test_pipeline = model.cfg.demo_pipeline
        model.cfg.pop('demo_pipeline')
        output = restoration_video_inference(model, img_dir, window_size,
                                             start_idx, filename_tmpl)
        assert output.shape == (1, 7, 3, 256, 448)

        # without test_pipeline and demo_pipeline
        model.cfg.val_pipeline = model.cfg.test_pipeline
        model.cfg.pop('test_pipeline')
        output = restoration_video_inference(model, img_dir, window_size,
                                             start_idx, filename_tmpl)
        assert output.shape == (1, 7, 3, 256, 448)

        # recurrent framework (BasicVSR)
        # the first element in the pipeline must be 'GenerateSegmentIndices'
        with pytest.raises(TypeError):
            model.cfg.val_pipeline = model.cfg.val_pipeline[1:]
            output = restoration_video_inference(model, img_dir, window_size,
                                                 start_idx, filename_tmpl)


@MODELS.register_module()
class InterpolateExample(nn.Module):
    """An example of interpolate network for testing BasicInterpolater.
    """

    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(3, 3, 3, 1, 1)

    def forward(self, x):
        return self.layer(x[:, 0])

    def init_weights(self, pretrained=None):
        pass


def test_video_interpolation_inference():
    if torch.cuda.is_available():
        model = init_model(
            './configs/restorers/basicvsr/basicvsr_reds4.py',
            None,
            device='cuda')
        model.cfg['demo_pipeline'] = [
            dict(
                type='LoadImageFromFileList',
                io_backend='disk',
                key='inputs',
                channel_order='rgb'),
            dict(type='RescaleToZeroOne', keys=['inputs']),
            dict(type='FramesToTensor', keys=['inputs']),
            dict(
                type='Collect',
                keys=['inputs'],
                meta_keys=['inputs_path', 'key'])
        ]

        model.cfg['model']['io_sequence'] = [1, 3, 1]

        input_dir = './tests/data/vimeo90k/00001/0266'
        output, fps = video_interpolation_inference(
            model, input_dir, batch_size=10)
        assert isinstance(output, list)
        assert isinstance(fps, float)

        input_dir = './tests/data/test_inference.mp4'
        output, fps = video_interpolation_inference(model, input_dir)
        assert isinstance(output, list)
        assert isinstance(fps, float)


if __name__ == '__main__':
    test_video_interpolation_inference()

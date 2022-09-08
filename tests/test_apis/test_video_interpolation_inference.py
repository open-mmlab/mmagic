# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil

import pytest
import torch

from mmedit.apis import init_model, video_interpolation_inference


def test_video_interpolation_inference():
    model = init_model(
        './configs/video_interpolators/cain/cain_b5_g1b32_vimeo90k_triplet.py',
        None,
        device='cpu')
    model.cfg['demo_pipeline'] = [
        dict(
            type='LoadImageFromFileList',
            io_backend='disk',
            key='inputs',
            channel_order='rgb'),
        dict(type='RescaleToZeroOne', keys=['inputs']),
        dict(type='FramesToTensor', keys=['inputs']),
        dict(
            type='Collect', keys=['inputs'], meta_keys=['inputs_path', 'key'])
    ]

    input_dir = './tests/data/vimeo90k/00001/0266'
    output_dir = './tests/data/vimeo90k/00001/out'
    os.mkdir(output_dir)
    video_interpolation_inference(model, input_dir, output_dir, batch_size=10)

    input_dir = './tests/data/test_inference.mp4'
    output_dir = './tests/data/test_inference_out.mp4'
    video_interpolation_inference(model, input_dir, output_dir)

    with pytest.raises(AssertionError):
        input_dir = './tests/data/test_inference.mp4'
        output_dir = './tests/data/test_inference_out.mp4'
        video_interpolation_inference(
            model, input_dir, output_dir, fps_multiplier=-1)

    if torch.cuda.is_available():
        model = init_model(
            './configs/video_interpolators/cain/'
            'cain_b5_g1b32_vimeo90k_triplet.py',
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

        input_dir = './tests/data/vimeo90k/00001/0266'
        output_dir = './tests/data/vimeo90k/00001'
        video_interpolation_inference(
            model, input_dir, output_dir, batch_size=10)

        input_dir = './tests/data/test_inference.mp4'
        output_dir = './tests/data/test_inference_out.mp4'
        video_interpolation_inference(model, input_dir, output_dir)

        with pytest.raises(AssertionError):
            input_dir = './tests/data/test_inference.mp4'
            output_dir = './tests/data/test_inference_out.mp4'
            video_interpolation_inference(
                model, input_dir, output_dir, fps_multiplier=-1)

    shutil.rmtree('./tests/data/vimeo90k/00001/out')
    os.remove('./tests/data/test_inference_out.mp4')

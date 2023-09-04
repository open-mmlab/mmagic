# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil

import numpy as np
import pytest
from mmengine import Config

from mmagic.apis.inferencers.eg3d_inferencer import EG3DInferencer
from mmagic.utils import register_all_modules

register_all_modules()

config = dict(
    model=dict(
        type='EG3D',
        generator=dict(
            type='TriplaneGenerator',
            out_size=32,
            noise_size=8,
            style_channels=8,
            num_mlps=1,
            triplane_size=8,
            triplane_channels=4,
            sr_in_size=8,
            sr_in_channels=8,
            neural_rendering_resolution=5,
            cond_scale=1,
            renderer_cfg=dict(
                ray_start=0.1,
                ray_end=2.6,
                box_warp=1.6,
                depth_resolution=4,
                white_back=True,
                depth_resolution_importance=4,
            ),
            rgb2bgr=True),
        camera=dict(
            type='UniformCamera',
            horizontal_mean=3.141,
            horizontal_std=3.141,
            vertical_mean=3.141 / 2,
            vertical_std=3.141 / 2,
            focal=1.025390625,
            up=[0, 0, 1],
            radius=1.2),
        data_preprocessor=dict(type='DataPreprocessor')))


def test_eg3d_inferencer():
    cfg = Config(config)

    result_out_dir = osp.join(
        osp.dirname(__file__), '..', '..', 'data/out', 'eg3d_output')

    inferencer_instance = EG3DInferencer(cfg, None)
    output = inferencer_instance(
        vis_mode='both',
        save_img=False,
        save_video=True,
        interpolation='camera',
        num_images=10,
        result_out_dir=result_out_dir)
    # contrust target file list
    target_file_list = [
        'fake_img_seed2022.mp4', 'depth_seed2022.mp4', 'combine_seed2022.mp4'
    ]
    assert set(target_file_list) == set(os.listdir(result_out_dir))
    assert isinstance(output, dict)
    shutil.rmtree(result_out_dir)

    output = inferencer_instance(
        vis_mode='both',
        save_img=True,
        save_video=False,
        interpolation='camera',
        num_images=10,
        result_out_dir=result_out_dir)
    # contrust target file list
    target_file_list = [
        f'fake_img_frame{idx}_seed2022.png' for idx in range(10)
    ]
    target_file_list += [f'depth_frame{idx}_seed2022.png' for idx in range(10)]
    target_file_list += [
        f'combine_frame{idx}_seed2022.png' for idx in range(10)
    ]
    assert set(target_file_list) == set(os.listdir(result_out_dir))
    assert isinstance(output, dict)
    shutil.rmtree(result_out_dir)

    output = inferencer_instance(
        vis_mode='depth',
        save_img=False,
        interpolation='camera',
        num_images=2,
        result_out_dir=result_out_dir)
    # contrust target file list
    target_file_list = ['depth_seed2022.mp4']
    assert set(target_file_list) == set(os.listdir(result_out_dir))
    assert isinstance(output, dict)
    shutil.rmtree(result_out_dir)

    output = inferencer_instance(
        vis_mode='depth',
        save_video=False,
        save_img=False,
        interpolation='camera',
        sample_model='orig',
        num_images=2,
        result_out_dir=result_out_dir)
    assert len(os.listdir(result_out_dir)) == 0

    # test cond input
    inferencer_instance = EG3DInferencer(
        cfg, None, extra_parameters=dict(sample_model='orig'))
    cond_input = [np.random.randn(25) for _ in range(2)]
    output = inferencer_instance(
        inputs=cond_input,
        vis_mode='both',
        save_video=True,
        save_img=False,
        interpolation='camera',
        num_images=2,
        result_out_dir=result_out_dir)
    # contrust target file list
    target_file_list = [
        'fake_img_seed2022.png',
        'depth_seed2022.png',
        'combine_seed2022.png',
    ]
    assert set(target_file_list) == set(os.listdir(result_out_dir))
    assert isinstance(output, dict)
    shutil.rmtree(result_out_dir)

    cond_input = [[0.111 for _ in range(25)] for _ in range(2)]
    output = inferencer_instance(
        inputs=cond_input,
        vis_mode='depth',
        save_video=False,
        save_img=True,
        interpolation='camera',
        num_images=2,
        result_out_dir=result_out_dir)
    target_file_list = ['depth_seed2022.png']
    assert set(target_file_list) == set(os.listdir(result_out_dir))
    assert isinstance(output, dict)
    shutil.rmtree(result_out_dir)

    cond_input = [['wrong'], ['input']]
    with pytest.raises(AssertionError):
        inferencer_instance(
            inputs=cond_input,
            vis_mode='depth',
            save_video=False,
            save_img=False,
            interpolation='camera',
            num_images=2,
            result_out_dir=result_out_dir)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

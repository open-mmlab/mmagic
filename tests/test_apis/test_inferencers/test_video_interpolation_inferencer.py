# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmagic.apis.inferencers.video_interpolation_inferencer import \
    VideoInterpolationInferencer
from mmagic.utils import register_all_modules

register_all_modules()


def test_video_interpolation_inferencer():
    cfg = osp.join(
        osp.dirname(__file__), '..', '..', '..', 'configs', 'flavr',
        'flavr_in4out1_8xb4_vimeo90k-septuplet.py')
    result_out_dir = osp.join(
        osp.dirname(__file__), '..', '..', 'data/out',
        'video_interpolation_result.mp4')
    data_root = osp.join(osp.dirname(__file__), '../../../')
    video_path = data_root + 'tests/data/frames/test_inference.mp4'

    inferencer_instance = \
        VideoInterpolationInferencer(cfg,
                                     None)
    inference_result = inferencer_instance(
        video=video_path, result_out_dir=result_out_dir)
    assert inference_result is None


def test_video_interpolation_inferencer_input_dir():
    data_root = osp.join(osp.dirname(__file__), '../../../')
    config = data_root + 'configs/cain/cain_g1b32_1xb5_vimeo90k-triplet.py'
    video_path = data_root + 'tests/data/frames/sequence/gt/sequence_1'
    result_out_dir = data_root + 'tests/data/out'

    inferencer_instance = \
        VideoInterpolationInferencer(config,
                                     None,
                                     extra_parameters={'fps': 60})
    inference_result = inferencer_instance(
        video=video_path, result_out_dir=result_out_dir)
    assert inference_result is None


def test_video_interpolation_inferencer_fps_multiplier():
    result_out_dir = osp.join(
        osp.dirname(__file__), '..', '..', 'data',
        'video_interpolation_result.mp4')
    data_root = osp.join(osp.dirname(__file__), '../../../')
    cfg = data_root + 'configs/cain/cain_g1b32_1xb5_vimeo90k-triplet.py'
    video_path = data_root + 'tests/data/frames/test_inference.mp4'

    inferencer_instance = \
        VideoInterpolationInferencer(cfg,
                                     None,
                                     extra_parameters={'fps_multiplier': 2})
    inference_result = inferencer_instance(
        video=video_path, result_out_dir=result_out_dir)
    assert inference_result is None


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

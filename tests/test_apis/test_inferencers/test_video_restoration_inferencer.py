# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import sys

import pytest

from mmedit.apis.inferencers.video_restoration_inferencer import \
    VideoRestorationInferencer
from mmedit.utils import register_all_modules

register_all_modules()


@pytest.mark.skipif(
    sys.version_info < (3, 7), reason='skip because python version is old.')
def test_video_restoration_inferencer():
    cfg = osp.join(
        osp.dirname(__file__), '..', '..', '..', 'configs', 'basicvsr',
        'basicvsr_2xb4_reds4.py')
    result_out_dir = osp.join(
        osp.dirname(__file__), '..', '..', 'data/out',
        'video_restoration_result.mp4')
    data_root = osp.join(osp.dirname(__file__), '../../../')
    video_path = data_root + 'tests/data/frames/test_inference.mp4'

    inferencer_instance = \
        VideoRestorationInferencer(
            cfg,
            None)
    inference_result = inferencer_instance(
        video=video_path, result_out_dir=result_out_dir)
    assert inference_result is None


@pytest.mark.skipif(
    sys.version_info < (3, 7), reason='skip because python version is old.')
def test_video_restoration_inferencer_input_dir():
    cfg = osp.join(
        osp.dirname(__file__), '..', '..', '..', 'configs', 'basicvsr',
        'basicvsr_2xb4_reds4.py')
    result_out_dir = osp.join(
        osp.dirname(__file__), '..', '..', 'data/out',
        'video_restoration_result.mp4')
    data_root = osp.join(osp.dirname(__file__), '../../../')
    input_dir = osp.join(data_root, 'tests/data/frames/sequence/gt/sequence_1')
    result_out_dir = data_root + 'tests/data/out'

    inferencer_instance = \
        VideoRestorationInferencer(
            cfg,
            None)
    inference_result = inferencer_instance(
        video=input_dir, result_out_dir=result_out_dir)
    assert inference_result is None


@pytest.mark.skipif(
    sys.version_info < (3, 7), reason='skip because python version is old.')
def test_video_restoration_inferencer_window_size():
    cfg = osp.join(
        osp.dirname(__file__), '..', '..', '..', 'configs', 'basicvsr',
        'basicvsr_2xb4_reds4.py')
    result_out_dir = osp.join(
        osp.dirname(__file__), '..', '..', 'data/out',
        'video_restoration_result.mp4')
    data_root = osp.join(osp.dirname(__file__), '../../../')
    video_path = data_root + 'tests/data/frames/test_inference.mp4'

    extra_parameters = {'window_size': 3}

    inferencer_instance = \
        VideoRestorationInferencer(
            cfg,
            None,
            extra_parameters=extra_parameters)
    inference_result = inferencer_instance(
        video=video_path, result_out_dir=result_out_dir)
    assert inference_result is None


@pytest.mark.skipif(
    sys.version_info < (3, 7), reason='skip because python version is old.')
def test_video_restoration_inferencer_max_seq_len():
    cfg = osp.join(
        osp.dirname(__file__), '..', '..', '..', 'configs', 'basicvsr',
        'basicvsr_2xb4_reds4.py')
    result_out_dir = osp.join(
        osp.dirname(__file__), '..', '..', 'data/out',
        'video_restoration_result.mp4')
    data_root = osp.join(osp.dirname(__file__), '../../../')
    video_path = data_root + 'tests/data/frames/test_inference.mp4'

    extra_parameters = {'max_seq_len': 3}

    inferencer_instance = \
        VideoRestorationInferencer(
            cfg,
            None,
            extra_parameters=extra_parameters)
    inference_result = inferencer_instance(
        video=video_path, result_out_dir=result_out_dir)
    assert inference_result is None

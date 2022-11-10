# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmedit.apis.inferencers.video_restoration_inferencer import \
    VideoRestorationInferencer
from mmedit.utils import register_all_modules

register_all_modules()


def test_video_restoration_inferencer():
    cfg = osp.join(
        osp.dirname(__file__), '..', '..', '..', 'configs', 'basicvsr',
        'basicvsr_2xb4_reds4.py')
    result_out_dir = osp.join(
        osp.dirname(__file__), '..', '..', 'data',
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


if __name__ == '__main__':
    test_video_restoration_inferencer()

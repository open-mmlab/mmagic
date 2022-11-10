# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmedit.apis.inferencers.video_interpolation_inferencer import \
    VideoInterpolationInferencer
from mmedit.utils import register_all_modules

register_all_modules()


def test_video_interpolation_inferencer():
    cfg = osp.join(
        osp.dirname(__file__), '..', '..', '..', 'configs', 'flavr',
        'flavr_in4out1_8xb4_vimeo90k-septuplet.py')
    result_out_dir = osp.join(
        osp.dirname(__file__), '..', '..', 'data',
        'video_interpolation_result.mp4')
    data_root = osp.join(osp.dirname(__file__), '../../../')
    video_path = data_root + 'tests/data/frames/test_inference.mp4'

    inferencer_instance = \
        VideoInterpolationInferencer(cfg,
                                     None)
    inference_result = inferencer_instance(
        video=video_path, result_out_dir=result_out_dir)
    assert inference_result is None


if __name__ == '__main__':
    test_video_interpolation_inferencer()

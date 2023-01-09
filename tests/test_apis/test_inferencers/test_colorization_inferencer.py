# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmedit.apis.inferencers.colorization_inferencer import \
    ColorizationInferencer
from mmedit.utils import register_all_modules

register_all_modules()


def test_colorization_inferencer():
    cfg = osp.join(
        osp.dirname(__file__), '..', '..', '..', 'configs',
        'inst_colorization',
        'inst-colorizatioon_full_official_cocostuff-256x256.py')
    data_path = osp.join(
        osp.dirname(__file__), '..', '..', 'data', 'unpaired', 'trainA',
        '1.jpg')
    result_out_dir = osp.join(
        osp.dirname(__file__), '..', '..', 'data',
        'inst_colorization_result.jpg')

    inferencer_instance = \
        ColorizationInferencer(cfg, None)
    inferencer_instance(img=data_path)
    inference_result = inferencer_instance(
        img=data_path, result_out_dir=result_out_dir)
    result_img = inference_result[1]
    assert result_img.shape == (256, 256, 3)

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import unittest

import torch

from mmagic.apis.inferencers.colorization_inferencer import \
    ColorizationInferencer
from mmagic.utils import register_all_modules

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
        osp.dirname(__file__), '..', '..', 'data/out',
        'inst_colorization_result.jpg')

    inferencer_instance = \
        ColorizationInferencer(cfg, None)
    del inferencer_instance.model.cfg.test_pipeline[1]

    inferencer_instance.preprocess(img=data_path)

    preds = torch.rand((1, 3, 256, 256))
    result_img = inferencer_instance.visualize(preds=preds)
    result_img = inferencer_instance.visualize(
        preds=preds, result_out_dir=result_out_dir)

    if not torch.cuda.is_available():
        # RoI pooling only support in GPU
        return unittest.skip('test requires GPU and torch+cuda')

    inferencer_instance(img=data_path)
    inference_result = inferencer_instance(
        img=data_path, result_out_dir=result_out_dir)
    result_img = inference_result[1]
    assert result_img.shape == (256, 256, 3)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

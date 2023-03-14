# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform

import pytest
import torch

from mmedit.apis.inferencers.restoration_inferencer import \
    RestorationInferencer
from mmedit.utils import register_all_modules

register_all_modules()


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_restoration_inferencer():
    data_root = osp.join(osp.dirname(__file__), '../../../')
    config = data_root + 'configs/esrgan/esrgan_x4c64b23g32_1xb16-400k_div2k.py'  # noqa
    img_path = data_root + 'tests/data/image/lq/baboon_x4.png'
    result_out_dir = osp.join(
        osp.dirname(__file__), '..', '..', 'data/out',
        'restoration_result.png')

    inferencer_instance = \
        RestorationInferencer(config, None)
    inferencer_instance(img=img_path)
    inference_result = inferencer_instance(
        img=img_path, result_out_dir=result_out_dir)
    result_img = inference_result[1]
    assert result_img.shape == (480, 500, 3)

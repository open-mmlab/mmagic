# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

from mmedit.apis.inferencers.base_mmedit_inferencer import BaseMMEditInferencer
from mmedit.utils import register_all_modules

register_all_modules()


def test_base_mmedit_inferencer():
    with pytest.raises(Exception):
        BaseMMEditInferencer(1, None)

    cfg = osp.join(
        osp.dirname(__file__), '..', '..', '..', 'configs', 'sngan_proj',
        'sngan-proj_woReLUinplace_lr2e-4-ndisc5-1xb64_cifar10-32x32.py')

    with pytest.raises(Exception):
        BaseMMEditInferencer(cfg, 'test')

    inferencer_instance = BaseMMEditInferencer(cfg, None)
    extra_parameters = inferencer_instance.get_extra_parameters()
    assert len(extra_parameters) == 0

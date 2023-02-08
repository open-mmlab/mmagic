# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.registry import BACKBONES
from mmedit.utils import register_all_modules

register_all_modules()


def test_gl_enc():
    input_x = torch.randn(1, 4, 256, 256)
    template_cfg = dict(type='AOTEncoderDecoder')

    cfg_ = template_cfg.copy()
    cfg_['encoder'] = dict(type='AOTEncoder')
    aot_encdec = BACKBONES.build(cfg_)
    output = aot_encdec(input_x)
    assert output.shape == (1, 3, 256, 256)

# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules


def test_gl_encoder():
    register_all_modules()
    input_x = torch.randn(1, 4, 256, 256)
    template_cfg = dict(type='GLEncoder')

    gl_encoder = MODELS.build(template_cfg)
    output = gl_encoder(input_x)
    assert output.shape == (1, 256, 64, 64)

    if torch.cuda.is_available():
        gl_encoder = MODELS.build(template_cfg)
        gl_encoder = gl_encoder.cuda()
        output = gl_encoder(input_x.cuda())
        assert output.shape == (1, 256, 64, 64)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules


def test_gl_decoder():
    register_all_modules()
    input_x = torch.randn(1, 256, 64, 64)
    template_cfg = dict(type='GLDecoder')

    gl_decoder = MODELS.build(template_cfg)
    output = gl_decoder(input_x)
    assert output.shape == (1, 3, 256, 256)

    cfg_copy = template_cfg.copy()
    cfg_copy['out_act'] = 'sigmoid'
    gl_decoder = MODELS.build(cfg_copy)
    output = gl_decoder(input_x)
    assert output.shape == (1, 3, 256, 256)

    with pytest.raises(ValueError):
        # conv_cfg must be a dict or None
        cfg_copy = template_cfg.copy()
        cfg_copy['out_act'] = 'relu'
        gl_decoder = MODELS.build(cfg_copy)
        output = gl_decoder(input_x)

    if torch.cuda.is_available():
        gl_decoder = MODELS.build(template_cfg)
        gl_decoder = gl_decoder.cuda()
        output = gl_decoder(input_x.cuda())
        assert output.shape == (1, 3, 256, 256)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules


def test_gl_encdec():
    register_all_modules()
    input_x = torch.randn(1, 4, 256, 256)
    template_cfg = dict(type='GLEncoderDecoder')

    gl_encdec = MODELS.build(template_cfg)
    gl_encdec.init_weights()
    output = gl_encdec(input_x)
    assert output.shape == (1, 3, 256, 256)

    cfg_ = template_cfg.copy()
    cfg_['decoder'] = dict(type='GLDecoder', out_act='sigmoid')
    gl_encdec = MODELS.build(cfg_)
    output = gl_encdec(input_x)
    assert output.shape == (1, 3, 256, 256)

    with pytest.raises(ValueError):
        cfg_ = template_cfg.copy()
        cfg_['decoder'] = dict(type='GLDecoder', out_act='igccc')
        gl_encdec = MODELS.build(cfg_)

    with pytest.raises(TypeError):
        gl_encdec.init_weights(pretrained=dict(igccc=4396))

    if torch.cuda.is_available():
        gl_encdec = MODELS.build(template_cfg)
        gl_encdec.init_weights()
        gl_encdec = gl_encdec.cuda()
        output = gl_encdec(input_x.cuda())
        assert output.shape == (1, 3, 256, 256)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

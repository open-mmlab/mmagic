# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()


def test_gl_encdec():
    input_x = torch.randn(1, 4, 64, 64)
    template_cfg = dict(type='AOTEncoderDecoder')

    aot_encdec = MODELS.build(template_cfg)
    aot_encdec.init_weights()
    output = aot_encdec(input_x)
    assert output.shape == (1, 3, 64, 64)

    cfg_ = template_cfg.copy()
    cfg_['encoder'] = dict(type='AOTEncoder')
    aot_encdec = MODELS.build(cfg_)
    output = aot_encdec(input_x)
    assert output.shape == (1, 3, 64, 64)

    cfg_ = template_cfg.copy()
    cfg_['decoder'] = dict(type='AOTDecoder')
    aot_encdec = MODELS.build(cfg_)
    output = aot_encdec(input_x)
    assert output.shape == (1, 3, 64, 64)

    if torch.cuda.is_available():
        aot_encdec = MODELS.build(template_cfg)
        aot_encdec.init_weights()
        aot_encdec = aot_encdec.cuda()
        output = aot_encdec(input_x.cuda())
        assert output.shape == (1, 3, 64, 64)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

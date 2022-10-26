# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.registry import BACKBONES


def test_gl_encdec():
    input_x = torch.randn(1, 4, 256, 256)
    template_cfg = dict(type='AOTEncoderDecoder')

    aot_encdec = BACKBONES.build(template_cfg)
    aot_encdec.init_weights()
    output = aot_encdec(input_x)
    assert output.shape == (1, 3, 256, 256)

    cfg_ = template_cfg.copy()
    cfg_['encoder'] = dict(type='AOTEncoder')
    aot_encdec = BACKBONES.build(cfg_)
    output = aot_encdec(input_x)
    assert output.shape == (1, 3, 256, 256)

    cfg_ = template_cfg.copy()
    cfg_['decoder'] = dict(type='AOTDecoder')
    aot_encdec = BACKBONES.build(cfg_)
    output = aot_encdec(input_x)
    assert output.shape == (1, 3, 256, 256)

    if torch.cuda.is_available():
        aot_encdec = BACKBONES.build(template_cfg)
        aot_encdec.init_weights()
        aot_encdec = aot_encdec.cuda()
        output = aot_encdec(input_x.cuda())
        assert output.shape == (1, 3, 256, 256)

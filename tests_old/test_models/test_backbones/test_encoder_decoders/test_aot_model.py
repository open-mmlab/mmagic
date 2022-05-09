# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models import build_backbone
from mmedit.models.backbones.encoder_decoders.necks import AOTBlockNeck


def test_gl_encdec():
    input_x = torch.randn(1, 4, 256, 256)
    template_cfg = dict(type='AOTEncoderDecoder')

    aot_encdec = build_backbone(template_cfg)
    aot_encdec.init_weights()
    output = aot_encdec(input_x)
    assert output.shape == (1, 3, 256, 256)

    cfg_ = template_cfg.copy()
    cfg_['encoder'] = dict(type='AOTEncoder')
    aot_encdec = build_backbone(cfg_)
    output = aot_encdec(input_x)
    assert output.shape == (1, 3, 256, 256)

    cfg_ = template_cfg.copy()
    cfg_['decoder'] = dict(type='AOTDecoder')
    aot_encdec = build_backbone(cfg_)
    output = aot_encdec(input_x)
    assert output.shape == (1, 3, 256, 256)

    if torch.cuda.is_available():
        aot_encdec = build_backbone(template_cfg)
        aot_encdec.init_weights()
        aot_encdec = aot_encdec.cuda()
        output = aot_encdec(input_x.cuda())
        assert output.shape == (1, 3, 256, 256)


def test_aot_dilation_neck():
    neck = AOTBlockNeck(
        in_channels=256, dilation_rates=(1, 2, 4, 8), num_aotblock=8)
    x = torch.rand((2, 256, 64, 64))
    res = neck(x)
    assert res.shape == (2, 256, 64, 64)

    if torch.cuda.is_available():
        neck = AOTBlockNeck(
            in_channels=256, dilation_rates=(1, 2, 4, 8),
            num_aotblock=8).cuda()
        x = torch.rand((2, 256, 64, 64)).cuda()
        res = neck(x)
        assert res.shape == (2, 256, 64, 64)

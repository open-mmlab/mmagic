# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models import build_backbone, build_component
from mmedit.models.backbones import GLDilationNeck
from mmedit.models.common import SimpleGatedConvModule


def test_gl_encdec():
    input_x = torch.randn(1, 4, 256, 256)
    template_cfg = dict(type='GLEncoderDecoder')

    gl_encdec = build_backbone(template_cfg)
    gl_encdec.init_weights()
    output = gl_encdec(input_x)
    assert output.shape == (1, 3, 256, 256)

    cfg_ = template_cfg.copy()
    cfg_['decoder'] = dict(type='GLDecoder', out_act='sigmoid')
    gl_encdec = build_backbone(cfg_)
    output = gl_encdec(input_x)
    assert output.shape == (1, 3, 256, 256)

    with pytest.raises(ValueError):
        cfg_ = template_cfg.copy()
        cfg_['decoder'] = dict(type='GLDecoder', out_act='igccc')
        gl_encdec = build_backbone(cfg_)

    with pytest.raises(TypeError):
        gl_encdec.init_weights(pretrained=dict(igccc=4396))

    if torch.cuda.is_available():
        gl_encdec = build_backbone(template_cfg)
        gl_encdec.init_weights()
        gl_encdec = gl_encdec.cuda()
        output = gl_encdec(input_x.cuda())
        assert output.shape == (1, 3, 256, 256)


def test_gl_dilation_neck():
    neck = GLDilationNeck(in_channels=8)
    x = torch.rand((2, 8, 64, 64))
    res = neck(x)
    assert res.shape == (2, 8, 64, 64)

    if torch.cuda.is_available():
        neck = GLDilationNeck(in_channels=8).cuda()
        x = torch.rand((2, 8, 64, 64)).cuda()
        res = neck(x)
        assert res.shape == (2, 8, 64, 64)

        neck = GLDilationNeck(in_channels=8, conv_type='gated_conv').cuda()
        res = neck(x)
        assert isinstance(neck.dilation_convs[0], SimpleGatedConvModule)
        assert res.shape == (2, 8, 64, 64)


def test_gl_discs():
    global_disc_cfg = dict(
        in_channels=3,
        max_channels=512,
        fc_in_channels=512 * 4 * 4,
        fc_out_channels=1024,
        num_convs=6,
        norm_cfg=dict(type='BN'))
    local_disc_cfg = dict(
        in_channels=3,
        max_channels=512,
        fc_in_channels=512 * 4 * 4,
        fc_out_channels=1024,
        num_convs=5,
        norm_cfg=dict(type='BN'))
    gl_disc_cfg = dict(
        type='GLDiscs',
        global_disc_cfg=global_disc_cfg,
        local_disc_cfg=local_disc_cfg)
    gl_discs = build_component(gl_disc_cfg)
    gl_discs.init_weights()

    input_g = torch.randn(1, 3, 256, 256)
    input_l = torch.randn(1, 3, 128, 128)
    output = gl_discs((input_g, input_l))
    assert output.shape == (1, 1)

    with pytest.raises(TypeError):
        gl_discs.init_weights(pretrained=dict(igccc=777))

    if torch.cuda.is_available():
        gl_discs = gl_discs.cuda()
        input_g = torch.randn(1, 3, 256, 256).cuda()
        input_l = torch.randn(1, 3, 128, 128).cuda()
        output = gl_discs((input_g, input_l))
        assert output.shape == (1, 1)

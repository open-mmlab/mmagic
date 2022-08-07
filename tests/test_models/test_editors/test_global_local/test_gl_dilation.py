# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models import build_component
from mmedit.models.inpaintors import GLDilationNeck
from mmedit.models.inpaintors.modules import SimpleGatedConvModule


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

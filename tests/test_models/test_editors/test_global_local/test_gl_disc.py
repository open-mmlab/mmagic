# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.registry import MODELS


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
    gl_discs = MODELS.build(gl_disc_cfg)
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


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

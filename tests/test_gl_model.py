import pytest
import torch
from mmedit.models import build_backbone, build_component
from mmedit.models.components import MultiLayerDiscriminator


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


def test_multi_layer_disc():
    input_g = torch.randn(1, 3, 256, 256)
    # test multi-layer discriminators without fc layer
    multi_disc = MultiLayerDiscriminator(
        in_channels=3, max_channels=256, fc_in_channels=0)
    multi_disc.init_weights()
    disc_pred = multi_disc(input_g)
    assert disc_pred.shape == (1, 256, 8, 8)

    with pytest.raises(TypeError):
        multi_disc.init_weights(pretrained=dict(igccc=4396))

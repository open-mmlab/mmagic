import pytest
import torch

from mmedit.models import build_component
from mmedit.models.extractors import HourGlass
from mmedit.models.extractors.feedback_hour_glass import ResBlock


def test_lte():
    model_cfg = dict(
        type='LTE', requires_grad=False, pixel_range=1., pretrained=None)

    lte = build_component(model_cfg)
    assert lte.__class__.__name__ == 'LTE'

    x = torch.rand(2, 3, 64, 64)

    x_level1, x_level2, x_level3 = lte(x)
    assert x_level1.shape == (2, 64, 64, 64)
    assert x_level2.shape == (2, 128, 32, 32)
    assert x_level3.shape == (2, 256, 16, 16)

    lte.init_weights(None)
    with pytest.raises(IOError):
        model_cfg['pretrained'] = ''
        lte = build_component(model_cfg)
        x_level1, x_level2, x_level3 = lte(x)
        lte.init_weights('')
    with pytest.raises(TypeError):
        lte.init_weights(1)


def test_res_block():

    res_block = ResBlock(16, 32)
    x = torch.rand(2, 16, 64, 64)
    y = res_block(x)
    assert y.shape == (2, 32, 64, 64)

    res_block = ResBlock(16, 16)
    x = torch.rand(2, 16, 64, 64)
    y = res_block(x)
    assert y.shape == (2, 16, 64, 64)


def test_hour_glass():
    hour_glass = HourGlass(2, 16)
    x = torch.rand(2, 16, 64, 64)
    y = hour_glass(x)
    assert y.shape == x.shape

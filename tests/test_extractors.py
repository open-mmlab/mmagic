import pytest
import torch

from mmedit.models import build_component


def test_lte():
    model_cfg = dict(
        type='LTE', requires_grad=False, pixel_range=1., pretrained=None)

    lte = build_component(model_cfg)
    assert lte.__class__.__name__ == 'LTE'

    x = torch.rand(2, 3, 64, 64)
    x_lv1, x_lv2, x_lv3 = lte(x)

    assert x_lv1.shape == (2, 64, 64, 64)
    assert x_lv2.shape == (2, 128, 32, 32)
    assert x_lv3.shape == (2, 256, 16, 16)
    lte.init_weights(None)
    with pytest.raises(IOError):
        lte.init_weights('')
    with pytest.raises(TypeError):
        lte.init_weights(1)


if __name__ == '__main__':
    test_lte()

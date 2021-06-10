import pytest
import torch

from mmedit.models import build_component
from mmedit.models.extractors import Hourglass
from mmedit.models.extractors.feedback_hour_glass import (
    ResBlock, reduce_to_five_heatmaps)


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
    hour_glass = Hourglass(2, 16)
    x = torch.rand(2, 16, 64, 64)
    y = hour_glass(x)
    assert y.shape == x.shape


def test_feedback_hour_glass():
    model_cfg = dict(
        type='FeedbackHourglass', mid_channels=16, num_keypoints=20)

    fhg = build_component(model_cfg)
    assert fhg.__class__.__name__ == 'FeedbackHourglass'

    x = torch.rand(2, 3, 64, 64)
    heatmap, last_hidden = fhg.forward(x)
    assert heatmap.shape == (2, 20, 16, 16)
    assert last_hidden.shape == (2, 16, 16, 16)
    heatmap, last_hidden = fhg.forward(x, last_hidden)
    assert heatmap.shape == (2, 20, 16, 16)
    assert last_hidden.shape == (2, 16, 16, 16)


def test_reduce_to_five_heatmaps():
    heatmap = torch.rand((2, 5, 64, 64))
    new_heatmap = reduce_to_five_heatmaps(heatmap, False)
    assert new_heatmap.shape == (2, 5, 64, 64)
    new_heatmap = reduce_to_five_heatmaps(heatmap, True)
    assert new_heatmap.shape == (2, 5, 64, 64)

    heatmap = torch.rand((2, 68, 64, 64))
    new_heatmap = reduce_to_five_heatmaps(heatmap, False)
    assert new_heatmap.shape == (2, 5, 64, 64)
    new_heatmap = reduce_to_five_heatmaps(heatmap, True)
    assert new_heatmap.shape == (2, 5, 64, 64)

    heatmap = torch.rand((2, 194, 64, 64))
    new_heatmap = reduce_to_five_heatmaps(heatmap, False)
    assert new_heatmap.shape == (2, 5, 64, 64)
    new_heatmap = reduce_to_five_heatmaps(heatmap, True)
    assert new_heatmap.shape == (2, 5, 64, 64)

    with pytest.raises(NotImplementedError):
        heatmap = torch.rand((2, 12, 64, 64))
        reduce_to_five_heatmaps(heatmap, False)

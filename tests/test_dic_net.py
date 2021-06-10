import torch

from mmedit.models.backbones.sr_backbones.dic_net import (
    FeedbackBlock, FeedbackBlockCustom, FeedbackBlockHeatmapAttention)


def test_feedback_block():
    x1 = torch.rand(2, 16, 32, 32)

    model = FeedbackBlock(16, 3, 8)
    x2 = model(x1)
    assert x2.shape == x1.shape
    x3 = model(x2)
    assert x3.shape == x2.shape


def test_feedback_block_custom():
    x1 = torch.rand(2, 3, 32, 32)

    model = FeedbackBlockCustom(3, 16, 3, 8)
    x2 = model(x1)
    assert x2.shape == (2, 16, 32, 32)


def test_feedback_block_heatmap_attention():
    x1 = torch.rand(2, 16, 32, 32)
    heatmap = torch.rand(2, 5, 32, 32)

    model = FeedbackBlockHeatmapAttention(16, 2, 8, 5, 2)
    x2 = model(x1, heatmap)
    assert x2.shape == x1.shape
    x3 = model(x2, heatmap)
    assert x3.shape == x2.shape

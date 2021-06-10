import torch

from mmedit.models.backbones.sr_backbones.dic_net import (FeedbackBlock,
                                                          FeedbackBlockCustom)


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

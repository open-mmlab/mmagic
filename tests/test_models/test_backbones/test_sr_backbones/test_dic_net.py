# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmedit.models import build_backbone
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


def test_dic_net():

    model_cfg = dict(
        type='DICNet',
        in_channels=3,
        out_channels=3,
        mid_channels=48,
        num_blocks=6,
        hg_mid_channels=256,
        hg_num_keypoints=68,
        num_steps=4,
        upscale_factor=8,
        detach_attention=False)

    # build model
    model = build_backbone(model_cfg)

    # test attributes
    assert model.__class__.__name__ == 'DICNet'

    # prepare data
    inputs = torch.rand(1, 3, 16, 16)
    targets = torch.rand(1, 3, 128, 128)

    # prepare loss
    loss_function = nn.L1Loss()

    # prepare optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # test on cpu
    output, _ = model(inputs)
    optimizer.zero_grad()
    loss = loss_function(output[-1], targets)
    loss.backward()
    optimizer.step()
    assert len(output) == 4
    assert torch.is_tensor(output[-1])
    assert output[-1].shape == targets.shape

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters())
        inputs = inputs.cuda()
        targets = targets.cuda()
        output, _ = model(inputs)
        optimizer.zero_grad()
        loss = loss_function(output[-1], targets)
        loss.backward()
        optimizer.step()
        assert len(output) == 4
        assert torch.is_tensor(output[-1])
        assert output[-1].shape == targets.shape

    with pytest.raises(OSError):
        model.init_weights('')
    with pytest.raises(TypeError):
        model.init_weights(1)

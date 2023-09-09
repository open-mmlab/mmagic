# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.models.editors.ddpm.attention import (ApproximateGELU,
                                                  CrossAttention, FeedForward,
                                                  Transformer2DModel)


def test_ApproximateGELU():
    input = torch.rand((16, 16))
    gelu = ApproximateGELU(16, 24)
    output = gelu.forward(input)
    assert output.shape == (16, 24)


def test_crossattention():
    input = torch.rand((2, 64, 64))
    crossattention = CrossAttention(64)
    crossattention._slice_size = 2
    output = crossattention.forward(input)
    assert output.shape == (2, 64, 64)


def test_Transformer2DModel_init():
    with pytest.raises(Exception):
        Transformer2DModel(in_channels=32, num_vector_embeds=4)

    with pytest.raises(Exception):
        Transformer2DModel()

    Transformer2DModel(in_channels=32, use_linear_projection=True)


def test_FeedForward():
    input = torch.rand((2, 64, 64))
    feed_forward = FeedForward(64, 64, activation_fn='geglu-approximate')
    output = feed_forward.forward(input)
    assert output.shape == (2, 64, 64)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.models.editors.animatediff.unet_block import (
    CrossAttnDownBlock3D, CrossAttnUpBlock3D, UNetMidBlock3DCrossAttn,
    get_down_block, get_up_block)


def test_UNetMidBlock3DCrossAttn():
    input = torch.rand((1, 64, 16, 64, 64))
    midblock = UNetMidBlock3DCrossAttn(
        64,
        64,
        cross_attention_dim=64,
        unet_use_cross_frame_attention=False,
        unet_use_temporal_attention=False)
    output = midblock.forward(input)
    assert output.shape == (1, 64, 16, 64, 64)


def test_CrossAttnDownBlock3D():
    input = torch.rand((1, 64, 16, 64, 64))
    downblock = CrossAttnDownBlock3D(
        64,
        64,
        64,
        cross_attention_dim=64,
        unet_use_cross_frame_attention=False,
        unet_use_temporal_attention=False)
    output, _ = downblock.forward(input)
    assert output.shape == (1, 64, 16, 32, 32)


def test_CrossAttnUpBlock3D():
    input = torch.rand((1, 64, 16, 64, 64))
    upblock = CrossAttnUpBlock3D(
        64,
        64,
        64,
        64,
        cross_attention_dim=64,
        unet_use_cross_frame_attention=False,
        unet_use_temporal_attention=False)
    output = upblock.forward(input, [input])
    assert output.shape == (1, 64, 16, 128, 128)


def test_get_down_block():
    with pytest.raises(Exception):
        get_down_block('tem', 1, 1, 1, 1, True, 'silu', 1)


def test_get_up_block():
    with pytest.raises(Exception):
        get_up_block('tem', 1, 1, 1, 1, 1, True, 'silu', 1)


if __name__ == '__main__':
    test_CrossAttnDownBlock3D()
    test_CrossAttnUpBlock3D()
    test_get_down_block()
    test_get_up_block()

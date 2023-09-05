# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.models.editors.ddpm.unet_blocks import (CrossAttnDownBlock2D,
                                                    CrossAttnUpBlock2D,
                                                    UNetMidBlock2DCrossAttn,
                                                    get_down_block,
                                                    get_up_block)


def test_UNetMidBlock2DCrossAttn():
    input = torch.rand((1, 64, 64, 64))
    midblock = UNetMidBlock2DCrossAttn(64, 64, cross_attention_dim=64)
    midblock.set_attention_slice(1)
    output = midblock.forward(input)
    assert output.shape == (1, 64, 64, 64)

    with pytest.raises(Exception):
        midblock.set_attention_slice(8)


def test_CrossAttnDownBlock2D():
    input = torch.rand((1, 64, 64, 64))
    downblock = CrossAttnDownBlock2D(64, 64, 64, cross_attention_dim=64)
    downblock.set_attention_slice(1)
    output, _ = downblock.forward(input)
    assert output.shape == (1, 64, 32, 32)

    with pytest.raises(Exception):
        downblock.set_attention_slice(8)


def test_CrossAttnUpBlock2D():
    downblock = CrossAttnUpBlock2D(64, 64, 64, 64, cross_attention_dim=64)
    downblock.set_attention_slice(1)


def test_get_down_block():
    with pytest.raises(Exception):
        get_down_block('tem', 1, 1, 1, 1, True, 'silu', 1)


def get_get_up_block():
    with pytest.raises(Exception):
        get_up_block('tem', 1, 1, 1, 1, 1, True, 'silu', 1)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

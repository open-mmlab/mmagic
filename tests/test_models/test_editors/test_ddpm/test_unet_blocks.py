# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors.ddpm.unet_blocks import (CrossAttnDownBlock2D,
                                                    CrossAttnUpBlock2D,
                                                    UNetMidBlock2DCrossAttn)


def test_UNetMidBlock2DCrossAttn():
    input = torch.rand((1, 64, 64, 64))
    midblock = UNetMidBlock2DCrossAttn(64, 64, cross_attention_dim=64)
    midblock.set_attention_slice(1)
    output = midblock.forward(input)
    assert output.shape == (1, 64, 64, 64)


def test_CrossAttnDownBlock2D():
    input = torch.rand((1, 64, 64, 64))
    downblock = CrossAttnDownBlock2D(64, 64, 64, cross_attention_dim=64)
    downblock.set_attention_slice(1)
    output, _ = downblock.forward(input)
    assert output.shape == (1, 64, 32, 32)


def test_CrossAttnUpBlock2D():
    downblock = CrossAttnUpBlock2D(64, 64, 64, 64, cross_attention_dim=64)
    downblock.set_attention_slice(1)


if __name__ == '__main__':
    test_UNetMidBlock2DCrossAttn()

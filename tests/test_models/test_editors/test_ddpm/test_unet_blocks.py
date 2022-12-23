# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors.ddpm.unet_blocks import UNetMidBlock2DCrossAttn


def test_UNetMidBlock2DCrossAttn():
    input = torch.rand((1, 64, 64, 64))
    midblock = UNetMidBlock2DCrossAttn(64, 64, cross_attention_dim=64)
    midblock.set_attention_slice(1)
    output = midblock.forward(input)
    assert output.shape == (1, 64, 64, 64)


if __name__ == '__main__':
    test_UNetMidBlock2DCrossAttn()

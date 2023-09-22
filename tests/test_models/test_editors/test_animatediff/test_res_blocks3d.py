# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmagic.models.editors.animatediff.unet_block import (Downsample3D,
                                                          ResnetBlock3D,
                                                          Upsample3D)


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
def test_resnetblock3d():
    input = torch.rand((1, 64, 16, 16, 16))
    resblock = ResnetBlock3D(in_channels=64)
    output = resblock.forward(input, None)
    assert output.shape == (1, 64, 16, 16, 16)


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
def test_Downsample3D():
    input = torch.rand((1, 64, 16, 16, 16))
    downsample = Downsample3D(channels=64, use_conv=True, padding=1)
    output = downsample.forward(input)
    assert output.shape == (1, 64, 16, 8, 8)


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
def test_Upsample3D():
    input = torch.rand((1, 64, 16, 16, 16))
    upsample = Upsample3D(channels=64, use_conv_transpose=False, use_conv=True)

    output = upsample.forward(input)
    assert output.shape == (1, 64, 16, 32, 32)


# if __name__ == '__main__':
#     test_Downsample3D()
#     test_Upsample3D()
#     test_resnetblock3d()

# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.ddpm.res_blocks import (Downsample2D, ResnetBlock2D,
                                                   Upsample2D)


def test_resnetblock2d():
    input = torch.rand((1, 64, 16, 16))
    resblock = ResnetBlock2D(in_channels=64, up=True)
    output = resblock.forward(input, None)
    assert output.shape == (1, 64, 64, 64)

    resblock = ResnetBlock2D(in_channels=64, down=True)
    output = resblock.forward(input, None)
    assert output.shape == (1, 64, 8, 8)


def test_Downsample2D():
    input = torch.rand((1, 64, 16, 16))
    downsample = Downsample2D(channels=64, use_conv=True, padding=0)
    output = downsample.forward(input)
    assert output.shape == (1, 64, 8, 8)


def test_Upsample2D():
    input = torch.rand((1, 64, 16, 16))
    upsample = Upsample2D(channels=64, use_conv_transpose=True)
    output = upsample.forward(input)
    assert output.shape == (1, 64, 32, 32)

    upsample = Upsample2D(channels=64)
    output = upsample.forward(input, output_size=(32, 32))
    assert output.shape == (1, 64, 64, 64)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

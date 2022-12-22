# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors.stable_diffusion.vae import (AutoencoderKL,
                                                        ResnetBlock2D)


def test_vae():
    input = torch.rand((1, 3, 32, 32))
    vae = AutoencoderKL()
    output = vae.forward(input)
    assert output['sample'].shape == (1, 3, 32, 32)


def test_resnetblock2d():
    input = torch.rand((1, 64, 16, 16))
    resblock = ResnetBlock2D(in_channels=64, up=True)
    output = resblock.forward(input, None)
    assert output.shape == (1, 64, 64, 64)

    resblock = ResnetBlock2D(in_channels=64, down=True)
    output = resblock.forward(input, None)
    assert output.shape == (1, 64, 8, 8)


if __name__ == '__main__':
    test_resnetblock2d()

# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors.stable_diffusion.vae import (
    AutoencoderKL, DiagonalGaussianDistribution, ResnetBlock2D)


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


def test_DiagonalGaussianDistribution():
    param = torch.rand((1, 2, 16, 16))
    sample = torch.rand((1, 1, 16, 16))

    gauss_dist = DiagonalGaussianDistribution(param, deterministic=False)
    gauss_dist.sample()
    gauss_dist.kl()
    output = gauss_dist.nll(sample)
    assert output.shape == (1, )

    gauss_dist = DiagonalGaussianDistribution(param, deterministic=True)
    gauss_dist.sample()
    gauss_dist.kl()
    output = gauss_dist.nll(sample)
    assert output.shape == (1, )


if __name__ == '__main__':
    test_DiagonalGaussianDistribution()

# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.stable_diffusion.vae import (
    AttentionBlock, AutoencoderKL, DiagonalGaussianDistribution, Downsample2D,
    ResnetBlock2D, Upsample2D)


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


def test_AttentionBlock():
    input = torch.rand((1, 64, 32, 32))
    attention = AttentionBlock(64, num_head_channels=8)
    output = attention.forward(input)
    assert output.shape == (1, 64, 32, 32)


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

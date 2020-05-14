import torch
import torch.nn as nn
from mmedit.models.backbones import ResidualDilationBlock, TMADDilationNeck


def test_res_dilation():
    block = ResidualDilationBlock(3, 5)
    x = torch.rand((2, 3, 8, 8))
    res = block(x)

    assert res.size() == (2, 5, 8, 8)
    assert block.with_skip_conv
    assert not block.skip_conv.with_activation
    assert block.conv1.dilation == (2, 2)
    assert block.conv2.dilation == (1, 1)
    assert isinstance(block.conv1.activate, nn.LeakyReLU)
    assert isinstance(block.conv2.activate, nn.LeakyReLU)

    block = ResidualDilationBlock(3, 5, with_spectral_norm=True)
    x = torch.rand((2, 3, 8, 8))
    res = block(x)

    assert res.size() == (2, 5, 8, 8)
    assert block.with_skip_conv
    assert not block.skip_conv.with_activation
    assert block.conv1.dilation == (2, 2)
    assert block.conv2.dilation == (1, 1)
    assert isinstance(block.conv1.activate, nn.LeakyReLU)
    assert isinstance(block.conv2.activate, nn.LeakyReLU)
    assert block.conv1.with_spectral_norm
    assert block.conv2.with_spectral_norm
    assert block.skip_conv.with_spectral_norm

    block = ResidualDilationBlock(3, 3)
    x = torch.rand((2, 3, 8, 8))
    res = block(x)

    assert res.size() == (2, 3, 8, 8)
    assert not block.with_skip_conv
    assert block.conv1.dilation == (2, 2)
    assert block.conv2.dilation == (1, 1)
    assert isinstance(block.conv1.activate, nn.LeakyReLU)
    assert isinstance(block.conv2.activate, nn.LeakyReLU)


def test_tmad_dilation_neck():
    neck = TMADDilationNeck(3, 6, num_blocks=2, dilation=2)
    x = torch.rand((2, 3, 8, 8))
    res = neck(x)

    assert res['out'].size() == (2, 6, 8, 8)
    assert res['dilation0'].size() == (2, 6, 8, 8)
    assert isinstance(neck.dilation_blocks[0], ResidualDilationBlock)
    assert neck.dilation_blocks[0].conv1.out_channels == 6
    assert neck.dilation_blocks[1].conv1.in_channels == 6

    neck = TMADDilationNeck(3, 6, num_blocks=2, dilation=2)
    x = torch.rand((2, 3, 8, 8))
    res = neck(dict(out=x))

    assert res['out'].size() == (2, 6, 8, 8)
    assert isinstance(neck.dilation_blocks[0], ResidualDilationBlock)
    assert neck.dilation_blocks[0].conv1.out_channels == 6
    assert neck.dilation_blocks[1].conv1.in_channels == 6

    neck = TMADDilationNeck(
        3, 6, num_blocks=2, dilation=2, with_spectral_norm=True)
    x = torch.rand((2, 3, 8, 8))
    res = neck(x)

    assert res['out'].size() == (2, 6, 8, 8)
    assert isinstance(neck.dilation_blocks[0], ResidualDilationBlock)
    assert neck.dilation_blocks[0].conv1.out_channels == 6
    assert neck.dilation_blocks[1].conv1.in_channels == 6
    assert neck.dilation_blocks[0].conv1.with_spectral_norm
    assert neck.dilation_blocks[1].conv2.with_spectral_norm

    neck = TMADDilationNeck(3, 6, num_blocks=2, dilation=2, act_cfg=None)
    x = torch.rand((2, 3, 8, 8))
    res = neck(x)

    assert res['out'].size() == (2, 6, 8, 8)
    assert isinstance(neck.dilation_blocks[0], ResidualDilationBlock)
    assert neck.dilation_blocks[0].conv1.out_channels == 6
    assert neck.dilation_blocks[1].conv1.in_channels == 6
    assert not neck.dilation_blocks[0].conv1.with_activation
    assert not neck.dilation_blocks[1].conv2.with_activation

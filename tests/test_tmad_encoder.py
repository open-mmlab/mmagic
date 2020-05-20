import pytest
import torch
import torch.nn as nn
from mmedit.models.backbones import PGDownsampleBlock, TMADEncoder


def test_pgdownsample_block():
    x = torch.rand((2, 3, 8, 8))
    pgdown = PGDownsampleBlock(3, 5)
    res = pgdown(x)

    assert res.size() == (2, 5, 4, 4)
    assert pgdown.conv1.out_channels == 3
    assert pgdown.conv2.out_channels == 5
    assert not pgdown.conv1.with_norm
    assert pgdown.conv1.with_activation
    assert isinstance(pgdown.conv2.activate, nn.LeakyReLU)

    x = torch.rand((2, 3, 8, 8))
    pgdown = PGDownsampleBlock(3, 5, interpolation='bilinear')
    res = pgdown(x)

    assert res.size() == (2, 5, 4, 4)
    assert pgdown.conv1.out_channels == 3
    assert pgdown.conv2.out_channels == 5
    assert not pgdown.conv1.with_norm
    assert pgdown.conv1.with_activation
    assert isinstance(pgdown.conv2.activate, nn.LeakyReLU)

    x = torch.rand((2, 3, 8, 8))
    pgdown = PGDownsampleBlock(3, 5, scale_factor=2, interpolation='avgpool2d')
    res = pgdown(x)

    assert res.size() == (2, 5, 4, 4)
    assert pgdown.conv1.out_channels == 3
    assert pgdown.conv2.out_channels == 5
    assert not pgdown.conv1.with_norm
    assert pgdown.conv1.with_activation
    assert isinstance(pgdown.conv2.activate, nn.LeakyReLU)

    x = torch.rand((2, 3, 8, 8))
    pgdown = PGDownsampleBlock(3, 5, act_cfg=dict(type='ReLU'))
    res = pgdown(x)

    assert res.size() == (2, 5, 4, 4)
    assert pgdown.conv1.out_channels == 3
    assert pgdown.conv2.out_channels == 5
    assert not pgdown.conv1.with_norm
    assert pgdown.conv1.with_activation
    assert isinstance(pgdown.conv2.activate, nn.ReLU)

    x = torch.rand((2, 3, 8, 8))
    pgdown = PGDownsampleBlock(
        3, 5, act_cfg=dict(type='ReLU'), with_spectral_norm=True)
    res = pgdown(x)

    assert res.size() == (2, 5, 4, 4)
    assert pgdown.conv1.out_channels == 3
    assert pgdown.conv2.out_channels == 5
    assert not pgdown.conv1.with_norm
    assert pgdown.conv1.with_activation
    assert isinstance(pgdown.conv2.activate, nn.ReLU)
    assert hasattr(pgdown.conv1.conv, 'weight_orig')
    assert hasattr(pgdown.conv2.conv, 'weight_orig')

    x = torch.rand((2, 3, 8, 8))
    pgdown = PGDownsampleBlock(3, 5, interpolation=None)
    res = pgdown(x)

    assert res.size() == (2, 5, 8, 8)
    assert pgdown.conv1.out_channels == 3
    assert pgdown.conv2.out_channels == 5
    assert not pgdown.conv1.with_norm
    assert pgdown.conv1.with_activation
    assert isinstance(pgdown.conv2.activate, nn.LeakyReLU)

    with pytest.raises(NotImplementedError):
        pgdown = PGDownsampleBlock(3, 5, interpolation='igccc')


def test_tmad_encoder():
    encoder = TMADEncoder(3, num_blocks=2)
    x = torch.rand((2, 3, 16, 16))
    res = encoder(x)

    assert res['out'].size() == (2, 64, 4, 4)
    assert res['dsblock0'].size() == (2, 32, 8, 8)
    assert encoder.input_conv.with_activation
    assert isinstance(encoder.input_conv.activate, nn.LeakyReLU)
    assert isinstance(encoder.encoder_blocks[0], PGDownsampleBlock)

    encoder = TMADEncoder(3, num_blocks=2, with_spectral_norm=True)
    x = torch.rand((2, 3, 16, 16))
    res = encoder(x)

    assert res['out'].size() == (2, 64, 4, 4)
    assert encoder.input_conv.with_activation
    assert isinstance(encoder.input_conv.activate, nn.LeakyReLU)
    assert isinstance(encoder.encoder_blocks[0], PGDownsampleBlock)
    assert hasattr(encoder.encoder_blocks[0].conv1.conv, 'weight_orig')
    assert len(encoder.encoder_blocks) == 2
    assert isinstance(encoder.encoder_blocks[0].conv1.activate, nn.LeakyReLU)

    encoder = TMADEncoder(
        3, num_blocks=2, norm_cfg=dict(type='BN'), with_spectral_norm=True)
    x = torch.rand((2, 3, 16, 16))
    res = encoder(x)

    assert res['out'].size() == (2, 64, 4, 4)
    assert encoder.input_conv.with_activation
    assert isinstance(encoder.input_conv.norm, nn.BatchNorm2d)
    assert isinstance(encoder.input_conv.activate, nn.LeakyReLU)
    assert isinstance(encoder.encoder_blocks[0], PGDownsampleBlock)
    assert hasattr(encoder.encoder_blocks[0].conv1.conv, 'weight_orig')
    assert len(encoder.encoder_blocks) == 2
    assert isinstance(encoder.encoder_blocks[0].conv1.activate, nn.LeakyReLU)
    assert isinstance(encoder.encoder_blocks[0].conv1.norm, nn.BatchNorm2d)

import pytest
import torch
import torch.nn as nn
from mmedit.models.backbones import PGUpsampleBlock, TMADDecoder


def test_pgupsample_block():
    pgblock = PGUpsampleBlock(10, 20)
    x = torch.rand((2, 10, 16, 16))
    res = pgblock(x)

    assert res.size() == (2, 20, 32, 32)
    assert pgblock.conv1.out_channels == 20
    assert pgblock.interpolation == 'nearest'
    assert pgblock.conv1.with_activation
    assert isinstance(pgblock.conv1.activate, nn.LeakyReLU)
    assert pgblock.conv2.with_activation
    assert isinstance(pgblock.conv2.activate, nn.LeakyReLU)

    pgblock = PGUpsampleBlock(10, 20, act_cfg=None)
    x = torch.rand((2, 10, 16, 16))
    res = pgblock(x)

    assert res.size() == (2, 20, 32, 32)
    assert pgblock.conv1.out_channels == 20
    assert pgblock.interpolation == 'nearest'
    assert not pgblock.conv1.with_activation
    assert not pgblock.conv2.with_activation

    pgblock = PGUpsampleBlock(10, 20, interpolation='bilinear')
    x = torch.rand((2, 10, 16, 16))
    res = pgblock(x)

    assert res.size() == (2, 20, 32, 32)
    assert pgblock.conv1.out_channels == 20
    assert pgblock.interpolation == 'bilinear'
    assert pgblock.conv1.with_activation
    assert isinstance(pgblock.conv1.activate, nn.LeakyReLU)
    assert pgblock.conv2.with_activation
    assert isinstance(pgblock.conv2.activate, nn.LeakyReLU)

    pgblock = PGUpsampleBlock(10, 20, interpolation=None)
    x = torch.rand((2, 10, 16, 16))
    res = pgblock(x)

    assert res.size() == (2, 20, 16, 16)
    assert pgblock.conv1.out_channels == 20
    assert pgblock.interpolation is None
    assert pgblock.conv1.with_activation
    assert isinstance(pgblock.conv1.activate, nn.LeakyReLU)
    assert pgblock.conv2.with_activation
    assert isinstance(pgblock.conv2.activate, nn.LeakyReLU)

    with pytest.raises(NotImplementedError):
        pgblock = PGUpsampleBlock(10, 20, interpolation='igccc')

    pgblock = PGUpsampleBlock(10, 20, with_spectral_norm=True)
    x = torch.rand((2, 10, 16, 16))
    res = pgblock(x)

    assert res.size() == (2, 20, 32, 32)
    assert pgblock.conv1.out_channels == 20
    assert pgblock.interpolation == 'nearest'
    assert pgblock.conv1.with_activation
    assert isinstance(pgblock.conv1.activate, nn.LeakyReLU)
    assert pgblock.conv2.with_activation
    assert isinstance(pgblock.conv2.activate, nn.LeakyReLU)
    assert pgblock.conv1.with_spectral_norm

    pgblock = PGUpsampleBlock(10, 20, padding_mode='reflect')
    x = torch.rand((2, 10, 16, 16))
    res = pgblock(x)

    assert res.size() == (2, 20, 32, 32)
    assert pgblock.conv1.out_channels == 20
    assert pgblock.interpolation == 'nearest'
    assert pgblock.conv1.with_activation
    assert isinstance(pgblock.conv1.activate, nn.LeakyReLU)
    assert pgblock.conv2.with_activation
    assert isinstance(pgblock.conv2.activate, nn.LeakyReLU)
    assert pgblock.conv1.with_explicit_padding
    assert pgblock.conv2.with_explicit_padding

    pgblock = PGUpsampleBlock(10, 20, norm_cfg=dict(type='BN'))
    x = torch.rand((2, 10, 16, 16))
    res = pgblock(x)

    assert res.size() == (2, 20, 32, 32)
    assert pgblock.conv1.out_channels == 20
    assert pgblock.interpolation == 'nearest'
    assert pgblock.conv1.with_activation
    assert isinstance(pgblock.conv1.activate, nn.LeakyReLU)
    assert pgblock.conv2.with_activation
    assert isinstance(pgblock.conv2.activate, nn.LeakyReLU)
    assert pgblock.conv1.with_norm
    assert pgblock.conv2.with_norm


def test_tmad_dec():
    dec = TMADDecoder(10, 3, num_blocks=2)
    x = torch.rand((2, 10, 16, 16))

    res = dec(x)
    assert res.size() == (2, 3, 64, 64)
    assert dec.with_out_activation
    assert len(dec.decoder_blocks) == 2
    assert isinstance(dec.decoder_blocks[0], PGUpsampleBlock)
    assert dec.output_conv1.with_activation
    assert not dec.output_conv2.with_activation
    assert not dec.output_conv1.with_norm
    assert not dec.output_conv2.with_norm

    dec = TMADDecoder(10, 3, num_blocks=2, norm_cfg=dict(type='BN'))
    x = torch.rand((2, 10, 16, 16))

    res = dec(x)
    assert res.size() == (2, 3, 64, 64)
    assert dec.with_out_activation
    assert len(dec.decoder_blocks) == 2
    assert isinstance(dec.decoder_blocks[0], PGUpsampleBlock)
    assert dec.output_conv1.with_activation
    assert not dec.output_conv2.with_activation
    assert dec.output_conv1.with_norm
    assert dec.output_conv2.with_norm

    dec = TMADDecoder(10, 3, num_blocks=2, act_cfg=dict(type='ReLU'))
    x = torch.rand((2, 10, 16, 16))

    res = dec(x)
    assert res.size() == (2, 3, 64, 64)
    assert dec.with_out_activation
    assert len(dec.decoder_blocks) == 2
    assert isinstance(dec.decoder_blocks[0], PGUpsampleBlock)
    assert isinstance(dec.decoder_blocks[0].conv1.activate, nn.ReLU)
    assert dec.output_conv1.with_activation
    assert isinstance(dec.output_conv1.activate, nn.ReLU)
    assert not dec.output_conv2.with_activation
    assert not dec.output_conv1.with_norm
    assert not dec.output_conv2.with_norm

    dec = TMADDecoder(10, 3, num_blocks=2, out_act_cfg=dict(type='Sigmoid'))
    x = torch.rand((2, 10, 16, 16))

    res = dec(x)
    assert res.size() == (2, 3, 64, 64)
    assert dec.with_out_activation
    assert len(dec.decoder_blocks) == 2
    assert isinstance(dec.decoder_blocks[0], PGUpsampleBlock)
    assert dec.output_conv1.with_activation
    assert not dec.output_conv2.with_activation
    assert not dec.output_conv1.with_norm
    assert not dec.output_conv2.with_norm
    assert isinstance(dec.out_act, nn.Sigmoid)

    dec = TMADDecoder(10, 3, num_blocks=2, with_spectral_norm=True)
    x = torch.rand((2, 10, 16, 16))

    res = dec(x)
    assert res.size() == (2, 3, 64, 64)
    assert dec.with_out_activation
    assert len(dec.decoder_blocks) == 2
    assert isinstance(dec.decoder_blocks[0], PGUpsampleBlock)
    assert dec.output_conv1.with_activation
    assert not dec.output_conv2.with_activation
    assert not dec.output_conv1.with_norm
    assert not dec.output_conv2.with_norm
    assert dec.decoder_blocks[0].conv1.with_spectral_norm
    assert dec.output_conv1.with_spectral_norm
    assert dec.output_conv2.with_spectral_norm

    dec = TMADDecoder(10, 3, num_blocks=2, out_act_cfg=None)
    x = torch.rand((2, 10, 16, 16))

    res = dec(x)
    assert res.size() == (2, 3, 64, 64)
    assert not dec.with_out_activation
    assert len(dec.decoder_blocks) == 2
    assert isinstance(dec.decoder_blocks[0], PGUpsampleBlock)
    assert dec.output_conv1.with_activation
    assert not dec.output_conv2.with_activation
    assert not dec.output_conv1.with_norm
    assert not dec.output_conv2.with_norm

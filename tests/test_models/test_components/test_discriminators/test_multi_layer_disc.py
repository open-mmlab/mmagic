import pytest
import torch
import torch.nn as nn

from mmedit.models.components import MultiLayerDiscriminator


def test_multi_layer_disc():
    with pytest.raises(AssertionError):
        # fc_in_channels must be greater than 0
        multi_disc = MultiLayerDiscriminator(
            3, 236, fc_in_channels=-100, out_act_cfg=None)

    with pytest.raises(TypeError):
        # stride_list must be a tuple of int with length of 1 or
        # length of num_conv
        multi_disc = MultiLayerDiscriminator(
            3, 256, num_convs=3, stride_list=(1, 2))

    input_g = torch.randn(1, 3, 256, 256)
    # test multi-layer discriminators without fc layer
    multi_disc = MultiLayerDiscriminator(
        in_channels=3, max_channels=256, fc_in_channels=None)
    multi_disc.init_weights()
    disc_pred = multi_disc(input_g)
    assert disc_pred.shape == (1, 256, 8, 8)
    multi_disc = MultiLayerDiscriminator(
        in_channels=3, max_channels=256, fc_in_channels=100)
    assert isinstance(multi_disc.fc.activate, nn.ReLU)

    multi_disc = MultiLayerDiscriminator(3, 236, fc_in_channels=None)
    assert multi_disc.with_out_act
    assert not multi_disc.with_fc
    assert isinstance(multi_disc.conv5.activate, nn.ReLU)

    multi_disc = MultiLayerDiscriminator(
        3, 236, fc_in_channels=None, out_act_cfg=None)
    assert not multi_disc.conv5.with_activation
    with pytest.raises(TypeError):
        multi_disc.init_weights(pretrained=dict(igccc=4396))

    input_g = torch.randn(1, 3, 16, 16)
    multi_disc = MultiLayerDiscriminator(
        in_channels=3,
        max_channels=256,
        num_convs=2,
        fc_in_channels=4 * 4 * 128,
        fc_out_channels=10,
        with_spectral_norm=True)
    multi_disc.init_weights()
    disc_pred = multi_disc(input_g)
    assert disc_pred.shape == (1, 10)
    assert multi_disc.conv1.with_spectral_norm
    assert multi_disc.conv2.with_spectral_norm
    assert hasattr(multi_disc.fc.linear, 'weight_orig')

    num_convs = 3
    multi_disc = MultiLayerDiscriminator(
        in_channels=64,
        max_channels=512,
        num_convs=num_convs,
        kernel_size=4,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        out_act_cfg=dict(type='ReLU'),
        with_input_norm=False,
        with_out_convs=True)
    # check input conv
    assert not multi_disc.conv1.with_norm
    assert isinstance(multi_disc.conv1.activate, nn.LeakyReLU)
    assert multi_disc.conv1.stride == (2, 2)

    # check intermediate conv
    for i in range(1, num_convs):
        assert getattr(multi_disc, f'conv{i + 1}').with_norm
        assert isinstance(
            getattr(multi_disc, f'conv{i + 1}').activate, nn.LeakyReLU)
        assert getattr(multi_disc, f'conv{i + 1}').stride == (2, 2)

    # check out_conv
    assert multi_disc.conv4.with_norm
    assert multi_disc.conv4.with_activation
    assert multi_disc.conv4.stride == (1, 1)
    assert not multi_disc.conv5.with_norm
    assert not multi_disc.conv5.with_activation
    assert multi_disc.conv5.stride == (1, 1)

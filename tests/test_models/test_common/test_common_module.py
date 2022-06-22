# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmedit.models.common import (LinearModule, MaskConvModule, PartialConv2d,
                                  SimpleGatedConvModule)


def test_mask_conv_module():
    with pytest.raises(KeyError):
        # conv_cfg must be a dict or None
        conv_cfg = dict(type='conv')
        MaskConvModule(3, 8, 2, conv_cfg=conv_cfg)

    with pytest.raises(AssertionError):
        # norm_cfg must be a dict or None
        norm_cfg = ['norm']
        MaskConvModule(3, 8, 2, norm_cfg=norm_cfg)

    with pytest.raises(AssertionError):
        # order elements must be ('conv', 'norm', 'act')
        order = ['conv', 'norm', 'act']
        MaskConvModule(3, 8, 2, order=order)

    with pytest.raises(AssertionError):
        # order elements must be ('conv', 'norm', 'act')
        order = ('conv', 'norm')
        MaskConvModule(3, 8, 2, order=order)

    with pytest.raises(KeyError):
        # softmax is not supported
        act_cfg = dict(type='softmax')
        MaskConvModule(3, 8, 2, act_cfg=act_cfg)

    conv_cfg = dict(type='PConv', multi_channel=True)
    conv = MaskConvModule(3, 8, 2, conv_cfg=conv_cfg)
    x = torch.rand(1, 3, 256, 256)
    mask_in = torch.ones_like(x)
    mask_in[..., 20:130, 120:150] = 0.
    output, mask_update = conv(x, mask_in)
    assert output.shape == (1, 8, 255, 255)
    assert mask_update.shape == (1, 8, 255, 255)

    # add test for ['norm', 'conv', 'act']
    conv = MaskConvModule(
        3, 8, 2, order=('norm', 'conv', 'act'), conv_cfg=conv_cfg)
    x = torch.rand(1, 3, 256, 256)
    output = conv(x, mask_in, return_mask=False)
    assert output.shape == (1, 8, 255, 255)

    conv = MaskConvModule(
        3, 8, 3, padding=1, conv_cfg=conv_cfg, with_spectral_norm=True)
    assert hasattr(conv.conv, 'weight_orig')
    output = conv(x, return_mask=False)
    assert output.shape == (1, 8, 256, 256)

    conv = MaskConvModule(
        3,
        8,
        3,
        padding=1,
        norm_cfg=dict(type='BN'),
        padding_mode='reflect',
        conv_cfg=conv_cfg)
    assert isinstance(conv.padding_layer, nn.ReflectionPad2d)
    output = conv(x, mask_in, return_mask=False)
    assert output.shape == (1, 8, 256, 256)

    conv = MaskConvModule(
        3, 8, 3, padding=1, act_cfg=dict(type='LeakyReLU'), conv_cfg=conv_cfg)
    output = conv(x, mask_in, return_mask=False)
    assert output.shape == (1, 8, 256, 256)

    with pytest.raises(KeyError):
        conv = MaskConvModule(3, 8, 3, padding=1, padding_mode='igccc')


def test_pconv2d():
    pconv2d = PartialConv2d(
        3, 2, kernel_size=1, stride=1, multi_channel=True, eps=1e-8)

    x = torch.rand(1, 3, 6, 6)
    mask = torch.ones_like(x)
    mask[..., 2, 2] = 0.
    output, updated_mask = pconv2d(x, mask=mask)
    assert output.shape == (1, 2, 6, 6)
    assert updated_mask.shape == (1, 2, 6, 6)

    output = pconv2d(x, mask=None)
    assert output.shape == (1, 2, 6, 6)

    pconv2d = PartialConv2d(
        3, 2, kernel_size=1, stride=1, multi_channel=True, eps=1e-8)
    output = pconv2d(x, mask=None)
    assert output.shape == (1, 2, 6, 6)

    pconv2d = PartialConv2d(
        3, 2, kernel_size=1, stride=1, multi_channel=False, eps=1e-8)
    output = pconv2d(x, mask=None)
    assert output.shape == (1, 2, 6, 6)

    pconv2d = PartialConv2d(
        3,
        2,
        kernel_size=1,
        stride=1,
        bias=False,
        multi_channel=True,
        eps=1e-8)
    output = pconv2d(x, mask=mask, return_mask=False)
    assert output.shape == (1, 2, 6, 6)

    with pytest.raises(AssertionError):
        pconv2d(x, mask=torch.ones(1, 1, 6, 6))

    pconv2d = PartialConv2d(
        3,
        2,
        kernel_size=1,
        stride=1,
        bias=False,
        multi_channel=False,
        eps=1e-8)
    output = pconv2d(x, mask=None)
    assert output.shape == (1, 2, 6, 6)

    with pytest.raises(AssertionError):
        output = pconv2d(x, mask=mask[0])

    with pytest.raises(AssertionError):
        output = pconv2d(x, mask=torch.ones(1, 3, 6, 6))

    if torch.cuda.is_available():
        pconv2d = PartialConv2d(
            3,
            2,
            kernel_size=1,
            stride=1,
            bias=False,
            multi_channel=False,
            eps=1e-8).cuda().half()
        output = pconv2d(x.cuda().half(), mask=None)
        assert output.shape == (1, 2, 6, 6)


def test_gated_conv():
    conv = SimpleGatedConvModule(3, 10, 3, padding=1)
    x = torch.rand((2, 3, 10, 10))
    assert not conv.conv.with_activation
    assert conv.with_feat_act
    assert conv.with_gate_act
    assert isinstance(conv.feat_act, nn.ELU)
    assert isinstance(conv.gate_act, nn.Sigmoid)
    assert conv.conv.out_channels == 20

    out = conv(x)
    assert out.shape == (2, 10, 10, 10)

    conv = SimpleGatedConvModule(
        3, 10, 3, padding=1, feat_act_cfg=None, gate_act_cfg=None)
    assert not conv.with_gate_act
    out = conv(x)
    assert out.shape == (2, 10, 10, 10)

    with pytest.raises(AssertionError):
        conv = SimpleGatedConvModule(
            3, 1, 3, padding=1, order=('linear', 'act', 'norm'))

    conv = SimpleGatedConvModule(3, out_channels=10, kernel_size=3, padding=1)
    assert conv.conv.out_channels == 20
    out = conv(x)
    assert out.shape == (2, 10, 10, 10)


def test_linear_module():
    linear = LinearModule(10, 20)
    linear.init_weights()
    x = torch.rand((3, 10))
    assert linear.with_bias
    assert not linear.with_spectral_norm
    assert linear.out_features == 20
    assert linear.in_features == 10
    assert isinstance(linear.activate, nn.ReLU)

    y = linear(x)
    assert y.shape == (3, 20)

    linear = LinearModule(10, 20, act_cfg=None, with_spectral_norm=True)

    assert hasattr(linear.linear, 'weight_orig')
    assert not linear.with_activation
    y = linear(x)
    assert y.shape == (3, 20)

    linear = LinearModule(
        10, 20, act_cfg=dict(type='LeakyReLU'), with_spectral_norm=True)
    y = linear(x)
    assert y.shape == (3, 20)
    assert isinstance(linear.activate, nn.LeakyReLU)

    linear = LinearModule(
        10, 20, bias=False, act_cfg=None, with_spectral_norm=True)
    y = linear(x)
    assert y.shape == (3, 20)
    assert not linear.with_bias

    linear = LinearModule(
        10,
        20,
        bias=False,
        act_cfg=None,
        with_spectral_norm=True,
        order=('act', 'linear'))

    assert linear.order == ('act', 'linear')

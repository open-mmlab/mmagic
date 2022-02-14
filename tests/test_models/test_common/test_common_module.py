# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmedit.models.common import (ASPP, DepthwiseSeparableConvModule,
                                  GCAModule, LinearModule, MaskConvModule,
                                  PartialConv2d, SimpleGatedConvModule)


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


def test_depthwise_separable_conv():
    with pytest.raises(AssertionError):
        # conv_cfg must be a dict or None
        DepthwiseSeparableConvModule(4, 8, 2, groups=2)

    # test default config
    conv = DepthwiseSeparableConvModule(3, 8, 2)
    assert conv.depthwise_conv.conv.groups == 3
    assert conv.pointwise_conv.conv.kernel_size == (1, 1)
    assert not conv.depthwise_conv.with_norm
    assert not conv.pointwise_conv.with_norm
    assert conv.depthwise_conv.activate.__class__.__name__ == 'ReLU'
    assert conv.pointwise_conv.activate.__class__.__name__ == 'ReLU'
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # test
    conv = DepthwiseSeparableConvModule(3, 8, 2, dw_norm_cfg=dict(type='BN'))
    assert conv.depthwise_conv.norm_name == 'bn'
    assert not conv.pointwise_conv.with_norm
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    conv = DepthwiseSeparableConvModule(3, 8, 2, pw_norm_cfg=dict(type='BN'))
    assert not conv.depthwise_conv.with_norm
    assert conv.pointwise_conv.norm_name == 'bn'
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # add test for ['norm', 'conv', 'act']
    conv = DepthwiseSeparableConvModule(3, 8, 2, order=('norm', 'conv', 'act'))
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    conv = DepthwiseSeparableConvModule(
        3, 8, 3, padding=1, with_spectral_norm=True)
    assert hasattr(conv.depthwise_conv.conv, 'weight_orig')
    assert hasattr(conv.pointwise_conv.conv, 'weight_orig')
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    conv = DepthwiseSeparableConvModule(
        3, 8, 3, padding=1, padding_mode='reflect')
    assert isinstance(conv.depthwise_conv.padding_layer, nn.ReflectionPad2d)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    conv = DepthwiseSeparableConvModule(
        3, 8, 3, padding=1, dw_act_cfg=dict(type='LeakyReLU'))
    assert conv.depthwise_conv.activate.__class__.__name__ == 'LeakyReLU'
    assert conv.pointwise_conv.activate.__class__.__name__ == 'ReLU'
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    conv = DepthwiseSeparableConvModule(
        3, 8, 3, padding=1, pw_act_cfg=dict(type='LeakyReLU'))
    assert conv.depthwise_conv.activate.__class__.__name__ == 'ReLU'
    assert conv.pointwise_conv.activate.__class__.__name__ == 'LeakyReLU'
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)


def test_aspp():
    # test aspp with normal conv
    aspp = ASPP(128, out_channels=512, mid_channels=128, dilations=(6, 12, 18))
    assert aspp.convs[0].activate.__class__.__name__ == 'ReLU'
    assert aspp.convs[0].conv.out_channels == 128
    assert aspp.convs[1].__class__.__name__ == 'ConvModule'
    for conv_idx in range(1, 4):
        assert aspp.convs[conv_idx].conv.dilation[0] == 6 * conv_idx
    x = torch.rand(2, 128, 8, 8)
    output = aspp(x)
    assert output.shape == (2, 512, 8, 8)

    # test aspp with separable conv
    aspp = ASPP(128, separable_conv=True)
    assert aspp.convs[1].__class__.__name__ == 'DepthwiseSeparableConvModule'
    x = torch.rand(2, 128, 8, 8)
    output = aspp(x)
    assert output.shape == (2, 256, 8, 8)

    # test aspp with ReLU6
    aspp = ASPP(128, dilations=(12, 24, 36), act_cfg=dict(type='ReLU6'))
    assert aspp.convs[0].activate.__class__.__name__ == 'ReLU6'
    for conv_idx in range(1, 4):
        assert aspp.convs[conv_idx].conv.dilation[0] == 12 * conv_idx
    x = torch.rand(2, 128, 8, 8)
    output = aspp(x)
    assert output.shape == (2, 256, 8, 8)


def test_gca_module():
    img_feat = torch.rand(1, 128, 64, 64)
    alpha_feat = torch.rand(1, 128, 64, 64)
    unknown = None
    gca = GCAModule(128, 128, rate=1)
    output = gca(img_feat, alpha_feat, unknown)
    assert output.shape == (1, 128, 64, 64)

    img_feat = torch.rand(1, 128, 64, 64)
    alpha_feat = torch.rand(1, 128, 64, 64)
    unknown = torch.rand(1, 1, 64, 64)
    gca = GCAModule(128, 128, rate=2)
    output = gca(img_feat, alpha_feat, unknown)
    assert output.shape == (1, 128, 64, 64)


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

# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmedit.models.mattors.modules import (ASPP, DepthwiseSeparableConvModule,
                                           GCAModule)


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

import pytest
import torch
import torch.nn as nn
from mmedit.models.common import (ConvModule, PartialConv2d, build_conv_layer,
                                  build_norm_layer, build_padding_layer, norm)


def test_build_conv_layer():
    with pytest.raises(AssertionError):
        # `type` must be in cfg
        cfg = dict(kernel='3x3')
        build_conv_layer(cfg)

    with pytest.raises(AssertionError):
        # cfg must be a dict
        cfg = ['3x3']
        build_conv_layer(cfg)

    with pytest.raises(KeyError):
        cfg = dict(type='Norm')
        build_conv_layer(cfg)

    args = dict(in_channels=3, out_channels=8, kernel_size=2)
    layer = build_conv_layer(None, **args)
    assert type(layer) == nn.Conv2d
    assert layer.in_channels == args['in_channels']
    assert layer.out_channels == args['out_channels']
    assert layer.kernel_size == (2, 2)

    cfg = dict(type='Conv')
    layer = build_conv_layer(cfg, **args)
    assert type(layer) == nn.Conv2d
    assert layer.in_channels == args['in_channels']
    assert layer.out_channels == args['out_channels']
    assert layer.kernel_size == (2, 2)


def test_build_padding_layer():
    with pytest.raises(AssertionError):
        cfg = dict(tmd=None)
        build_padding_layer(cfg)

    with pytest.raises(KeyError):
        cfg = dict(type='tmd')
        build_padding_layer(cfg)

    input_x = torch.randn(1, 2, 5, 5)
    cfg = dict(type='reflect')
    padding_layer = build_padding_layer(cfg, 2)
    res = padding_layer(input_x)
    assert res.shape == (1, 2, 9, 9)


def test_conv_module():
    with pytest.raises(AssertionError):
        # conv_cfg must be a dict or None
        conv_cfg = ['conv']
        ConvModule(3, 8, 2, conv_cfg=conv_cfg)

    with pytest.raises(AssertionError):
        # norm_cfg must be a dict or None
        norm_cfg = ['norm']
        ConvModule(3, 8, 2, norm_cfg=norm_cfg)

    with pytest.raises(AssertionError):
        # order elements must be ('conv', 'norm', 'act')
        order = ['conv', 'norm', 'act']
        ConvModule(3, 8, 2, order=order)

    with pytest.raises(AssertionError):
        # order elements must be ('conv', 'norm', 'act')
        order = ('conv', 'norm')
        ConvModule(3, 8, 2, order=order)

    with pytest.raises(KeyError):
        # softmax is not supported
        act_cfg = dict(type='softmax')
        ConvModule(3, 8, 2, act_cfg=act_cfg)

    conv = ConvModule(3, 8, 2)
    conv.init_weights()
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    # add test for ['norm', 'conv', 'act']
    conv = ConvModule(3, 8, 2, order=('norm', 'conv', 'act'))
    conv.init_weights()
    x = torch.rand(1, 3, 256, 256)
    output = conv(x)
    assert output.shape == (1, 8, 255, 255)

    conv = ConvModule(3, 8, 3, padding=1, with_spectral_norm=True)
    assert hasattr(conv.conv, 'weight_orig')
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    conv = ConvModule(3, 8, 3, padding=1, padding_mode='reflect')
    assert isinstance(conv.padding_layer, nn.ReflectionPad2d)
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    conv = ConvModule(3, 8, 3, padding=1, act_cfg=dict(type='LeakyReLU'))
    output = conv(x)
    assert output.shape == (1, 8, 256, 256)

    with pytest.raises(KeyError):
        conv = ConvModule(3, 8, 3, padding=1, padding_mode='igccc')


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


def test_norm_layer():
    with pytest.raises(AssertionError):
        # `type` must be in cfg
        cfg = dict()
        build_norm_layer(cfg, 3)

    with pytest.raises(AssertionError):
        # cfg must be a dict
        cfg = ['BN']
        build_norm_layer(cfg, 3)

    with pytest.raises(KeyError):
        # cfg type must be in ['BN', 'BN3d', 'SyncBN', 'GN']
        cfg = dict(type='Conv')
        build_norm_layer(cfg, 3)

    norm.norm_cfg['testBN'] = ('testBN', None)
    with pytest.raises(NotImplementedError):
        cfg = dict(type='testBN')
        build_norm_layer(cfg, 3)

    with pytest.raises(AssertionError):
        # profix must be int or str
        cfg = dict(type='BN')
        build_norm_layer(cfg, 3, [1, 2])

    with pytest.raises(AssertionError):
        # 'num_groups' must be in cfg when using 'GN'
        cfg = dict(type='GN')
        build_norm_layer(cfg, 3)

    cfg = dict(type='BN')
    name, layer = build_norm_layer(cfg, 3, postfix=1)
    assert type(layer) == nn.BatchNorm2d
    assert name == 'bn1'
    assert layer.num_features == 3

    cfg = dict(type='BN3d')
    name, layer = build_norm_layer(cfg, 3, postfix='2')
    assert type(layer) == nn.BatchNorm3d
    assert name == 'bn2'
    assert layer.num_features == 3

    cfg = dict(type='SyncBN')
    name, layer = build_norm_layer(cfg, 3, postfix=3)
    assert type(layer) == nn.SyncBatchNorm
    assert name == 'bn3'
    assert layer.num_features == 3

    cfg = dict(type='GN', num_groups=3)
    name, layer = build_norm_layer(cfg, 3, postfix='4')
    assert type(layer) == nn.GroupNorm
    assert layer.num_channels == 3
    assert name == 'gn4'
    assert layer.num_groups == 3

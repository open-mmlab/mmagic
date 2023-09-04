# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmagic.models.editors import MaskConvModule
from mmagic.utils import register_all_modules

register_all_modules()


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


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

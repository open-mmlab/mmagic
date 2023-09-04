# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmagic.models.archs import LinearModule


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


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

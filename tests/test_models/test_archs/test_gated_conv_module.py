# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmagic.models.archs import SimpleGatedConvModule


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


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

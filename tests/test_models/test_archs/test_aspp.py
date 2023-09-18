# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.archs import ASPP


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


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

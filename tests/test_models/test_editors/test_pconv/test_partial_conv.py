# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.models.editors import PartialConv2d


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


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

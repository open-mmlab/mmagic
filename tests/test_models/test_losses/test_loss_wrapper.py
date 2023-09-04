# Copyright (c) OpenMMLab. All rights reserved.
import numpy.testing as npt
import pytest
import torch

from mmagic.models import mask_reduce_loss, reduce_loss


def test_utils():
    loss = torch.rand(1, 3, 4, 4)
    weight = torch.zeros(1, 3, 4, 4)
    weight[:, :, :2, :2] = 1

    # test reduce_loss()
    reduced = reduce_loss(loss, 'none')
    assert reduced is loss

    reduced = reduce_loss(loss, 'mean')
    npt.assert_almost_equal(reduced.numpy(), loss.mean())

    reduced = reduce_loss(loss, 'sum')
    npt.assert_almost_equal(reduced.numpy(), loss.sum())

    # test mask_reduce_loss()
    reduced = mask_reduce_loss(loss, weight=None, reduction='none')
    assert reduced is loss

    reduced = mask_reduce_loss(loss, weight=weight, reduction='mean')
    target = (loss *
              weight).sum(dim=[1, 2, 3]) / weight.sum(dim=[1, 2, 3]).mean()
    npt.assert_almost_equal(reduced.numpy(), target)

    reduced = mask_reduce_loss(loss, weight=weight, reduction='sum')
    npt.assert_almost_equal(reduced.numpy(), (loss * weight).sum())

    weight_single_channel = weight[:, 0:1, ...]
    reduced = mask_reduce_loss(
        loss, weight=weight_single_channel, reduction='mean')
    target = (loss *
              weight).sum(dim=[1, 2, 3]) / weight.sum(dim=[1, 2, 3]).mean()
    npt.assert_almost_equal(reduced.numpy(), target)

    loss_b = torch.rand(2, 3, 4, 4)
    weight_b = torch.zeros(2, 1, 4, 4)
    weight_b[0, :, :3, :3] = 1
    weight_b[1, :, :2, :2] = 1
    reduced = mask_reduce_loss(loss_b, weight=weight_b, reduction='mean')
    target = (loss_b * weight_b).sum() / weight_b.sum() / 3.
    npt.assert_almost_equal(reduced.numpy(), target)

    with pytest.raises(AssertionError):
        weight_wrong = weight[0, 0, ...]
        reduced = mask_reduce_loss(loss, weight=weight_wrong, reduction='mean')

    with pytest.raises(AssertionError):
        weight_wrong = weight[:, 0:2, ...]
        reduced = mask_reduce_loss(loss, weight=weight_wrong, reduction='mean')


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

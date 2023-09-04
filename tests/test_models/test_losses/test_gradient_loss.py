# Copyright (c) OpenMMLab. All rights reserved.
import numpy.testing as npt
import pytest
import torch

from mmagic.models import DiscShiftLoss, GradientLoss, GradientPenaltyLoss


def test_gradient_loss():
    with pytest.raises(ValueError):
        # only 'none', 'mean' and 'sum' are supported
        GradientLoss(reduction='InvalidValue')

    unknown_h, unknown_w = (32, 32)
    weight = torch.zeros(1, 1, 64, 64)
    weight[0, 0, :unknown_h, :unknown_w] = 1
    pred = weight.clone()
    target = weight.clone() * 2

    gradient_loss = GradientLoss(loss_weight=1.0, reduction='mean')
    loss = gradient_loss(pred, target)
    assert loss.shape == ()
    npt.assert_almost_equal(loss.item(), 0.1860352)

    gradient_loss = GradientLoss(loss_weight=0.5, reduction='none')
    loss = gradient_loss(pred, target, weight)
    assert loss.shape == (1, 1, 64, 64)
    npt.assert_almost_equal(torch.sum(loss).item(), 252)

    gradient_loss = GradientLoss(loss_weight=0.5, reduction='sum')
    loss = gradient_loss(pred, target, weight)
    assert loss.shape == ()
    npt.assert_almost_equal(loss.item(), 252)


def test_gradient_penalty_losses():
    """Test gradient penalty losses."""
    input = torch.ones(1, 3, 6, 6) * 2

    gan_loss = GradientPenaltyLoss(loss_weight=10.0)
    loss = gan_loss(lambda x: x, input, input, mask=None)
    assert loss.item() > 0
    mask = torch.ones(1, 3, 6, 6)
    mask[:, :, 2:4, 2:4] = 0
    loss = gan_loss(lambda x: x, input, input, mask=mask)
    assert loss.item() > 0


def test_disc_shift_loss():
    loss_disc_shift = DiscShiftLoss()
    x = torch.Tensor([0.1])
    loss = loss_disc_shift(x)

    npt.assert_almost_equal(loss.item(), 0.001)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

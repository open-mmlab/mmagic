# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy.testing as npt
import pytest
import torch

from mmagic.models import (CharbonnierLoss, L1Loss, MaskedTVLoss, MSELoss,
                           PSNRLoss)


def test_pixelwise_losses():
    with pytest.raises(ValueError):
        # only 'none', 'mean' and 'sum' are supported
        L1Loss(reduction='InvalidValue')

    with pytest.raises(ValueError):
        # only 'none', 'mean' and 'sum' are supported
        MSELoss(reduction='InvalidValue')

    with pytest.raises(ValueError):
        # only 'none', 'mean' and 'sum' are supported
        CharbonnierLoss(reduction='InvalidValue')

    unknown_h, unknown_w = (32, 32)
    weight = torch.zeros(1, 1, 64, 64)
    weight[0, 0, :unknown_h, :unknown_w] = 1
    pred = weight.clone()
    target = weight.clone() * 2

    # test l1 loss
    l1_loss = L1Loss(loss_weight=1.0, reduction='mean')
    loss = l1_loss(pred, target)
    assert loss.shape == ()
    assert loss.item() == 0.25

    l1_loss = L1Loss(loss_weight=0.5, reduction='none')
    loss = l1_loss(pred, target, weight)
    assert loss.shape == (1, 1, 64, 64)
    assert (loss == torch.ones(1, 1, 64, 64) * weight * 0.5).all()

    l1_loss = L1Loss(loss_weight=0.5, reduction='sum')
    loss = l1_loss(pred, target, weight)
    assert loss.shape == ()
    assert loss.item() == 512

    # test mse loss
    mse_loss = MSELoss(loss_weight=1.0, reduction='mean')
    loss = mse_loss(pred, target)
    assert loss.shape == ()
    assert loss.item() == 0.25

    mse_loss = MSELoss(loss_weight=0.5, reduction='none')
    loss = mse_loss(pred, target, weight)
    assert loss.shape == (1, 1, 64, 64)
    assert (loss == torch.ones(1, 1, 64, 64) * weight * 0.5).all()

    mse_loss = MSELoss(loss_weight=0.5, reduction='sum')
    loss = mse_loss(pred, target, weight)
    assert loss.shape == ()
    assert loss.item() == 512

    # test charbonnier loss
    charbonnier_loss = CharbonnierLoss(
        loss_weight=1.0, reduction='mean', eps=1e-12)
    loss = charbonnier_loss(pred, target)
    assert loss.shape == ()
    assert math.isclose(loss.item(), 0.25, rel_tol=1e-5)

    charbonnier_loss = CharbonnierLoss(
        loss_weight=0.5, reduction='none', eps=1e-6)
    loss = charbonnier_loss(pred, target, weight)
    assert loss.shape == (1, 1, 64, 64)
    npt.assert_almost_equal(
        loss.numpy(), torch.ones(1, 1, 64, 64) * weight * 0.5, decimal=6)

    charbonnier_loss = CharbonnierLoss(
        loss_weight=0.5, reduction='sum', eps=1e-12)
    loss = charbonnier_loss(pred, target)
    assert loss.shape == ()
    assert math.isclose(loss.item(), 512, rel_tol=1e-5)

    # test samplewise option, use L1Loss as an example
    unknown_h, unknown_w = (32, 32)
    weight = torch.zeros(2, 1, 64, 64)
    weight[0, 0, :unknown_h, :unknown_w] = 1
    # weight[1, 0, :unknown_h // 2, :unknown_w // 2] = 1
    pred = weight.clone()
    target = weight.clone()
    # make mean l1_loss of sample 2 different from sample 1
    target[0, ...] *= 2
    l1_loss = L1Loss(loss_weight=1.0, reduction='mean', sample_wise=True)
    loss = l1_loss(pred, target, weight)
    assert loss.shape == ()
    assert loss.item() == 0.5

    masked_tv_loss = MaskedTVLoss(loss_weight=1.0)
    pred = torch.zeros((1, 1, 6, 6))
    mask = torch.zeros_like(pred)
    mask[..., 2:4, 2:4] = 1.
    pred[..., 3, :] = 1.
    loss = masked_tv_loss(pred, mask)
    assert loss.shape == ()
    npt.assert_almost_equal(loss.item(), 1.)

    # test PSNR Loss
    psnr_loss = PSNRLoss(loss_weight=1.0)
    pred = torch.ones(1, 1, 64, 64)
    target = torch.zeros(1, 1, 64, 64)
    loss = psnr_loss(pred, target)
    assert loss.shape == ()
    assert loss.item() == 0.0
    loss = psnr_loss(target, target)
    assert loss.item() == -80.0


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

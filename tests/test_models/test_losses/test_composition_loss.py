# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy.testing as npt
import pytest
import torch

from mmagic.models import (CharbonnierCompLoss, L1CompositionLoss,
                           MSECompositionLoss)


def test_composition_losses():
    with pytest.raises(ValueError):
        # only 'none', 'mean' and 'sum' are supported
        L1CompositionLoss(reduction='InvalidValue')

    with pytest.raises(ValueError):
        # only 'none', 'mean' and 'sum' are supported
        MSECompositionLoss(reduction='InvalidValue')

    with pytest.raises(ValueError):
        # only 'none', 'mean' and 'sum' are supported
        CharbonnierCompLoss(reduction='InvalidValue')

    unknown_h, unknown_w = (32, 32)
    weight = torch.zeros(1, 1, 64, 64)
    weight[0, 0, :unknown_h, :unknown_w] = 1
    pred_alpha = weight.clone() * 0.5
    ori_merged = torch.ones(1, 3, 64, 64)
    fg = torch.zeros(1, 3, 64, 64)
    bg = torch.ones(1, 3, 64, 64) * 4

    l1_comp_loss = L1CompositionLoss(loss_weight=1.0, reduction='mean')
    loss = l1_comp_loss(pred_alpha, fg, bg, ori_merged)
    assert loss.shape == ()
    assert loss.item() == 2.5

    l1_comp_loss = L1CompositionLoss(loss_weight=0.5, reduction='none')
    loss = l1_comp_loss(pred_alpha, fg, bg, ori_merged, weight)
    assert loss.shape == (1, 3, 64, 64)
    assert (loss == torch.ones(1, 3, 64, 64) * weight * 0.5).all()

    l1_comp_loss = L1CompositionLoss(loss_weight=0.5, reduction='sum')
    loss = l1_comp_loss(pred_alpha, fg, bg, ori_merged, weight)
    assert loss.shape == ()
    assert loss.item() == 1536

    mse_comp_loss = MSECompositionLoss(loss_weight=1.0, reduction='mean')
    loss = mse_comp_loss(pred_alpha, fg, bg, ori_merged)
    assert loss.shape == ()
    assert loss.item() == 7.0

    mse_comp_loss = MSECompositionLoss(loss_weight=0.5, reduction='none')
    loss = mse_comp_loss(pred_alpha, fg, bg, ori_merged, weight)
    assert loss.shape == (1, 3, 64, 64)
    assert (loss == torch.ones(1, 3, 64, 64) * weight * 0.5).all()

    mse_comp_loss = MSECompositionLoss(loss_weight=0.5, reduction='sum')
    loss = mse_comp_loss(pred_alpha, fg, bg, ori_merged, weight)
    assert loss.shape == ()
    assert loss.item() == 1536

    cb_comp_loss = CharbonnierCompLoss(
        loss_weight=1.0, reduction='mean', eps=1e-12)
    loss = cb_comp_loss(pred_alpha, fg, bg, ori_merged)
    assert loss.shape == ()
    assert loss.item() == 2.5

    cb_comp_loss = CharbonnierCompLoss(
        loss_weight=0.5, reduction='none', eps=1e-6)
    loss = cb_comp_loss(pred_alpha, fg, bg, ori_merged, weight)
    assert loss.shape == (1, 3, 64, 64)
    npt.assert_almost_equal(
        loss.numpy(), torch.ones(1, 3, 64, 64) * weight * 0.5, decimal=6)

    cb_comp_loss = CharbonnierCompLoss(
        loss_weight=0.5, reduction='sum', eps=1e-6)
    loss = cb_comp_loss(pred_alpha, fg, bg, ori_merged, weight)
    assert loss.shape == ()
    assert math.isclose(loss.item(), 1536, rel_tol=1e-6)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

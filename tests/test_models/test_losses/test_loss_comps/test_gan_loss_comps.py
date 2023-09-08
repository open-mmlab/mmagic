# Copyright (c) OpenMMLab. All rights reserved.
import numpy.testing as npt
import pytest
import torch

from mmagic.models.losses import GANLossComps


def test_gan_losses():
    """Test gan losses."""
    with pytest.raises(NotImplementedError):
        GANLossComps(
            'xixihaha',
            loss_weight=1.0,
            real_label_val=1.0,
            fake_label_val=0.0)

    input_1 = torch.ones(1, 1)
    input_2 = torch.ones(1, 3, 6, 6) * 2

    # vanilla
    gan_loss = GANLossComps(
        'vanilla', loss_weight=2.0, real_label_val=1.0, fake_label_val=0.0)
    loss = gan_loss(input_1, True, is_disc=False)
    npt.assert_almost_equal(loss.item(), 0.6265233)
    loss = gan_loss(input_1, False, is_disc=False)
    npt.assert_almost_equal(loss.item(), 2.6265232)
    loss = gan_loss(input_1, True, is_disc=True)
    npt.assert_almost_equal(loss.item(), 0.3132616)
    loss = gan_loss(input_1, False, is_disc=True)
    npt.assert_almost_equal(loss.item(), 1.3132616)

    # lsgan
    gan_loss = GANLossComps(
        'lsgan', loss_weight=2.0, real_label_val=1.0, fake_label_val=0.0)
    loss = gan_loss(input_2, True, is_disc=False)
    npt.assert_almost_equal(loss.item(), 2.0)
    loss = gan_loss(input_2, False, is_disc=False)
    npt.assert_almost_equal(loss.item(), 8.0)
    loss = gan_loss(input_2, True, is_disc=True)
    npt.assert_almost_equal(loss.item(), 1.0)
    loss = gan_loss(input_2, False, is_disc=True)
    npt.assert_almost_equal(loss.item(), 4.0)

    # wgan
    gan_loss = GANLossComps(
        'wgan', loss_weight=2.0, real_label_val=1.0, fake_label_val=0.0)
    loss = gan_loss(input_2, True, is_disc=False)
    npt.assert_almost_equal(loss.item(), -4.0)
    loss = gan_loss(input_2, False, is_disc=False)
    npt.assert_almost_equal(loss.item(), 4)
    loss = gan_loss(input_2, True, is_disc=True)
    npt.assert_almost_equal(loss.item(), -2.0)
    loss = gan_loss(input_2, False, is_disc=True)
    npt.assert_almost_equal(loss.item(), 2.0)

    # wgan
    gan_loss = GANLossComps(
        'wgan-logistic-ns',
        loss_weight=2.0,
        real_label_val=1.0,
        fake_label_val=0.0)
    loss = gan_loss(input_2, True, is_disc=False)
    assert loss.item() > 0
    loss = gan_loss(input_2, False, is_disc=False)
    assert loss.item() > 0

    # hinge
    gan_loss = GANLossComps(
        'hinge', loss_weight=2.0, real_label_val=1.0, fake_label_val=0.0)
    loss = gan_loss(input_2, True, is_disc=False)
    npt.assert_almost_equal(loss.item(), -4.0)
    loss = gan_loss(input_2, False, is_disc=False)
    npt.assert_almost_equal(loss.item(), -4.0)
    loss = gan_loss(input_2, True, is_disc=True)
    npt.assert_almost_equal(loss.item(), 0.0)
    loss = gan_loss(input_2, False, is_disc=True)
    npt.assert_almost_equal(loss.item(), 3.0)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

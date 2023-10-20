# Copyright (c) OpenMMLab. All rights reserved.
import numpy
import numpy.testing as npt
import pytest
import torch

from mmagic.models import GANLoss, GaussianBlur


def test_gan_losses():
    """Test gan losses."""
    with pytest.raises(NotImplementedError):
        GANLoss(
            'xixihaha',
            loss_weight=1.0,
            real_label_val=1.0,
            fake_label_val=0.0)

    input_1 = torch.ones(1, 1)
    input_2 = torch.ones(1, 3, 6, 6) * 2

    # vanilla
    gan_loss = GANLoss(
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
    gan_loss = GANLoss(
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
    gan_loss = GANLoss(
        'wgan', loss_weight=2.0, real_label_val=1.0, fake_label_val=0.0)
    loss = gan_loss(input_2, True, is_disc=False)
    npt.assert_almost_equal(loss.item(), -4.0)
    loss = gan_loss(input_2, False, is_disc=False)
    npt.assert_almost_equal(loss.item(), 4)
    loss = gan_loss(input_2, True, is_disc=True)
    npt.assert_almost_equal(loss.item(), -2.0)
    loss = gan_loss(input_2, False, is_disc=True)
    npt.assert_almost_equal(loss.item(), 2.0)

    # hinge
    gan_loss = GANLoss(
        'hinge', loss_weight=2.0, real_label_val=1.0, fake_label_val=0.0)
    loss = gan_loss(input_2, True, is_disc=False)
    npt.assert_almost_equal(loss.item(), -4.0)
    loss = gan_loss(input_2, False, is_disc=False)
    npt.assert_almost_equal(loss.item(), -4.0)
    loss = gan_loss(input_2, True, is_disc=True)
    npt.assert_almost_equal(loss.item(), 0.0)
    loss = gan_loss(input_2, False, is_disc=True)
    npt.assert_almost_equal(loss.item(), 3.0)

    # smgan
    mask = torch.ones(1, 3, 6, 6)
    gan_loss = GANLoss(
        'smgan', loss_weight=2.0, real_label_val=1.0, fake_label_val=0.0)
    loss = gan_loss(input_2, True, is_disc=False, mask=mask)
    npt.assert_almost_equal(loss.item(), 2.0)
    loss = gan_loss(input_2, False, is_disc=False, mask=mask)
    npt.assert_almost_equal(loss.item(), 8.0)
    loss = gan_loss(input_2, True, is_disc=True, mask=mask)
    npt.assert_almost_equal(loss.item(), 1.0)
    loss = gan_loss(input_2, False, is_disc=True, mask=mask)
    npt.assert_almost_equal(loss.item(), 3.786323, decimal=6)
    mask = torch.ones(1, 3, 6, 5)
    loss = gan_loss(input_2, True, is_disc=False, mask=mask)
    npt.assert_almost_equal(loss.item(), 2.0)

    if torch.cuda.is_available():
        input_2 = input_2.cuda()
        mask = torch.ones(1, 3, 6, 6).cuda()
        gan_loss = GANLoss(
            'smgan', loss_weight=2.0, real_label_val=1.0, fake_label_val=0.0)
        loss = gan_loss(input_2, True, is_disc=False, mask=mask)
        npt.assert_almost_equal(loss.item(), 2.0)
        loss = gan_loss(input_2, False, is_disc=False, mask=mask)
        npt.assert_almost_equal(loss.item(), 8.0)
        loss = gan_loss(input_2, True, is_disc=True, mask=mask)
        npt.assert_almost_equal(loss.item(), 1.0)
        loss = gan_loss(input_2, False, is_disc=True, mask=mask)
        npt.assert_almost_equal(loss.item(), 3.786323, decimal=6)

    # test GaussianBlur for smgan
    with pytest.raises(TypeError):
        gausian_blur = GaussianBlur(kernel_size=71, sigma=2)
        gausian_blur(mask).detach().cpu()

    with pytest.raises(TypeError):
        gausian_blur = GaussianBlur(kernel_size=(70, 70))
        gausian_blur(mask).detach().cpu()

    with pytest.raises(TypeError):
        mask = numpy.ones((1, 3, 6, 6))
        gausian_blur = GaussianBlur()
        gausian_blur(mask).detach().cpu()

    with pytest.raises(ValueError):
        mask = torch.ones(1, 3)
        gausian_blur = GaussianBlur()
        gausian_blur(mask).detach().cpu()


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

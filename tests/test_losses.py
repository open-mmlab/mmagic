import math
from unittest.mock import patch

import numpy.testing as npt
import pytest
import torch

from mmedit.models.losses import (CharbonnierCompLoss, CharbonnierLoss,
                                  DiscShiftLoss, GANLoss, GradientLoss,
                                  GradientPenaltyLoss, L1CompositionLoss,
                                  L1Loss, MaskedTVLoss, MSECompositionLoss,
                                  MSELoss, PerceptualLoss, PerceptualVGG,
                                  mask_reduce_loss, reduce_loss)


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


@patch.object(PerceptualVGG, 'init_weights')
def test_perceptual_loss(init_weights):
    if torch.cuda.is_available():
        loss_percep = PerceptualLoss(layer_weights={'0': 1.}).cuda()
        x = torch.randn(1, 3, 16, 16).cuda()
        x.requires_grad = True
        gt = torch.randn(1, 3, 16, 16).cuda()
        percep, style = loss_percep(x, gt)

        assert percep.item() > 0
        assert style.item() > 0

        optim = torch.optim.SGD(params=[x], lr=10)
        optim.zero_grad()
        percep.backward()
        optim.step()

        percep_new, _ = loss_percep(x, gt)
        assert percep_new < percep

        loss_percep = PerceptualLoss(
            layer_weights={
                '0': 1.
            }, perceptual_weight=0.).cuda()
        x = torch.randn(1, 3, 16, 16).cuda()
        gt = torch.randn(1, 3, 16, 16).cuda()
        percep, style = loss_percep(x, gt)
        assert percep is None and style > 0

        loss_percep = PerceptualLoss(
            layer_weights={
                '0': 1.
            }, style_weight=0.).cuda()
        x = torch.randn(1, 3, 16, 16).cuda()
        gt = torch.randn(1, 3, 16, 16).cuda()
        percep, style = loss_percep(x, gt)
        assert style is None and percep > 0
    # test whether vgg type is valid
    with pytest.raises(AssertionError):
        loss_percep = PerceptualLoss(layer_weights={'0': 1.}, vgg_type='igccc')
    # test whether criterion is valid
    with pytest.raises(NotImplementedError):
        loss_percep = PerceptualLoss(
            layer_weights={'0': 1.}, criterion='igccc')

    layer_name_list = ['2', '10', '30']
    vgg_model = PerceptualVGG(
        layer_name_list,
        use_input_norm=False,
        vgg_type='vgg16',
        pretrained='torchvision://vgg16')
    x = torch.rand((1, 3, 32, 32))
    output = vgg_model(x)
    assert isinstance(output, dict)
    assert len(output) == len(layer_name_list)
    assert set(output.keys()) == set(layer_name_list)

    # test whether the layer name is valid
    with pytest.raises(AssertionError):
        layer_name_list = ['2', '10', '30', '100']
        vgg_model = PerceptualVGG(
            layer_name_list,
            use_input_norm=False,
            vgg_type='vgg16',
            pretrained='torchvision://vgg16')

    # reset mock to clear some memory usage
    init_weights.reset_mock()


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

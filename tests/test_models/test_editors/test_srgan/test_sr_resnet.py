# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

import pytest
import torch
from mmengine.optim import OptimWrapper
from torch.optim import Adam

from mmedit.data_element import EditDataSample, PixelData
from mmedit.models import SRGAN, EditDataPreprocessor, ModifiedVGG, MSRResNet
from mmedit.models.losses import GANLoss, L1Loss, PerceptualLoss, PerceptualVGG


@patch.object(PerceptualVGG, 'init_weights')
def test_srgan_resnet(init_weights):

    model = SRGAN(
        generator=dict(
            type='MSRResNet',
            in_channels=3,
            out_channels=3,
            mid_channels=4,
            num_blocks=4,
            upscale_factor=4),
        discriminator=dict(type='ModifiedVGG', in_channels=3, mid_channels=8),
        pixel_loss=dict(type='L1Loss', loss_weight=1e-2, reduction='mean'),
        perceptual_loss=dict(
            type='PerceptualLoss',
            layer_weights={'34': 1.0},
            vgg_type='vgg19',
            perceptual_weight=1.0,
            style_weight=0,
            norm_img=False),
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            loss_weight=5e-3,
            real_label_val=1.0,
            fake_label_val=0),
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=EditDataPreprocessor())

    assert isinstance(model, SRGAN)
    assert isinstance(model.generator, MSRResNet)
    assert isinstance(model.discriminator, ModifiedVGG)
    assert isinstance(model.pixel_loss, L1Loss)
    assert isinstance(model.perceptual_loss, PerceptualLoss)
    assert isinstance(model.gan_loss, GANLoss)

    optimizer_g = Adam(
        model.generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    optimizer_d = Adam(
        model.discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    optim_wrapper = dict(
        generator=OptimWrapper(optimizer_g),
        discriminator=OptimWrapper(optimizer_d))

    # prepare data
    inputs = torch.rand(3, 32, 32)
    target = torch.rand(3, 128, 128)
    data_sample = EditDataSample(gt_img=PixelData(data=target))
    data = [dict(inputs=inputs, data_sample=data_sample)]

    # train
    log_vars = model.train_step(data, optim_wrapper)
    assert isinstance(log_vars, dict)
    assert set(log_vars.keys()) == set([
        'loss_gan', 'loss_pix', 'loss_perceptual', 'loss_d_real', 'loss_d_fake'
    ])

    # val
    output = model.val_step(data)
    assert output[0].pred_img.data.shape == (3, 128, 128)

    # feat
    output = model(torch.rand(1, 3, 32, 32), mode='tensor')
    assert output.shape == (1, 3, 128, 128)

    # reset mock to clear some memory usage
    init_weights.reset_mock()


def test_srresnet_backbone():
    """Test SRResNet backbone."""

    # x2 model
    MSRResNet(
        in_channels=3,
        out_channels=3,
        mid_channels=8,
        num_blocks=2,
        upscale_factor=2)
    # x3 model, initialization and forward (cpu)
    net = MSRResNet(
        in_channels=3,
        out_channels=3,
        mid_channels=8,
        num_blocks=2,
        upscale_factor=3)
    input_shape = (1, 3, 12, 12)
    img = torch.rand(input_shape)
    output = net(img)
    assert output.shape == (1, 3, 36, 36)
    # x4 modeland, initialization and forward (cpu)
    net = MSRResNet(
        in_channels=3,
        out_channels=3,
        mid_channels=8,
        num_blocks=2,
        upscale_factor=4)
    output = net(img)
    assert output.shape == (1, 3, 48, 48)

    # x4 model forward (gpu)
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 3, 48, 48)

    with pytest.raises(ValueError):
        # Currently supported upscale_factor is [2, 3, 4]
        MSRResNet(
            in_channels=3,
            out_channels=3,
            mid_channels=64,
            num_blocks=16,
            upscale_factor=16)


def test_discriminator():
    """Test discriminator backbone."""

    # model, initialization and forward (cpu)
    net = ModifiedVGG(in_channels=3, mid_channels=64)
    input_shape = (1, 3, 128, 128)
    img = torch.rand(input_shape)
    output = net(img)
    assert output.shape == (1, 1)

    # model, initialization and forward (gpu)
    if torch.cuda.is_available():
        net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 1)

    with pytest.raises(AssertionError):
        # input size must be 128 * 128
        input_shape = (1, 3, 64, 64)
        img = torch.rand(input_shape)
        output = net(img)

# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

import torch
from mmengine.optim import OptimWrapper
from torch.optim import Adam

from mmedit.models import SRGAN, EditDataPreprocessor, GLEANStyleGANv2
from mmedit.models.editors.glean import StyleGANv2Discriminator
from mmedit.models.losses import (GANLoss, MSELoss, PerceptualLoss,
                                  PerceptualVGG)
from mmedit.structures import EditDataSample, PixelData


@patch.object(PerceptualVGG, 'init_weights')
def test_glean(init_weights):

    model = SRGAN(
        generator=dict(
            type='GLEANStyleGANv2', in_size=16, out_size=64, style_channels=4),
        discriminator=dict(
            type='StyleGANv2Discriminator', in_size=64, pretrained=None),
        pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
        perceptual_loss=dict(
            type='PerceptualLoss',
            layer_weights={'21': 1.0},
            vgg_type='vgg16',
            perceptual_weight=1e-2,
            style_weight=0,
            norm_img=True,
            criterion='mse',
            pretrained='torchvision://vgg16'),
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            loss_weight=1e-2,
            real_label_val=1.0,
            fake_label_val=0),
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=EditDataPreprocessor())

    assert isinstance(model, SRGAN)
    assert isinstance(model.generator, GLEANStyleGANv2)
    assert isinstance(model.discriminator, StyleGANv2Discriminator)
    assert isinstance(model.pixel_loss, MSELoss)
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
    inputs = torch.rand(3, 16, 16)
    target = torch.rand(3, 64, 64)
    data_sample = EditDataSample(gt_img=PixelData(data=target))
    data = [dict(inputs=inputs, data_sample=data_sample)]

    # train
    log_vars = model.train_step(data, optim_wrapper)
    assert isinstance(log_vars, dict)
    assert set(log_vars.keys()) == set([
        'loss_pix', 'loss_perceptual', 'loss_gan', 'loss_d_real', 'loss_d_fake'
    ])

    # val
    output = model.val_step(data)
    assert output[0].pred_img.data.shape == (3, 64, 64)

    # feat
    output = model(torch.rand(1, 3, 16, 16), mode='tensor')
    assert output.shape == (1, 3, 64, 64)

    # reset mock to clear some memory usage
    init_weights.reset_mock()

# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

import torch
from mmengine.optim import OptimWrapper
from torch.optim import Adam

from mmagic.models import (DataPreprocessor, RealESRGAN, RRDBNet,
                           UNetDiscriminatorWithSpectralNorm)
from mmagic.models.losses import GANLoss, L1Loss, PerceptualLoss, PerceptualVGG
from mmagic.structures import DataSample


@patch.object(PerceptualVGG, 'init_weights')
def test_real_esrgan(init_weights):

    model = RealESRGAN(
        generator=dict(
            type='RRDBNet',
            in_channels=3,
            out_channels=3,
            mid_channels=4,
            num_blocks=4,
            growth_channels=4,
            upscale_factor=4),
        discriminator=dict(
            type='UNetDiscriminatorWithSpectralNorm',
            in_channels=3,
            mid_channels=4,
            skip_connection=True),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        perceptual_loss=dict(
            type='PerceptualLoss',
            layer_weights={
                '2': 0.1,
                '7': 0.1,
                '16': 1.0,
                '25': 1.0,
                '34': 1.0,
            },
            vgg_type='vgg19',
            perceptual_weight=1.0,
            style_weight=0,
            norm_img=False),
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            loss_weight=1e-1,
            real_label_val=1.0,
            fake_label_val=0),
        is_use_sharpened_gt_in_pixel=False,
        is_use_sharpened_gt_in_percep=False,
        is_use_sharpened_gt_in_gan=False,
        is_use_ema=False,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=DataPreprocessor())

    assert isinstance(model, RealESRGAN)
    assert isinstance(model.generator, RRDBNet)
    assert isinstance(model.discriminator, UNetDiscriminatorWithSpectralNorm)
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
    inputs = torch.rand(1, 3, 32, 32)
    target = torch.rand(3, 128, 128)
    data_sample = DataSample(gt_img=target, gt_unsharp=target)
    data = dict(inputs=inputs, data_samples=[data_sample])

    # train
    log_vars = model.train_step(data, optim_wrapper)
    assert isinstance(log_vars, dict)
    assert set(log_vars.keys()) == set([
        'loss_gan', 'loss_pix', 'loss_perceptual', 'loss_d_real', 'loss_d_fake'
    ])

    # val
    output = model.val_step(data)
    assert output[0].output.pred_img.shape == (3, 128, 128)

    # val_ema
    model.generator_ema = model.generator
    model.is_use_ema = True
    output = model.val_step(data)
    assert output[0].output.pred_img.shape == (3, 128, 128)

    # feat
    output = model(torch.rand(1, 3, 32, 32), mode='tensor')
    assert output.shape == (1, 3, 128, 128)

    # train_unsharp
    model.is_use_sharpened_gt_in_pixel = True
    model.is_use_sharpened_gt_in_percep = True
    model.is_use_sharpened_gt_in_gan = False
    log_vars = model.train_step(data, optim_wrapper)
    assert isinstance(log_vars, dict)
    assert set(log_vars.keys()) == set([
        'loss_gan', 'loss_pix', 'loss_perceptual', 'loss_d_real', 'loss_d_fake'
    ])

    # reset mock to clear some memory usage
    init_weights.reset_mock()


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

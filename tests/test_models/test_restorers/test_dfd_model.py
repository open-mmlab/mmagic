# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from mmcv.runner import obj_from_dict
from mmcv.utils.config import Config

from mmedit.models.builder import build_model


def test_dfd_model():
    dictionary = dict()
    parts = ['left_eye', 'right_eye', 'nose', 'mouth']
    part_sizes = np.array([80, 80, 50, 110])
    channel_sizes = np.array([128, 256, 512, 512])
    for j, size in enumerate([256, 128, 64, 32]):
        dictionary[size] = dict()
        for i, part in enumerate(parts):
            dictionary[size][part] = torch.rand(32, channel_sizes[j],
                                                part_sizes[i] // (2**(j + 1)),
                                                part_sizes[i] // (2**(j + 1)))

    model_cfg_pre = dict(
        type='DFD',
        generator=dict(type='DFDNet', dictionary=dictionary),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))

    model_cfg = dict(
        type='DFD',
        generator=dict(type='DFDNet', dictionary=dictionary),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        discriminator=dict(
            type='MultiLayerDiscriminator',
            in_channels=3,
            max_channels=64,
            with_spectral_norm=True),
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            loss_weight=0.005,
            real_label_val=1.0,
            fake_label_val=0),
        perceptual_loss=dict(
            type='PerceptualLoss',
            layer_weights={'29': 1.0},
            vgg_type='vgg19',
            perceptual_weight=1e-2,
            style_weight=0,
            criterion='mse'))

    train_cfg = dict(fix_iter=0, disc_steps=1)
    test_cfg = Config(dict(metrics=['PSNR', 'SSIM'], crop_border=1))

    # build restorer
    build_model(model_cfg_pre)
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test attributes
    assert restorer.__class__.__name__ == 'DFD'

    # prepare data
    inputs = torch.rand(1, 3, 512, 512)
    part_locations = dict(
        left_eye=torch.tensor([[146, 184, 225, 263]]),
        right_eye=torch.tensor([[283, 179, 374, 270]]),
        nose=torch.tensor([[229, 296, 282, 349]]),
        mouth=torch.tensor([[229, 296, 282, 349]]))
    targets = torch.rand(1, 3, 512, 512)
    data_batch = {'lq': inputs, 'gt': targets, 'location': part_locations}

    # prepare optimizer
    optim_cfg = dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))
    generator = obj_from_dict(optim_cfg, torch.optim,
                              dict(params=restorer.parameters()))
    discriminator = obj_from_dict(optim_cfg, torch.optim,
                                  dict(params=restorer.parameters()))
    optimizer = dict(generator=generator, discriminator=discriminator)

    # test train_step and forward_test (cpu)
    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['log_vars'], dict)
    assert isinstance(outputs['log_vars']['loss_pixel'], float)
    assert outputs['num_samples'] == 1
    assert outputs['results']['lq'].shape == data_batch['lq'].shape
    assert outputs['results']['gt'].shape == data_batch['gt'].shape
    assert torch.is_tensor(outputs['results']['output'])
    assert outputs['results']['output'].size() == targets.shape

    # test train_step and forward_test (gpu)
    if torch.cuda.is_available():
        restorer = restorer.cuda()
        data_batch = {
            'lq': inputs.cuda(),
            'gt': targets.cuda(),
            'location': part_locations
        }

        # train_step
        optim_cfg = dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))
        generator = obj_from_dict(optim_cfg, torch.optim,
                                  dict(params=restorer.parameters()))
        discriminator = obj_from_dict(optim_cfg, torch.optim,
                                      dict(params=restorer.parameters()))
        optimizer = dict(generator=generator, discriminator=discriminator)
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        assert isinstance(outputs['log_vars']['loss_pixel'], float)
        assert outputs['num_samples'] == 1
        assert outputs['results']['lq'].shape == data_batch['lq'].shape
        assert outputs['results']['gt'].shape == data_batch['gt'].shape
        assert torch.is_tensor(outputs['results']['output'])
        assert outputs['results']['output'].size() == targets.shape

        # val_step
        result = restorer.val_step(data_batch, meta=[{'gt_path': ''}])
        assert isinstance(result, dict)
        assert isinstance(result['eval_result'], dict)
        assert result['eval_result'].keys() == set({'PSNR', 'SSIM'})
        assert isinstance(result['eval_result']['PSNR'], np.float64)
        assert isinstance(result['eval_result']['SSIM'], np.float64)

        with pytest.raises(AssertionError):
            # evaluation with metrics must have gt images
            restorer(lq=inputs.cuda(), location=part_locations, test_mode=True)

        with pytest.raises(TypeError):
            restorer.init_weights(pretrained=1)
        with pytest.raises(OSError):
            restorer.init_weights(pretrained='')

# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

import torch
from mmcv.runner import obj_from_dict

from mmedit.models import build_model
from mmedit.models.backbones import MSRResNet
from mmedit.models.components import ModifiedVGG
from mmedit.models.losses import GANLoss, L1Loss


def test_esrgan():

    model_cfg = dict(
        type='ESRGAN',
        generator=dict(
            type='MSRResNet',
            in_channels=3,
            out_channels=3,
            mid_channels=4,
            num_blocks=1,
            upscale_factor=4),
        discriminator=dict(type='ModifiedVGG', in_channels=3, mid_channels=2),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0,
            loss_weight=5e-3))

    train_cfg = None
    test_cfg = None

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test attributes
    assert restorer.__class__.__name__ == 'ESRGAN'
    assert isinstance(restorer.generator, MSRResNet)
    assert isinstance(restorer.discriminator, ModifiedVGG)
    assert isinstance(restorer.pixel_loss, L1Loss)
    assert isinstance(restorer.gan_loss, GANLoss)

    # prepare data
    inputs = torch.rand(1, 3, 32, 32)
    targets = torch.rand(1, 3, 128, 128)
    data_batch = {'lq': inputs, 'gt': targets}

    # prepare optimizer
    optim_cfg = dict(type='Adam', lr=2e-4, betas=(0.9, 0.999))
    optimizer = {
        'generator':
        obj_from_dict(optim_cfg, torch.optim,
                      dict(
                          params=getattr(restorer, 'generator').parameters())),
        'discriminator':
        obj_from_dict(
            optim_cfg, torch.optim,
            dict(params=getattr(restorer, 'discriminator').parameters()))
    }

    # test train_step
    with patch.object(
            restorer,
            'perceptual_loss',
            return_value=(torch.tensor(1.0), torch.tensor(2.0))):
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        for v in [
                'loss_perceptual', 'loss_gan', 'loss_d_real', 'loss_d_fake',
                'loss_pix'
        ]:
            assert isinstance(outputs['log_vars'][v], float)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['lq'], data_batch['lq'])
        assert torch.equal(outputs['results']['gt'], data_batch['gt'])
        assert torch.is_tensor(outputs['results']['output'])
        assert outputs['results']['output'].size() == (1, 3, 128, 128)

    # test train_step and forward_test (gpu)
    if torch.cuda.is_available():
        restorer = restorer.cuda()
        optimizer = {
            'generator':
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(params=getattr(restorer, 'generator').parameters())),
            'discriminator':
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(params=getattr(restorer, 'discriminator').parameters()))
        }
        data_batch = {'lq': inputs.cuda(), 'gt': targets.cuda()}

        # train_step
        with patch.object(
                restorer,
                'perceptual_loss',
                return_value=(torch.tensor(1.0).cuda(),
                              torch.tensor(2.0).cuda())):
            outputs = restorer.train_step(data_batch, optimizer)
            assert isinstance(outputs, dict)
            assert isinstance(outputs['log_vars'], dict)
            for v in [
                    'loss_perceptual', 'loss_gan', 'loss_d_real',
                    'loss_d_fake', 'loss_pix'
            ]:
                assert isinstance(outputs['log_vars'][v], float)
            assert outputs['num_samples'] == 1
            assert torch.equal(outputs['results']['lq'],
                               data_batch['lq'].cpu())
            assert torch.equal(outputs['results']['gt'],
                               data_batch['gt'].cpu())
            assert torch.is_tensor(outputs['results']['output'])
            assert outputs['results']['output'].size() == (1, 3, 128, 128)

    # test disc_steps and disc_init_steps
    data_batch = {'lq': inputs.cpu(), 'gt': targets.cpu()}
    train_cfg = dict(disc_steps=2, disc_init_steps=2)
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    with patch.object(
            restorer,
            'perceptual_loss',
            return_value=(torch.tensor(1.0), torch.tensor(2.0))):
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        for v in ['loss_d_real', 'loss_d_fake']:
            assert isinstance(outputs['log_vars'][v], float)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['lq'], data_batch['lq'])
        assert torch.equal(outputs['results']['gt'], data_batch['gt'])
        assert torch.is_tensor(outputs['results']['output'])
        assert outputs['results']['output'].size() == (1, 3, 128, 128)

    # test without pixel loss and perceptual loss
    model_cfg_ = model_cfg.copy()
    model_cfg_.pop('pixel_loss')
    restorer = build_model(model_cfg_, train_cfg=None, test_cfg=None)

    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['log_vars'], dict)
    for v in ['loss_gan', 'loss_d_real', 'loss_d_fake']:
        assert isinstance(outputs['log_vars'][v], float)
    assert outputs['num_samples'] == 1
    assert torch.equal(outputs['results']['lq'], data_batch['lq'])
    assert torch.equal(outputs['results']['gt'], data_batch['gt'])
    assert torch.is_tensor(outputs['results']['output'])
    assert outputs['results']['output'].size() == (1, 3, 128, 128)

    # test train_step w/o loss_percep
    restorer = build_model(model_cfg, train_cfg=None, test_cfg=None)
    with patch.object(
            restorer, 'perceptual_loss',
            return_value=(None, torch.tensor(2.0))):
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        for v in [
                'loss_style', 'loss_gan', 'loss_d_real', 'loss_d_fake',
                'loss_pix'
        ]:
            assert isinstance(outputs['log_vars'][v], float)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['lq'], data_batch['lq'])
        assert torch.equal(outputs['results']['gt'], data_batch['gt'])
        assert torch.is_tensor(outputs['results']['output'])
        assert outputs['results']['output'].size() == (1, 3, 128, 128)

    # test train_step w/o loss_style
    restorer = build_model(model_cfg, train_cfg=None, test_cfg=None)
    with patch.object(
            restorer, 'perceptual_loss',
            return_value=(torch.tensor(2.0), None)):
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        for v in [
                'loss_perceptual', 'loss_gan', 'loss_d_real', 'loss_d_fake',
                'loss_pix'
        ]:
            assert isinstance(outputs['log_vars'][v], float)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['lq'], data_batch['lq'])
        assert torch.equal(outputs['results']['gt'], data_batch['gt'])
        assert torch.is_tensor(outputs['results']['output'])
        assert outputs['results']['output'].size() == (1, 3, 128, 128)

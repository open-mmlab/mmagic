# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from mmcv.runner import obj_from_dict
from mmcv.utils.config import Config

from mmedit.models.builder import build_model


def test_dic_model():
    pretrained = 'https://download.openmmlab.com/mmediting/' + \
        'restorers/dic/light_cnn_feature.pth'

    model_cfg_pre = dict(
        type='DIC',
        generator=dict(
            type='DICNet', in_channels=3, out_channels=3, mid_channels=48),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        align_loss=dict(type='MSELoss', loss_weight=0.1, reduction='mean'))

    model_cfg = dict(
        type='DIC',
        generator=dict(
            type='DICNet', in_channels=3, out_channels=3, mid_channels=48),
        discriminator=dict(type='LightCNN', in_channels=3),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        align_loss=dict(type='MSELoss', loss_weight=0.1, reduction='mean'),
        feature_loss=dict(
            type='LightCNNFeatureLoss',
            pretrained=pretrained,
            loss_weight=0.1,
            criterion='l1'),
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            loss_weight=0.005,
            real_label_val=1.0,
            fake_label_val=0))

    scale = 8
    train_cfg = None
    test_cfg = Config(dict(metrics=['PSNR', 'SSIM'], crop_border=scale))

    # build restorer
    build_model(model_cfg_pre, train_cfg=train_cfg, test_cfg=test_cfg)
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test attributes
    assert restorer.__class__.__name__ == 'DIC'

    # prepare data
    inputs = torch.rand(1, 3, 16, 16)
    targets = torch.rand(1, 3, 128, 128)
    heatmap = torch.rand(1, 68, 32, 32)
    data_batch = {'lq': inputs, 'gt': targets, 'heatmap': heatmap}

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
    assert isinstance(outputs['log_vars']['loss_pixel_v3'], float)
    assert outputs['num_samples'] == 1
    assert outputs['results']['lq'].shape == data_batch['lq'].shape
    assert outputs['results']['gt'].shape == data_batch['gt'].shape
    assert torch.is_tensor(outputs['results']['output'])
    assert outputs['results']['output'].size() == (1, 3, 128, 128)

    # test train_step and forward_test (gpu)
    if torch.cuda.is_available():
        restorer = restorer.cuda()
        data_batch = {
            'lq': inputs.cuda(),
            'gt': targets.cuda(),
            'heatmap': heatmap.cuda()
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
        assert isinstance(outputs['log_vars']['loss_pixel_v3'], float)
        assert outputs['num_samples'] == 1
        assert outputs['results']['lq'].shape == data_batch['lq'].shape
        assert outputs['results']['gt'].shape == data_batch['gt'].shape
        assert torch.is_tensor(outputs['results']['output'])
        assert outputs['results']['output'].size() == (1, 3, 128, 128)

        # val_step
        data_batch.pop('heatmap')
        result = restorer.val_step(data_batch, meta=[{'gt_path': ''}])
        assert isinstance(result, dict)
        assert isinstance(result['eval_result'], dict)
        assert result['eval_result'].keys() == set({'PSNR', 'SSIM'})
        assert isinstance(result['eval_result']['PSNR'], np.float64)
        assert isinstance(result['eval_result']['SSIM'], np.float64)

        with pytest.raises(AssertionError):
            # evaluation with metrics must have gt images
            restorer(lq=inputs.cuda(), test_mode=True)

        with pytest.raises(TypeError):
            restorer.init_weights(pretrained=1)
        with pytest.raises(OSError):
            restorer.init_weights(pretrained='')

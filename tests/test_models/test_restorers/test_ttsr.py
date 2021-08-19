# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.runner import obj_from_dict
from mmcv.utils.config import Config

from mmedit.models import build_backbone, build_model
from mmedit.models.backbones.sr_backbones.ttsr_net import (CSFI2, CSFI3, SFE,
                                                           MergeFeatures)


def test_sfe():
    inputs = torch.rand(2, 3, 48, 48)
    sfe = SFE(3, 64, 16, 1.)
    outputs = sfe(inputs)
    assert outputs.shape == (2, 64, 48, 48)


def test_csfi():
    inputs1 = torch.rand(2, 16, 24, 24)
    inputs2 = torch.rand(2, 16, 48, 48)
    inputs4 = torch.rand(2, 16, 96, 96)

    csfi2 = CSFI2(mid_channels=16)
    out1, out2 = csfi2(inputs1, inputs2)
    assert out1.shape == (2, 16, 24, 24)
    assert out2.shape == (2, 16, 48, 48)

    csfi3 = CSFI3(mid_channels=16)
    out1, out2, out4 = csfi3(inputs1, inputs2, inputs4)
    assert out1.shape == (2, 16, 24, 24)
    assert out2.shape == (2, 16, 48, 48)
    assert out4.shape == (2, 16, 96, 96)


def test_merge_features():
    inputs1 = torch.rand(2, 16, 24, 24)
    inputs2 = torch.rand(2, 16, 48, 48)
    inputs4 = torch.rand(2, 16, 96, 96)

    merge_features = MergeFeatures(mid_channels=16, out_channels=3)
    out = merge_features(inputs1, inputs2, inputs4)
    assert out.shape == (2, 3, 96, 96)


def test_ttsr_net():
    inputs = torch.rand(2, 3, 24, 24)
    soft_attention = torch.rand(2, 1, 24, 24)
    t_level3 = torch.rand(2, 64, 24, 24)
    t_level2 = torch.rand(2, 32, 48, 48)
    t_level1 = torch.rand(2, 16, 96, 96)

    ttsr_cfg = dict(
        type='TTSRNet',
        in_channels=3,
        out_channels=3,
        mid_channels=16,
        texture_channels=16)
    ttsr = build_backbone(ttsr_cfg)
    outputs = ttsr(inputs, soft_attention, (t_level3, t_level2, t_level1))

    assert outputs.shape == (2, 3, 96, 96)


def test_ttsr():
    model_cfg = dict(
        type='TTSR',
        generator=dict(
            type='TTSRNet',
            in_channels=3,
            out_channels=3,
            mid_channels=64,
            num_blocks=(16, 16, 8, 4)),
        extractor=dict(type='LTE', load_pretrained_vgg=False),
        transformer=dict(type='SearchTransformer'),
        discriminator=dict(type='TTSRDiscriminator', in_size=64),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        perceptual_loss=dict(
            type='PerceptualLoss',
            layer_weights={'29': 1.0},
            vgg_type='vgg19',
            perceptual_weight=1e-2,
            style_weight=0.001,
            criterion='mse'),
        transferal_perceptual_loss=dict(
            type='TransferalPerceptualLoss',
            loss_weight=1e-2,
            use_attention=False,
            criterion='mse'),
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            loss_weight=1e-3,
            real_label_val=1.0,
            fake_label_val=0))

    scale = 4
    train_cfg = None
    test_cfg = Config(dict(metrics=['PSNR', 'SSIM'], crop_border=scale))

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    model_cfg = dict(
        type='TTSR',
        generator=dict(
            type='TTSRNet',
            in_channels=3,
            out_channels=3,
            mid_channels=64,
            num_blocks=(16, 16, 8, 4)),
        extractor=dict(type='LTE'),
        transformer=dict(type='SearchTransformer'),
        discriminator=dict(type='TTSRDiscriminator', in_size=64),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        perceptual_loss=dict(
            type='PerceptualLoss',
            layer_weights={'29': 1.0},
            vgg_type='vgg19',
            perceptual_weight=1e-2,
            style_weight=0.001,
            criterion='mse'),
        transferal_perceptual_loss=dict(
            type='TransferalPerceptualLoss',
            loss_weight=1e-2,
            use_attention=False,
            criterion='mse'),
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            loss_weight=1e-3,
            real_label_val=1.0,
            fake_label_val=0))

    scale = 4
    train_cfg = None
    test_cfg = Config(dict(metrics=['PSNR', 'SSIM'], crop_border=scale))

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test attributes
    assert restorer.__class__.__name__ == 'TTSR'

    # prepare data
    inputs = torch.rand(1, 3, 16, 16)
    targets = torch.rand(1, 3, 64, 64)
    ref = torch.rand(1, 3, 64, 64)
    data_batch = {
        'lq': inputs,
        'gt': targets,
        'ref': ref,
        'lq_up': ref,
        'ref_downup': ref
    }

    # prepare optimizer
    optim_cfg_g = dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))
    optim_cfg_d = dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))
    optimizer = dict(
        generator=obj_from_dict(optim_cfg_g, torch.optim,
                                dict(params=restorer.parameters())),
        discriminator=obj_from_dict(optim_cfg_d, torch.optim,
                                    dict(params=restorer.parameters())))

    # test train_step and forward_test (cpu)
    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['log_vars'], dict)
    assert isinstance(outputs['log_vars']['loss_pix'], float)
    assert outputs['num_samples'] == 1
    assert outputs['results']['lq'].shape == data_batch['lq'].shape
    assert outputs['results']['gt'].shape == data_batch['gt'].shape
    assert torch.is_tensor(outputs['results']['output'])
    assert outputs['results']['output'].size() == (1, 3, 64, 64)

    # test train_step and forward_test (gpu)
    if torch.cuda.is_available():
        restorer = restorer.cuda()
        data_batch = {
            'lq': inputs.cuda(),
            'gt': targets.cuda(),
            'ref': ref.cuda(),
            'lq_up': ref.cuda(),
            'ref_downup': ref.cuda()
        }

        # train_step
        optimizer = dict(
            generator=obj_from_dict(optim_cfg_g, torch.optim,
                                    dict(params=restorer.parameters())),
            discriminator=obj_from_dict(optim_cfg_d, torch.optim,
                                        dict(params=restorer.parameters())))
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        assert isinstance(outputs['log_vars']['loss_pix'], float)
        assert outputs['num_samples'] == 1
        assert outputs['results']['lq'].shape == data_batch['lq'].shape
        assert outputs['results']['gt'].shape == data_batch['gt'].shape
        assert torch.is_tensor(outputs['results']['output'])
        assert outputs['results']['output'].size() == (1, 3, 64, 64)

        # val_step
        result = restorer.val_step(data_batch, meta=[{'gt_path': ''}])
        assert isinstance(result, dict)
        assert isinstance(result['eval_result'], dict)
        assert result['eval_result'].keys() == set({'PSNR', 'SSIM'})
        assert isinstance(result['eval_result']['PSNR'], np.float64)
        assert isinstance(result['eval_result']['SSIM'], np.float64)

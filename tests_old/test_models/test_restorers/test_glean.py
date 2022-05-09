# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import pytest
import torch

from mmedit.models import build_model


def test_glean():

    model_cfg = dict(
        type='GLEAN',
        generator=dict(
            type='GLEANStyleGANv2',
            in_size=16,
            out_size=64,
            style_channels=512),
        discriminator=dict(type='StyleGAN2Discriminator', in_size=64),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0,
            loss_weight=5e-3))

    train_cfg = None
    test_cfg = mmcv.Config(dict(metrics=['PSNR'], crop_border=0))

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # prepare data
    inputs = torch.rand(1, 3, 16, 16)
    targets = torch.rand(1, 3, 64, 64)
    data_batch = {'lq': inputs, 'gt': targets}

    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    meta = [{'lq_path': ''}]

    # test forward_test (cpu)
    with pytest.raises(ValueError):  # iteration is not None or number
        with torch.no_grad():
            restorer(
                **data_batch,
                test_mode=True,
                save_image=True,
                meta=meta,
                iteration='1')
    with pytest.raises(AssertionError):  # test with metric but gt is None
        with torch.no_grad():
            data_batch.pop('gt')
            restorer(**data_batch, test_mode=True)

    # test forward_test (gpu)
    if torch.cuda.is_available():
        data_batch = {'lq': inputs.cuda(), 'gt': targets.cuda()}
        restorer = restorer.cuda()
        with pytest.raises(ValueError):  # iteration is not None or number
            with torch.no_grad():
                restorer(
                    **data_batch,
                    test_mode=True,
                    save_image=True,
                    meta=meta,
                    iteration='1')
        with pytest.raises(AssertionError):  # test with metric but gt is None
            with torch.no_grad():
                data_batch.pop('gt')
                restorer(**data_batch, test_mode=True)

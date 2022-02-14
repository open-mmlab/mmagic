# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import mmcv
import pytest
import torch
from mmcv.runner import obj_from_dict

from mmedit.models import build_model
from mmedit.models.backbones.sr_backbones import BasicVSRNet
from mmedit.models.losses import MSELoss


def test_basicvsr_model():

    model_cfg = dict(
        type='BasicVSR',
        generator=dict(
            type='BasicVSRNet',
            mid_channels=64,
            num_blocks=30,
            spynet_pretrained=None),
        pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='sum'),
    )

    train_cfg = dict(fix_iter=1)
    train_cfg = mmcv.Config(train_cfg)
    test_cfg = None

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test attributes
    assert restorer.__class__.__name__ == 'BasicVSR'
    assert isinstance(restorer.generator, BasicVSRNet)
    assert isinstance(restorer.pixel_loss, MSELoss)

    # prepare data
    inputs = torch.rand(1, 5, 3, 64, 64)
    targets = torch.rand(1, 5, 3, 256, 256)

    if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()
        restorer = restorer.cuda()

    # prepare data and optimizer
    data_batch = {'lq': inputs, 'gt': targets}
    optim_cfg = dict(type='Adam', lr=2e-4, betas=(0.9, 0.999))
    optimizer = {
        'generator':
        obj_from_dict(optim_cfg, torch.optim,
                      dict(params=getattr(restorer, 'generator').parameters()))
    }

    # train_step (without updating spynet)
    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['log_vars'], dict)
    assert isinstance(outputs['log_vars']['loss_pix'], float)
    assert outputs['num_samples'] == 1
    assert torch.equal(outputs['results']['lq'], data_batch['lq'].cpu())
    assert torch.equal(outputs['results']['gt'], data_batch['gt'].cpu())
    assert torch.is_tensor(outputs['results']['output'])
    assert outputs['results']['output'].size() == (1, 5, 3, 256, 256)

    # train with spynet updated
    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['log_vars'], dict)
    assert isinstance(outputs['log_vars']['loss_pix'], float)
    assert outputs['num_samples'] == 1
    assert torch.equal(outputs['results']['lq'], data_batch['lq'].cpu())
    assert torch.equal(outputs['results']['gt'], data_batch['gt'].cpu())
    assert torch.is_tensor(outputs['results']['output'])
    assert outputs['results']['output'].size() == (1, 5, 3, 256, 256)

    # test forward_dummy
    with torch.no_grad():
        output = restorer.forward_dummy(data_batch['lq'])
    assert torch.is_tensor(output)
    assert output.size() == (1, 5, 3, 256, 256)

    # forward_test
    with torch.no_grad():
        outputs = restorer(**data_batch, test_mode=True)
    assert torch.equal(outputs['lq'], data_batch['lq'].cpu())
    assert torch.equal(outputs['gt'], data_batch['gt'].cpu())
    assert torch.is_tensor(outputs['output'])
    assert outputs['output'].size() == (1, 5, 3, 256, 256)

    with torch.no_grad():
        outputs = restorer(inputs, test_mode=True)
    assert torch.equal(outputs['lq'], data_batch['lq'].cpu())
    assert torch.is_tensor(outputs['output'])
    assert outputs['output'].size() == (1, 5, 3, 256, 256)

    # test with metric and save image
    train_cfg = mmcv.ConfigDict(fix_iter=1)
    test_cfg = dict(metrics=('PSNR', 'SSIM'), crop_border=0)
    test_cfg = mmcv.Config(test_cfg)

    data_batch = {
        'lq': inputs,
        'gt': targets,
        'meta': [{
            'gt_path': 'fake_path/fake_name.png',
            'key': '000'
        }]
    }

    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    if torch.cuda.is_available():
        restorer = restorer.cuda()

    with pytest.raises(AssertionError):
        # evaluation with metrics must have gt images
        restorer(lq=inputs, test_mode=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = restorer(
            **data_batch,
            test_mode=True,
            save_image=True,
            save_path=tmpdir,
            iteration=None)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['eval_result'], dict)
        assert isinstance(outputs['eval_result']['PSNR'], float)
        assert isinstance(outputs['eval_result']['SSIM'], float)

        outputs = restorer(
            **data_batch,
            test_mode=True,
            save_image=True,
            save_path=tmpdir,
            iteration=100)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['eval_result'], dict)
        assert isinstance(outputs['eval_result']['PSNR'], float)
        assert isinstance(outputs['eval_result']['SSIM'], float)

        with pytest.raises(ValueError):
            # iteration should be number or None
            restorer(
                **data_batch,
                test_mode=True,
                save_image=True,
                save_path=tmpdir,
                iteration='100')

    # forward_test (with ensemble)
    model_cfg = dict(
        type='BasicVSR',
        generator=dict(
            type='BasicVSRNet',
            mid_channels=64,
            num_blocks=30,
            spynet_pretrained=None),
        pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='sum'),
        ensemble=dict(
            type='SpatialTemporalEnsemble', is_temporal_ensemble=False),
    )

    train_cfg = dict(fix_iter=1)
    train_cfg = mmcv.Config(train_cfg)
    test_cfg = None

    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    inputs = torch.rand(1, 5, 3, 64, 64)
    targets = torch.rand(1, 5, 3, 256, 256)

    if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()
        restorer = restorer.cuda()

    data_batch = {'lq': inputs, 'gt': targets}

    with torch.no_grad():
        outputs = restorer(**data_batch, test_mode=True)
    assert torch.equal(outputs['lq'], data_batch['lq'].cpu())
    assert torch.equal(outputs['gt'], data_batch['gt'].cpu())
    assert torch.is_tensor(outputs['output'])
    assert outputs['output'].size() == (1, 5, 3, 256, 256)

    # forward_test (with unsupported ensemble)
    model_cfg = dict(
        type='BasicVSR',
        generator=dict(
            type='BasicVSRNet',
            mid_channels=64,
            num_blocks=30,
            spynet_pretrained=None),
        pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='sum'),
        ensemble=dict(type='abc', is_temporal_ensemble=False),
    )

    with pytest.raises(NotImplementedError):
        restorer = build_model(
            model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

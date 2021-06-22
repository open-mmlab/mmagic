import tempfile

import mmcv
import pytest
import torch
from mmcv.runner import obj_from_dict

from mmedit.models import build_model
from mmedit.models.backbones import MSRResNet
from mmedit.models.losses import L1Loss


def test_basic_restorer():
    model_cfg = dict(
        type='BasicRestorer',
        generator=dict(
            type='MSRResNet',
            in_channels=3,
            out_channels=3,
            mid_channels=4,
            num_blocks=1,
            upscale_factor=4),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))

    train_cfg = None
    test_cfg = None

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test attributes
    assert restorer.__class__.__name__ == 'BasicRestorer'
    assert isinstance(restorer.generator, MSRResNet)
    assert isinstance(restorer.pixel_loss, L1Loss)

    # prepare data
    inputs = torch.rand(1, 3, 2, 2)
    targets = torch.rand(1, 3, 8, 8)
    data_batch = {'lq': inputs, 'gt': targets}

    # prepare optimizer
    optim_cfg = dict(type='Adam', lr=2e-4, betas=(0.9, 0.999))
    optimizer = {
        'generator':
        obj_from_dict(optim_cfg, torch.optim,
                      dict(params=restorer.parameters()))
    }

    # test forward train
    outputs = restorer(**data_batch, test_mode=False)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['losses'], dict)
    assert isinstance(outputs['losses']['loss_pix'], torch.FloatTensor)
    assert outputs['num_samples'] == 1
    assert torch.equal(outputs['results']['lq'], data_batch['lq'])
    assert torch.equal(outputs['results']['gt'], data_batch['gt'])
    assert torch.is_tensor(outputs['results']['output'])
    assert outputs['results']['output'].size() == (1, 3, 8, 8)

    # test forward_test
    with torch.no_grad():
        outputs = restorer(**data_batch, test_mode=True)
    assert torch.equal(outputs['lq'], data_batch['lq'])
    assert torch.is_tensor(outputs['output'])
    assert outputs['output'].size() == (1, 3, 8, 8)

    # test forward_dummy
    with torch.no_grad():
        output = restorer.forward_dummy(data_batch['lq'])
    assert torch.is_tensor(output)
    assert output.size() == (1, 3, 8, 8)

    # test train_step
    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['log_vars'], dict)
    assert isinstance(outputs['log_vars']['loss_pix'], float)
    assert outputs['num_samples'] == 1
    assert torch.equal(outputs['results']['lq'], data_batch['lq'])
    assert torch.equal(outputs['results']['gt'], data_batch['gt'])
    assert torch.is_tensor(outputs['results']['output'])
    assert outputs['results']['output'].size() == (1, 3, 8, 8)

    # test train_step and forward_test (gpu)
    if torch.cuda.is_available():
        restorer = restorer.cuda()
        optimizer['generator'] = obj_from_dict(
            optim_cfg, torch.optim, dict(params=restorer.parameters()))
        data_batch = {'lq': inputs.cuda(), 'gt': targets.cuda()}

        # test forward train
        outputs = restorer(**data_batch, test_mode=False)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['losses'], dict)
        assert isinstance(outputs['losses']['loss_pix'],
                          torch.cuda.FloatTensor)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['lq'], data_batch['lq'].cpu())
        assert torch.equal(outputs['results']['gt'], data_batch['gt'].cpu())
        assert torch.is_tensor(outputs['results']['output'])
        assert outputs['results']['output'].size() == (1, 3, 8, 8)

        # forward_test
        with torch.no_grad():
            outputs = restorer(**data_batch, test_mode=True)
        assert torch.equal(outputs['lq'], data_batch['lq'].cpu())
        assert torch.is_tensor(outputs['output'])
        assert outputs['output'].size() == (1, 3, 8, 8)

        # train_step
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        assert isinstance(outputs['log_vars']['loss_pix'], float)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['lq'], data_batch['lq'].cpu())
        assert torch.equal(outputs['results']['gt'], data_batch['gt'].cpu())
        assert torch.is_tensor(outputs['results']['output'])
        assert outputs['results']['output'].size() == (1, 3, 8, 8)

    # test with metric and save image
    test_cfg = dict(metrics=('PSNR', 'SSIM'), crop_border=0)
    test_cfg = mmcv.Config(test_cfg)

    data_batch = {
        'lq': inputs,
        'gt': targets,
        'meta': [{
            'lq_path': 'fake_path/fake_name.png'
        }]
    }

    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

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

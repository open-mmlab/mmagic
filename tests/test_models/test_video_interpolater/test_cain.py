# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import mmcv
import pytest
import torch
from mmcv.runner import obj_from_dict

from mmedit.models import build_model
from mmedit.models.backbones import CAINNet
from mmedit.models.losses import L1Loss


def test_cain():
    model_cfg = dict(
        type='CAIN',
        generator=dict(type='CAINNet'),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))

    train_cfg = None
    test_cfg = None

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test attributes
    assert restorer.__class__.__name__ == 'CAIN'
    assert isinstance(restorer.generator, CAINNet)
    assert isinstance(restorer.pixel_loss, L1Loss)

    # prepare data
    inputs = torch.rand(1, 2, 3, 128, 128)
    target = torch.rand(1, 3, 128, 128)
    data_batch = {'inputs': inputs, 'target': target, 'meta': [{'key': '001'}]}

    # prepare optimizer
    optim_cfg = dict(type='Adam', lr=2e-4, betas=(0.9, 0.999))
    optimizer = {
        'generator':
        obj_from_dict(optim_cfg, torch.optim,
                      dict(params=restorer.parameters()))
    }

    # test forward_test
    with torch.no_grad():
        outputs = restorer.forward_test(**data_batch)
    assert torch.equal(outputs['inputs'], data_batch['inputs'])
    assert torch.is_tensor(outputs['output'])
    assert outputs['output'].size() == (1, 3, 128, 128)

    # test train_step
    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['log_vars'], dict)
    assert isinstance(outputs['log_vars']['loss_pix'], float)
    assert outputs['num_samples'] == 1
    assert torch.equal(outputs['results']['inputs'], data_batch['inputs'])
    assert torch.equal(outputs['results']['target'], data_batch['target'])
    assert torch.is_tensor(outputs['results']['output'])
    assert outputs['results']['output'].size() == (1, 3, 128, 128)

    # test train_step and forward_test (gpu)
    if torch.cuda.is_available():
        restorer = restorer.cuda()
        optimizer['generator'] = obj_from_dict(
            optim_cfg, torch.optim, dict(params=restorer.parameters()))
        data_batch = {
            'inputs': inputs.cuda(),
            'target': target.cuda(),
            'meta': [{
                'key': '001'
            }]
        }

        # forward_test
        with torch.no_grad():
            outputs = restorer.forward_test(**data_batch)
        assert torch.equal(outputs['inputs'], data_batch['inputs'].cpu())
        assert torch.is_tensor(outputs['output'])
        assert outputs['output'].size() == (1, 3, 128, 128)

        # train_step
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        assert isinstance(outputs['log_vars']['loss_pix'], float)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['inputs'],
                           data_batch['inputs'].cpu())
        assert torch.equal(outputs['results']['target'],
                           data_batch['target'].cpu())
        assert torch.is_tensor(outputs['results']['output'])
        assert outputs['results']['output'].size() == (1, 3, 128, 128)

    # test with metric and save image
    test_cfg = dict(metrics=('PSNR', 'SSIM'), crop_border=0)
    test_cfg = mmcv.Config(test_cfg)

    data_batch = {
        'inputs': inputs,
        'target': target,
        'meta': [{
            'key': 'fake_path/fake_name'
        }]
    }

    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    with pytest.raises(AssertionError):
        # evaluation with metrics must have target images
        restorer(inputs=inputs, test_mode=True)

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

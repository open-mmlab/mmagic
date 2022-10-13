# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import mmcv
import pytest
import torch
import torch.nn as nn
from mmcv.runner import obj_from_dict

from mmedit.models import build_model
from mmedit.models.losses import L1Loss
from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class InterpolateExample(nn.Module):
    """An example of interpolate network for testing BasicInterpolator."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(3, 3, 3, 1, 1)

    def forward(self, x):
        return self.layer(x[:, 0])

    def init_weights(self, pretrained=None):
        pass


@COMPONENTS.register_module()
class InterpolateExample2(nn.Module):
    """An example of interpolate network for testing BasicInterpolator."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(3, 3, 3, 1, 1)

    def forward(self, x):
        return self.layer(x[:, 0]).unsqueeze(1)

    def init_weights(self, pretrained=None):
        pass


def test_basic_interpolator():
    model_cfg = dict(
        type='BasicInterpolator',
        generator=dict(type='InterpolateExample'),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))

    train_cfg = None
    test_cfg = None

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test attributes
    assert restorer.__class__.__name__ == 'BasicInterpolator'
    assert isinstance(restorer.generator, InterpolateExample)
    assert isinstance(restorer.pixel_loss, L1Loss)

    # prepare data
    inputs = torch.rand(1, 2, 3, 20, 20)
    target = torch.rand(1, 3, 20, 20)
    data_batch = {'inputs': inputs, 'target': target}

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
    assert torch.equal(outputs['results']['inputs'], data_batch['inputs'])
    assert torch.equal(outputs['results']['target'], data_batch['target'])
    assert torch.is_tensor(outputs['results']['output'])
    assert outputs['results']['output'].size() == (1, 3, 20, 20)

    # test forward_test
    with torch.no_grad():
        restorer.val_step(data_batch)
        outputs = restorer(**data_batch, test_mode=True)
    assert torch.equal(outputs['inputs'], data_batch['inputs'])
    assert torch.is_tensor(outputs['output'])
    assert outputs['output'].size() == (1, 3, 20, 20)
    assert outputs['output'].max() <= 1.
    assert outputs['output'].min() >= 0.

    # test forward_dummy
    with torch.no_grad():
        output = restorer.forward_dummy(data_batch['inputs'])
    assert torch.is_tensor(output)
    assert output.size() == (1, 3, 20, 20)

    # test train_step
    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['log_vars'], dict)
    assert isinstance(outputs['log_vars']['loss_pix'], float)
    assert outputs['num_samples'] == 1
    assert torch.equal(outputs['results']['inputs'], data_batch['inputs'])
    assert torch.equal(outputs['results']['target'], data_batch['target'])
    assert torch.is_tensor(outputs['results']['output'])
    assert outputs['results']['output'].size() == (1, 3, 20, 20)

    # test train_step and forward_test (gpu)
    if torch.cuda.is_available():
        restorer = restorer.cuda()
        optimizer['generator'] = obj_from_dict(
            optim_cfg, torch.optim, dict(params=restorer.parameters()))
        data_batch = {'inputs': inputs.cuda(), 'target': target.cuda()}

        # test forward train
        outputs = restorer(**data_batch, test_mode=False)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['losses'], dict)
        assert isinstance(outputs['losses']['loss_pix'],
                          torch.cuda.FloatTensor)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['inputs'],
                           data_batch['inputs'].cpu())
        assert torch.equal(outputs['results']['target'],
                           data_batch['target'].cpu())
        assert torch.is_tensor(outputs['results']['output'])
        assert outputs['results']['output'].size() == (1, 3, 20, 20)

        # forward_test
        with torch.no_grad():
            restorer.val_step(data_batch)
            outputs = restorer(**data_batch, test_mode=True)
        assert torch.equal(outputs['inputs'], data_batch['inputs'].cpu())
        assert torch.is_tensor(outputs['output'])
        assert outputs['output'].size() == (1, 3, 20, 20)
        assert outputs['output'].max() <= 1.
        assert outputs['output'].min() >= 0.

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
        assert outputs['results']['output'].size() == (1, 3, 20, 20)

    # test with metric and save image
    test_cfg = dict(metrics=('PSNR', 'SSIM'), crop_border=0)
    test_cfg = mmcv.Config(test_cfg)

    data_batch = {
        'inputs': inputs,
        'target': target,
        'meta': [{
            'key': '000001/0000',
            'target_path': 'fake_path/fake_name.png'
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

        outputs = restorer(
            inputs=inputs,
            target=target,
            meta=[{
                'key':
                '000001/0000',
                'inputs_path':
                ['fake_path/fake_name.png', 'fake_path/fake_name.png']
            }],
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

        # test forward_test when output.shape==5
        model_cfg = dict(
            type='BasicInterpolator',
            generator=dict(type='InterpolateExample2'),
            pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
        train_cfg = None
        test_cfg = None
        restorer = build_model(
            model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
        outputs = restorer(
            inputs=inputs,
            target=target.unsqueeze(1),
            meta=[{
                'key':
                '000001/0000',
                'inputs_path':
                ['fake_path/fake_name.png', 'fake_path/fake_name.png']
            }],
            test_mode=True,
            save_image=True,
            save_path=tmpdir,
            iteration=100)
        outputs = restorer(
            inputs=inputs,
            target=target.unsqueeze(1),
            meta=[{
                'key':
                '000001/0000',
                'inputs_path':
                ['fake_path/fake_name.png', 'fake_path/fake_name.png']
            }],
            test_mode=True,
            save_image=True,
            save_path=tmpdir,
            iteration=None)

        with pytest.raises(ValueError):
            # iteration should be number or None
            restorer(
                **data_batch,
                test_mode=True,
                save_image=True,
                save_path=tmpdir,
                iteration='100')

    # test merge_frames
    input_tensors = torch.rand(2, 2, 3, 256, 256)
    output_tensors = torch.rand(2, 1, 3, 256, 256)
    result = restorer.merge_frames(input_tensors, output_tensors)
    assert isinstance(result, list)
    assert len(result) == 5
    assert result[0].shape == (256, 256, 3)

    # test split_frames
    tensors = torch.rand(1, 10, 3, 256, 256)
    result = restorer.split_frames(tensors)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (9, 2, 3, 256, 256)

    # test evaluate 5d output
    test_cfg = dict(metrics=('PSNR', 'SSIM'), crop_border=0)
    test_cfg = mmcv.Config(test_cfg)
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    output = torch.rand(1, 2, 3, 256, 256)
    target = torch.rand(1, 2, 3, 256, 256)
    restorer.evaluate(output, target)

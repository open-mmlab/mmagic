import numpy as np
import pytest
import torch
from mmcv.runner import obj_from_dict
from mmcv.utils.config import Config

from mmedit.models.builder import build_model


def test_dic_model():

    model_cfg = dict(
        type='DIC',
        generator=dict(
            type='DICNet',
            in_channels=3,
            out_channels=3,
            mid_channels=48,
            num_blocks=6,
            hg_mid_channels=256,
            hg_num_keypoints=68,
            num_steps=4,
            upscale_factor=8,
            detach_attention=False),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        align_loss=dict(type='MSELoss', loss_weight=0.1, reduction='mean'))

    scale = 8
    train_cfg = None
    test_cfg = Config(dict(metrics=['PSNR', 'SSIM'], crop_border=scale))

    # build restorer
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
    optimizer = dict(
        generator=obj_from_dict(optim_cfg, torch.optim,
                                dict(params=restorer.parameters())))

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
        optimizer = dict(
            generator=obj_from_dict(optim_cfg, torch.optim,
                                    dict(params=restorer.parameters())))
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

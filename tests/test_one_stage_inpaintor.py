import copy
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from mmcv import Config
from mmedit.models import build_model
from mmedit.models.backbones import GLEncoderDecoder


def test_one_stage_inpaintor():
    cfg = Config.fromfile(
        Path(__file__).parent.joinpath(
            'data/inpaintor_config/one_stage_gl.py'))

    # mock perceptual loss for test speed
    cfg.model.loss_composed_percep = None
    inpaintor = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # modify attributes for mocking
    inpaintor.with_composed_percep_loss = True
    inpaintor.loss_percep = None

    # test attributes
    assert inpaintor.__class__.__name__ == 'OneStageInpaintor'
    assert isinstance(inpaintor.generator, GLEncoderDecoder)
    assert inpaintor.with_l1_hole_loss
    assert inpaintor.with_l1_valid_loss
    assert inpaintor.with_tv_loss
    assert inpaintor.with_composed_percep_loss
    assert inpaintor.with_out_percep_loss
    assert inpaintor.with_gan
    assert inpaintor.with_gp_loss
    assert inpaintor.with_disc_shift_loss
    assert inpaintor.is_train
    assert inpaintor.train_cfg['disc_step'] == 1
    assert inpaintor.test_cfg['test'] == 1
    assert inpaintor.disc_step_count == 0

    with patch.object(
            inpaintor, 'loss_percep', return_value=(torch.tensor(1.0), None)):

        input_x = torch.randn(1, 3, 256, 256)
        with pytest.raises(NotImplementedError):
            inpaintor.forward_train(input_x)

        if torch.cuda.is_available():
            gt_img = torch.randn(1, 3, 256, 256).cuda()
            mask = torch.zeros_like(gt_img)[:, 0:1, ...]
            mask[..., 20:100, 100:120] = 1.
            masked_img = gt_img * (1. - mask)
            inpaintor.cuda()
            data_batch = dict(gt_img=gt_img, mask=mask, masked_img=masked_img)
            output = inpaintor.forward_test(**data_batch)
            assert output['fake_res'].shape == (1, 3, 256, 256)
            assert output['fake_img'].shape == (1, 3, 256, 256)

            output = inpaintor.val_step(data_batch)
            assert output['fake_res'].shape == (1, 3, 256, 256)
            assert output['fake_img'].shape == (1, 3, 256, 256)

            optim_g = torch.optim.SGD(inpaintor.generator.parameters(), lr=0.1)
            optim_d = torch.optim.SGD(inpaintor.disc.parameters(), lr=0.1)
            optim_dict = dict(generator=optim_g, disc=optim_d)

            outputs = inpaintor.train_step(data_batch, optim_dict)
            assert outputs['num_samples'] == 1
            results = outputs['results']
            assert results['fake_res'].shape == (1, 3, 256, 256)
            assert 'loss_l1_hole' in outputs['log_vars']
            assert 'loss_l1_valid' in outputs['log_vars']
            assert 'loss_composed_percep' in outputs['log_vars']
            assert 'loss_composed_style' not in outputs['log_vars']
            assert 'loss_out_percep' in outputs['log_vars']
            assert 'loss_out_style' not in outputs['log_vars']
            assert 'loss_tv' in outputs['log_vars']
            assert 'fake_loss' in outputs['log_vars']
            assert 'real_loss' in outputs['log_vars']
            assert 'loss_g_fake' in outputs['log_vars']

            cfg_ = copy.deepcopy(cfg)
            cfg_.train_cfg.disc_step = 2
            inpaintor = build_model(
                cfg_.model, train_cfg=cfg_.train_cfg, test_cfg=cfg_.test_cfg)
            inpaintor.cuda()
            assert inpaintor.train_cfg.disc_step == 2
            outputs = inpaintor.train_step(data_batch, optim_dict)
            assert 'loss_l1_hole' not in outputs['log_vars']

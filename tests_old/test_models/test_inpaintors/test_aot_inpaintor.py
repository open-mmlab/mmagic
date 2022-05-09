# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import tempfile

import pytest
import torch
from mmcv import Config

from mmedit.models import build_model
from mmedit.models.backbones.encoder_decoders import AOTEncoderDecoder


def test_aot_inpaintor():
    cfg = Config.fromfile('tests/data/inpaintor_config/aot_test.py')

    inpaintor = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # test attributes
    assert inpaintor.__class__.__name__ == 'AOTInpaintor'
    assert isinstance(inpaintor.generator, AOTEncoderDecoder)
    assert inpaintor.with_l1_valid_loss
    assert inpaintor.with_composed_percep_loss
    assert inpaintor.with_out_percep_loss
    assert inpaintor.with_gan
    assert inpaintor.is_train
    assert inpaintor.train_cfg['disc_step'] == 1
    assert inpaintor.disc_step_count == 0

    input_x = torch.randn(1, 3, 256, 256)
    with pytest.raises(NotImplementedError):
        inpaintor.forward_train(input_x)

    gt_img = torch.randn(1, 3, 256, 256)
    mask = torch.zeros_like(gt_img)[:, 0:1, ...]
    mask[..., 20:100, 100:120] = 1.
    masked_img = gt_img * (1. - mask) + mask
    inpaintor
    data_batch = dict(gt_img=gt_img, mask=mask, masked_img=masked_img)
    output = inpaintor.forward_test(**data_batch)
    assert 'eval_result' in output

    output = inpaintor.val_step(data_batch)
    assert 'eval_result' in output

    optim_g = torch.optim.SGD(inpaintor.generator.parameters(), lr=0.1)
    optim_d = torch.optim.SGD(inpaintor.disc.parameters(), lr=0.1)
    optim_dict = dict(generator=optim_g, disc=optim_d)

    outputs = inpaintor.train_step(data_batch, optim_dict)
    assert outputs['num_samples'] == 1
    results = outputs['results']
    assert results['fake_res'].shape == (1, 3, 256, 256)
    assert 'loss_l1_valid' in outputs['log_vars']
    assert 'loss_out_percep' in outputs['log_vars']
    assert 'disc_losses' in outputs['log_vars']
    assert 'loss_g_fake' in outputs['log_vars']

    # test forward test w/o save image
    outputs = inpaintor.forward_test(
        masked_img[0:1], mask[0:1], gt_img=gt_img[0:1, ...])
    assert 'eval_result' in outputs
    assert outputs['eval_result']['l1'] > 0
    assert outputs['eval_result']['psnr'] > 0
    assert outputs['eval_result']['ssim'] > 0

    # test forward test w/o eval metrics
    inpaintor.test_cfg = dict()
    inpaintor.eval_with_metrics = False
    outputs = inpaintor.forward_test(masked_img[0:1], mask[0:1])
    for key in ['fake_res', 'fake_img']:
        assert outputs[key].size() == (1, 3, 256, 256)

    # test forward test w/ save image
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = inpaintor.forward_test(
            masked_img[0:1],
            mask[0:1],
            save_image=True,
            save_path=tmpdir,
            iteration=4396,
            meta=[dict(gt_img_path='igccc.png')])

        assert os.path.exists(os.path.join(tmpdir, 'igccc_4396.png'))

    # test forward test w/ save image w/ gt_img
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = inpaintor.forward_test(
            masked_img[0:1],
            mask[0:1],
            save_image=True,
            save_path=tmpdir,
            meta=[dict(gt_img_path='igccc.png')],
            gt_img=gt_img[0:1, ...])

        assert os.path.exists(os.path.join(tmpdir, 'igccc.png'))

        with pytest.raises(AssertionError):
            outputs = inpaintor.forward_test(
                masked_img[0:1],
                mask[0:1],
                save_image=True,
                save_path=tmpdir,
                iteration=4396,
                gt_img=gt_img[0:1, ...])
        with pytest.raises(AssertionError):
            outputs = inpaintor.forward_test(
                masked_img[0:1],
                mask[0:1],
                save_image=True,
                save_path=None,
                iteration=4396,
                meta=[dict(gt_img_path='igccc.png')],
                gt_img=gt_img[0:1, ...])

    cfg_ = copy.deepcopy(cfg)
    cfg_.train_cfg.disc_step = 2
    inpaintor = build_model(
        cfg_.model, train_cfg=cfg_.train_cfg, test_cfg=cfg_.test_cfg)
    assert inpaintor.train_cfg.disc_step == 2
    outputs = inpaintor.train_step(data_batch, optim_dict)
    assert 'loss_l1_hole' not in outputs['log_vars']

    # Test on GPU
    if torch.cuda.is_available():
        gt_img = torch.randn(1, 3, 256, 256).cuda()
        mask = torch.zeros_like(gt_img)[:, 0:1, ...]
        mask[..., 20:100, 100:120] = 1.
        masked_img = gt_img * (1. - mask) + mask
        inpaintor.cuda()
        data_batch = dict(gt_img=gt_img, mask=mask, masked_img=masked_img)
        output = inpaintor.forward_test(**data_batch)
        assert 'eval_result' in output

        output = inpaintor.val_step(data_batch)
        assert 'eval_result' in output

        optim_g = torch.optim.SGD(inpaintor.generator.parameters(), lr=0.1)
        optim_d = torch.optim.SGD(inpaintor.disc.parameters(), lr=0.1)
        optim_dict = dict(generator=optim_g, disc=optim_d)

        outputs = inpaintor.train_step(data_batch, optim_dict)
        assert outputs['num_samples'] == 1
        results = outputs['results']
        assert results['fake_res'].shape == (1, 3, 256, 256)
        assert 'loss_l1_valid' in outputs['log_vars']
        assert 'loss_out_percep' in outputs['log_vars']
        assert 'disc_losses' in outputs['log_vars']
        assert 'loss_g_fake' in outputs['log_vars']

        # test forward test w/o save image
        outputs = inpaintor.forward_test(
            masked_img[0:1], mask[0:1], gt_img=gt_img[0:1, ...])
        assert 'eval_result' in outputs
        assert outputs['eval_result']['l1'] > 0
        assert outputs['eval_result']['psnr'] > 0
        assert outputs['eval_result']['ssim'] > 0

        # test forward test w/o eval metrics
        inpaintor.test_cfg = dict()
        inpaintor.eval_with_metrics = False
        outputs = inpaintor.forward_test(masked_img[0:1], mask[0:1])
        for key in ['fake_res', 'fake_img']:
            assert outputs[key].size() == (1, 3, 256, 256)

        # test forward test w/ save image
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = inpaintor.forward_test(
                masked_img[0:1],
                mask[0:1],
                save_image=True,
                save_path=tmpdir,
                iteration=4396,
                meta=[dict(gt_img_path='igccc.png')])

            assert os.path.exists(os.path.join(tmpdir, 'igccc_4396.png'))

        # test forward test w/ save image w/ gt_img
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = inpaintor.forward_test(
                masked_img[0:1],
                mask[0:1],
                save_image=True,
                save_path=tmpdir,
                meta=[dict(gt_img_path='igccc.png')],
                gt_img=gt_img[0:1, ...])

            assert os.path.exists(os.path.join(tmpdir, 'igccc.png'))

            with pytest.raises(AssertionError):
                outputs = inpaintor.forward_test(
                    masked_img[0:1],
                    mask[0:1],
                    save_image=True,
                    save_path=tmpdir,
                    iteration=4396,
                    gt_img=gt_img[0:1, ...])
            with pytest.raises(AssertionError):
                outputs = inpaintor.forward_test(
                    masked_img[0:1],
                    mask[0:1],
                    save_image=True,
                    save_path=None,
                    iteration=4396,
                    meta=[dict(gt_img_path='igccc.png')],
                    gt_img=gt_img[0:1, ...])

        cfg_ = copy.deepcopy(cfg)
        cfg_.train_cfg.disc_step = 2
        inpaintor = build_model(
            cfg_.model, train_cfg=cfg_.train_cfg, test_cfg=cfg_.test_cfg)
        inpaintor.cuda()
        assert inpaintor.train_cfg.disc_step == 2
        outputs = inpaintor.train_step(data_batch, optim_dict)
        assert 'loss_l1_hole' not in outputs['log_vars']

# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest.mock import patch

import mmcv
import pytest
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import obj_from_dict

from mmedit.models import build_model
from mmedit.models.backbones import ResnetGenerator
from mmedit.models.components import PatchDiscriminator
from mmedit.models.losses import GANLoss, L1Loss


def test_cyclegan():

    model_cfg = dict(
        type='CycleGAN',
        generator=dict(
            type='ResnetGenerator',
            in_channels=3,
            out_channels=3,
            base_channels=64,
            norm_cfg=dict(type='IN'),
            use_dropout=False,
            num_blocks=9,
            padding_mode='reflect',
            init_cfg=dict(type='normal', gain=0.02)),
        discriminator=dict(
            type='PatchDiscriminator',
            in_channels=3,
            base_channels=64,
            num_conv=3,
            norm_cfg=dict(type='IN'),
            init_cfg=dict(type='normal', gain=0.02)),
        gan_loss=dict(
            type='GANLoss',
            gan_type='lsgan',
            real_label_val=1.0,
            fake_label_val=0,
            loss_weight=1.0),
        cycle_loss=dict(type='L1Loss', loss_weight=10.0, reduction='mean'),
        id_loss=dict(type='L1Loss', loss_weight=0.5, reduction='mean'))

    train_cfg = None
    test_cfg = None

    # build synthesizer
    synthesizer = build_model(
        model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test checking if id loss > 0, in_channels == out_channels
    with pytest.raises(AssertionError):
        bad_model_cfg = copy.deepcopy(model_cfg)
        bad_model_cfg['generator']['out_channels'] = 1
        _ = build_model(bad_model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test checking gan loss cannot be None
    with pytest.raises(AssertionError):
        bad_model_cfg = copy.deepcopy(model_cfg)
        bad_model_cfg['gan_loss'] = None
        _ = build_model(bad_model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test checking cycle loss cannot be None
    with pytest.raises(AssertionError):
        bad_model_cfg = copy.deepcopy(model_cfg)
        bad_model_cfg['cycle_loss'] = None
        _ = build_model(bad_model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test attributes
    assert synthesizer.__class__.__name__ == 'CycleGAN'
    assert isinstance(synthesizer.generators['a'], ResnetGenerator)
    assert isinstance(synthesizer.generators['b'], ResnetGenerator)
    assert isinstance(synthesizer.discriminators['a'], PatchDiscriminator)
    assert isinstance(synthesizer.discriminators['b'], PatchDiscriminator)
    assert isinstance(synthesizer.gan_loss, GANLoss)
    assert isinstance(synthesizer.cycle_loss, L1Loss)
    assert isinstance(synthesizer.id_loss, L1Loss)
    assert synthesizer.train_cfg is None
    assert synthesizer.test_cfg is None

    # prepare data
    inputs = torch.rand(1, 3, 64, 64)
    targets = torch.rand(1, 3, 64, 64)
    data_batch = {'img_a': inputs, 'img_b': targets}
    img_meta = {}
    img_meta['img_a_path'] = 'img_a_path'
    img_meta['img_b_path'] = 'img_b_path'
    data_batch['meta'] = [img_meta]

    # prepare optimizer
    optim_cfg = dict(type='Adam', lr=2e-4, betas=(0.5, 0.999))
    optimizer = {
        'generators':
        obj_from_dict(
            optim_cfg, torch.optim,
            dict(params=getattr(synthesizer, 'generators').parameters())),
        'discriminators':
        obj_from_dict(
            optim_cfg, torch.optim,
            dict(params=getattr(synthesizer, 'discriminators').parameters()))
    }

    # test forward_dummy
    with torch.no_grad():
        output = synthesizer.forward_dummy(data_batch['img_a'])
    assert torch.is_tensor(output)
    assert output.size() == (1, 3, 64, 64)

    # test forward_test
    with torch.no_grad():
        outputs = synthesizer(inputs, targets, [img_meta], test_mode=True)
    assert torch.equal(outputs['real_a'], data_batch['img_a'])
    assert torch.equal(outputs['real_b'], data_batch['img_b'])
    assert torch.is_tensor(outputs['fake_b'])
    assert torch.is_tensor(outputs['fake_a'])
    assert outputs['fake_b'].size() == (1, 3, 64, 64)
    assert outputs['fake_a'].size() == (1, 3, 64, 64)

    # val_step
    with torch.no_grad():
        outputs = synthesizer.val_step(data_batch)
    assert torch.equal(outputs['real_a'], data_batch['img_a'])
    assert torch.equal(outputs['real_b'], data_batch['img_b'])
    assert torch.is_tensor(outputs['fake_b'])
    assert torch.is_tensor(outputs['fake_a'])
    assert outputs['fake_b'].size() == (1, 3, 64, 64)
    assert outputs['fake_a'].size() == (1, 3, 64, 64)

    # test forward_train
    outputs = synthesizer(inputs, targets, [img_meta], test_mode=False)
    assert torch.equal(outputs['real_a'], data_batch['img_a'])
    assert torch.equal(outputs['real_b'], data_batch['img_b'])
    assert torch.is_tensor(outputs['fake_b'])
    assert torch.is_tensor(outputs['fake_a'])
    assert torch.is_tensor(outputs['rec_a'])
    assert torch.is_tensor(outputs['rec_b'])
    assert outputs['fake_b'].size() == (1, 3, 64, 64)
    assert outputs['fake_a'].size() == (1, 3, 64, 64)
    assert outputs['rec_a'].size() == (1, 3, 64, 64)
    assert outputs['rec_b'].size() == (1, 3, 64, 64)

    # test train_step
    outputs = synthesizer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['log_vars'], dict)
    assert isinstance(outputs['results'], dict)
    for v in [
            'loss_gan_d_a', 'loss_gan_d_b', 'loss_id_a', 'loss_id_b',
            'loss_gan_g_a', 'loss_gan_g_b', 'loss_cycle_a', 'loss_cycle_b'
    ]:
        assert isinstance(outputs['log_vars'][v], float)
    assert outputs['num_samples'] == 1
    assert torch.equal(outputs['results']['real_a'], data_batch['img_a'])
    assert torch.equal(outputs['results']['real_b'], data_batch['img_b'])
    assert torch.is_tensor(outputs['results']['fake_b'])
    assert torch.is_tensor(outputs['results']['fake_a'])
    assert outputs['results']['fake_b'].size() == (1, 3, 64, 64)
    assert outputs['results']['fake_a'].size() == (1, 3, 64, 64)

    # test train_step and forward_test (gpu)
    if torch.cuda.is_available():
        synthesizer = synthesizer.cuda()
        optimizer = {
            'generators':
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(params=getattr(synthesizer, 'generators').parameters())),
            'discriminators':
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(
                    params=getattr(synthesizer,
                                   'discriminators').parameters()))
        }
        data_batch_cuda = copy.deepcopy(data_batch)
        data_batch_cuda['img_a'] = inputs.cuda()
        data_batch_cuda['img_b'] = targets.cuda()
        data_batch_cuda['meta'] = [DC(img_meta, cpu_only=True).data]

        # forward_test
        with torch.no_grad():
            outputs = synthesizer(
                data_batch_cuda['img_a'],
                data_batch_cuda['img_b'],
                data_batch_cuda['meta'],
                test_mode=True)
        assert torch.equal(outputs['real_a'], data_batch_cuda['img_a'].cpu())
        assert torch.equal(outputs['real_b'], data_batch_cuda['img_b'].cpu())
        assert torch.is_tensor(outputs['fake_b'])
        assert torch.is_tensor(outputs['fake_a'])
        assert outputs['fake_b'].size() == (1, 3, 64, 64)
        assert outputs['fake_a'].size() == (1, 3, 64, 64)

        # val_step
        with torch.no_grad():
            outputs = synthesizer.val_step(data_batch_cuda)
        assert torch.equal(outputs['real_a'], data_batch_cuda['img_a'].cpu())
        assert torch.equal(outputs['real_b'], data_batch_cuda['img_b'].cpu())
        assert torch.is_tensor(outputs['fake_b'])
        assert torch.is_tensor(outputs['fake_a'])
        assert outputs['fake_b'].size() == (1, 3, 64, 64)
        assert outputs['fake_a'].size() == (1, 3, 64, 64)

        # test forward_train
        outputs = synthesizer(
            data_batch_cuda['img_a'],
            data_batch_cuda['img_b'],
            data_batch_cuda['meta'],
            test_mode=False)
        assert torch.equal(outputs['real_a'], data_batch_cuda['img_a'])
        assert torch.equal(outputs['real_b'], data_batch_cuda['img_b'])
        assert torch.is_tensor(outputs['fake_b'])
        assert torch.is_tensor(outputs['fake_a'])
        assert torch.is_tensor(outputs['rec_a'])
        assert torch.is_tensor(outputs['rec_b'])
        assert outputs['fake_b'].size() == (1, 3, 64, 64)
        assert outputs['fake_a'].size() == (1, 3, 64, 64)
        assert outputs['rec_a'].size() == (1, 3, 64, 64)
        assert outputs['rec_b'].size() == (1, 3, 64, 64)

        # train_step
        outputs = synthesizer.train_step(data_batch_cuda, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        assert isinstance(outputs['results'], dict)
        for v in [
                'loss_gan_d_a', 'loss_gan_d_b', 'loss_id_a', 'loss_id_b',
                'loss_gan_g_a', 'loss_gan_g_b', 'loss_cycle_a', 'loss_cycle_b'
        ]:
            assert isinstance(outputs['log_vars'][v], float)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['real_a'],
                           data_batch_cuda['img_a'].cpu())
        assert torch.equal(outputs['results']['real_b'],
                           data_batch_cuda['img_b'].cpu())
        assert torch.is_tensor(outputs['results']['fake_b'])
        assert torch.is_tensor(outputs['results']['fake_a'])
        assert outputs['results']['fake_b'].size() == (1, 3, 64, 64)
        assert outputs['results']['fake_a'].size() == (1, 3, 64, 64)

    # test disc_steps and disc_init_steps
    data_batch['img_a'] = inputs.cpu()
    data_batch['img_b'] = targets.cpu()
    train_cfg = dict(disc_steps=2, disc_init_steps=2)
    synthesizer = build_model(
        model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    optimizer = {
        'generators':
        obj_from_dict(
            optim_cfg, torch.optim,
            dict(params=getattr(synthesizer, 'generators').parameters())),
        'discriminators':
        obj_from_dict(
            optim_cfg, torch.optim,
            dict(params=getattr(synthesizer, 'discriminators').parameters()))
    }

    # iter 0, 1
    for i in range(2):
        assert synthesizer.step_counter == i
        outputs = synthesizer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        assert isinstance(outputs['results'], dict)
        for v in [
                'loss_id_a', 'loss_id_b', 'loss_gan_g_a', 'loss_gan_g_b',
                'loss_cycle_a', 'loss_cycle_b'
        ]:
            assert outputs['log_vars'].get(v) is None
        assert isinstance(outputs['log_vars']['loss_gan_d_a'], float)
        assert isinstance(outputs['log_vars']['loss_gan_d_b'], float)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['real_a'], data_batch['img_a'])
        assert torch.equal(outputs['results']['real_b'], data_batch['img_b'])
        assert torch.is_tensor(outputs['results']['fake_b'])
        assert torch.is_tensor(outputs['results']['fake_a'])
        assert outputs['results']['fake_b'].size() == (1, 3, 64, 64)
        assert outputs['results']['fake_a'].size() == (1, 3, 64, 64)
        assert synthesizer.step_counter == i + 1

    # iter 2, 3, 4, 5
    for i in range(2, 6):
        assert synthesizer.step_counter == i
        outputs = synthesizer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        assert isinstance(outputs['results'], dict)
        log_check_list = [
            'loss_gan_d_a', 'loss_gan_d_b', 'loss_id_a', 'loss_id_b',
            'loss_gan_g_a', 'loss_gan_g_b', 'loss_cycle_a', 'loss_cycle_b'
        ]
        if i % 2 == 1:
            log_None_list = [
                'loss_id_a', 'loss_id_b', 'loss_gan_g_a', 'loss_gan_g_b',
                'loss_cycle_a', 'loss_cycle_b'
            ]
            for v in log_None_list:
                assert outputs['log_vars'].get(v) is None
                log_check_list.remove(v)
        for v in log_check_list:
            assert isinstance(outputs['log_vars'][v], float)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['real_a'], data_batch['img_a'])
        assert torch.equal(outputs['results']['real_b'], data_batch['img_b'])
        assert torch.is_tensor(outputs['results']['fake_b'])
        assert torch.is_tensor(outputs['results']['fake_a'])
        assert outputs['results']['fake_b'].size() == (1, 3, 64, 64)
        assert outputs['results']['fake_a'].size() == (1, 3, 64, 64)
        assert synthesizer.step_counter == i + 1

    # test without id loss
    model_cfg_ = copy.deepcopy(model_cfg)
    model_cfg_.pop('id_loss')
    synthesizer = build_model(model_cfg_, train_cfg=None, test_cfg=None)
    optimizer = {
        'generators':
        obj_from_dict(
            optim_cfg, torch.optim,
            dict(params=getattr(synthesizer, 'generators').parameters())),
        'discriminators':
        obj_from_dict(
            optim_cfg, torch.optim,
            dict(params=getattr(synthesizer, 'discriminators').parameters()))
    }
    data_batch['img_a'] = inputs.cpu()
    data_batch['img_b'] = targets.cpu()
    outputs = synthesizer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['log_vars'], dict)
    assert isinstance(outputs['results'], dict)
    assert outputs['log_vars'].get('loss_id_a') is None
    assert outputs['log_vars'].get('loss_id_b') is None
    log_check_list = [
        'loss_gan_d_a', 'loss_gan_d_b', 'loss_gan_g_a', 'loss_gan_g_b',
        'loss_cycle_a', 'loss_cycle_b'
    ]
    for v in log_check_list:
        assert isinstance(outputs['log_vars'][v], float)
    assert outputs['num_samples'] == 1
    assert torch.equal(outputs['results']['real_a'], data_batch['img_a'])
    assert torch.equal(outputs['results']['real_b'], data_batch['img_b'])
    assert torch.is_tensor(outputs['results']['fake_b'])
    assert torch.is_tensor(outputs['results']['fake_a'])
    assert outputs['results']['fake_b'].size() == (1, 3, 64, 64)
    assert outputs['results']['fake_a'].size() == (1, 3, 64, 64)

    # test b2a translation
    data_batch['img_a'] = inputs.cpu()
    data_batch['img_b'] = targets.cpu()
    train_cfg = dict(direction='b2a')
    synthesizer = build_model(
        model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    optimizer = {
        'generators':
        obj_from_dict(
            optim_cfg, torch.optim,
            dict(params=getattr(synthesizer, 'generators').parameters())),
        'discriminators':
        obj_from_dict(
            optim_cfg, torch.optim,
            dict(params=getattr(synthesizer, 'discriminators').parameters()))
    }
    assert synthesizer.step_counter == 0
    outputs = synthesizer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['log_vars'], dict)
    assert isinstance(outputs['results'], dict)
    for v in [
            'loss_gan_d_a', 'loss_gan_d_b', 'loss_id_a', 'loss_id_b',
            'loss_gan_g_a', 'loss_gan_g_b', 'loss_cycle_a', 'loss_cycle_b'
    ]:
        assert isinstance(outputs['log_vars'][v], float)
    assert outputs['num_samples'] == 1
    assert torch.equal(outputs['results']['real_a'], data_batch['img_b'])
    assert torch.equal(outputs['results']['real_b'], data_batch['img_a'])
    assert torch.is_tensor(outputs['results']['fake_b'])
    assert torch.is_tensor(outputs['results']['fake_a'])
    assert outputs['results']['fake_b'].size() == (1, 3, 64, 64)
    assert outputs['results']['fake_a'].size() == (1, 3, 64, 64)
    assert synthesizer.step_counter == 1

    # test GAN image buffer size = 0
    data_batch['img_a'] = inputs.cpu()
    data_batch['img_b'] = targets.cpu()
    train_cfg = dict(buffer_size=0)
    synthesizer = build_model(
        model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    optimizer = {
        'generators':
        obj_from_dict(
            optim_cfg, torch.optim,
            dict(params=getattr(synthesizer, 'generators').parameters())),
        'discriminators':
        obj_from_dict(
            optim_cfg, torch.optim,
            dict(params=getattr(synthesizer, 'discriminators').parameters()))
    }
    assert synthesizer.step_counter == 0
    outputs = synthesizer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['log_vars'], dict)
    assert isinstance(outputs['results'], dict)
    for v in [
            'loss_gan_d_a', 'loss_gan_d_b', 'loss_id_a', 'loss_id_b',
            'loss_gan_g_a', 'loss_gan_g_b', 'loss_cycle_a', 'loss_cycle_b'
    ]:
        assert isinstance(outputs['log_vars'][v], float)
    assert outputs['num_samples'] == 1
    assert torch.equal(outputs['results']['real_a'], data_batch['img_a'])
    assert torch.equal(outputs['results']['real_b'], data_batch['img_b'])
    assert torch.is_tensor(outputs['results']['fake_b'])
    assert torch.is_tensor(outputs['results']['fake_a'])
    assert outputs['results']['fake_b'].size() == (1, 3, 64, 64)
    assert outputs['results']['fake_a'].size() == (1, 3, 64, 64)
    assert synthesizer.step_counter == 1

    # test save image
    # show input
    train_cfg = None
    test_cfg = dict(show_input=True)
    synthesizer = build_model(
        model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    with patch.object(mmcv, 'imwrite', return_value=True):
        # test save path not None Assertion
        with pytest.raises(AssertionError):
            with torch.no_grad():
                _ = synthesizer(
                    inputs,
                    targets, [img_meta],
                    test_mode=True,
                    save_image=True)
        # iteration is None
        with torch.no_grad():
            outputs = synthesizer(
                inputs,
                targets, [img_meta],
                test_mode=True,
                save_image=True,
                save_path='save_path')
        assert torch.equal(outputs['real_a'], data_batch['img_a'])
        assert torch.equal(outputs['real_b'], data_batch['img_b'])
        assert torch.is_tensor(outputs['fake_b'])
        assert torch.is_tensor(outputs['fake_a'])
        assert outputs['fake_b'].size() == (1, 3, 64, 64)
        assert outputs['fake_a'].size() == (1, 3, 64, 64)
        assert outputs['saved_flag']
        # iteration is not None
        with torch.no_grad():
            outputs = synthesizer(
                inputs,
                targets, [img_meta],
                test_mode=True,
                save_image=True,
                save_path='save_path',
                iteration=1000)
        assert torch.equal(outputs['real_a'], data_batch['img_a'])
        assert torch.equal(outputs['real_b'], data_batch['img_b'])
        assert torch.is_tensor(outputs['fake_b'])
        assert torch.is_tensor(outputs['fake_a'])
        assert outputs['fake_b'].size() == (1, 3, 64, 64)
        assert outputs['fake_a'].size() == (1, 3, 64, 64)
        assert outputs['saved_flag']

    # not show input, test_direction a2b
    train_cfg = None
    test_cfg = dict(show_input=False, test_direction='a2b')
    synthesizer = build_model(
        model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    with patch.object(mmcv, 'imwrite', return_value=True):
        # test save path not None Assertion
        with pytest.raises(AssertionError):
            with torch.no_grad():
                _ = synthesizer(
                    inputs,
                    targets, [img_meta],
                    test_mode=True,
                    save_image=True)
        # iteration is None
        with torch.no_grad():
            outputs = synthesizer(
                inputs,
                targets, [img_meta],
                test_mode=True,
                save_image=True,
                save_path='save_path')
        assert torch.equal(outputs['real_a'], data_batch['img_a'])
        assert torch.equal(outputs['real_b'], data_batch['img_b'])
        assert torch.is_tensor(outputs['fake_b'])
        assert torch.is_tensor(outputs['fake_a'])
        assert outputs['fake_b'].size() == (1, 3, 64, 64)
        assert outputs['fake_a'].size() == (1, 3, 64, 64)
        assert outputs['saved_flag']
        # iteration is not None
        with torch.no_grad():
            outputs = synthesizer(
                inputs,
                targets, [img_meta],
                test_mode=True,
                save_image=True,
                save_path='save_path',
                iteration=1000)
        assert torch.equal(outputs['real_a'], data_batch['img_a'])
        assert torch.equal(outputs['real_b'], data_batch['img_b'])
        assert torch.is_tensor(outputs['fake_b'])
        assert torch.is_tensor(outputs['fake_a'])
        assert outputs['fake_b'].size() == (1, 3, 64, 64)
        assert outputs['fake_a'].size() == (1, 3, 64, 64)
        assert outputs['saved_flag']

    # not show input, test_direction b2a
    train_cfg = None
    test_cfg = dict(show_input=False, test_direction='b2a')
    synthesizer = build_model(
        model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    with patch.object(mmcv, 'imwrite', return_value=True):
        # test save path not None Assertion
        with pytest.raises(AssertionError):
            with torch.no_grad():
                _ = synthesizer(
                    inputs,
                    targets, [img_meta],
                    test_mode=True,
                    save_image=True)
        # iteration is None
        with torch.no_grad():
            outputs = synthesizer(
                inputs,
                targets, [img_meta],
                test_mode=True,
                save_image=True,
                save_path='save_path')
        assert torch.equal(outputs['real_a'], data_batch['img_a'])
        assert torch.equal(outputs['real_b'], data_batch['img_b'])
        assert torch.is_tensor(outputs['fake_b'])
        assert torch.is_tensor(outputs['fake_a'])
        assert outputs['fake_b'].size() == (1, 3, 64, 64)
        assert outputs['fake_a'].size() == (1, 3, 64, 64)
        assert outputs['saved_flag']
        # iteration is not None
        with torch.no_grad():
            outputs = synthesizer(
                inputs,
                targets, [img_meta],
                test_mode=True,
                save_image=True,
                save_path='save_path',
                iteration=1000)
        assert torch.equal(outputs['real_a'], data_batch['img_a'])
        assert torch.equal(outputs['real_b'], data_batch['img_b'])
        assert torch.is_tensor(outputs['fake_b'])
        assert torch.is_tensor(outputs['fake_a'])
        assert outputs['fake_b'].size() == (1, 3, 64, 64)
        assert outputs['fake_a'].size() == (1, 3, 64, 64)
        assert outputs['saved_flag']

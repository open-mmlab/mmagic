# Copyright (c) OpenMMLab. All rights reserved.
import copy
import sys

import torch
from mmengine import MessageHub
from mmengine.optim import OptimWrapper, OptimWrapperDict

from mmagic.models import CycleGAN, DataPreprocessor
from mmagic.models.archs import PatchDiscriminator
from mmagic.models.editors.cyclegan import ResnetGenerator
from mmagic.structures import DataSample


def obj_from_dict(info: dict, parent=None, default_args=None):
    """Initialize an object from dict.

    The dict must contain the key "type", which indicates the object type, it
    can be either a string or type, such as "list" or ``list``. Remaining
    fields are treated as the arguments for constructing the object.

    Args:
        info (dict): Object types and arguments.
        parent (:class:`module`): Module which may containing expected object
            classes.
        default_args (dict, optional): Default arguments for initializing the
            object.

    Returns:
        any type: Object built from the dict.
    """
    assert isinstance(info, dict) and 'type' in info
    assert isinstance(default_args, dict) or default_args is None
    args = info.copy()
    obj_type = args.pop('type')
    # if mmcv.is_str(obj_type):
    if isinstance(obj_type, str):
        if parent is not None:
            obj_type = getattr(parent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but '
                        f'got {type(obj_type)}')
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


def test_cyclegan():

    model_cfg = dict(
        default_domain='photo',
        reachable_domains=['photo', 'mask'],
        related_domains=['photo', 'mask'],
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
            init_cfg=dict(type='normal', gain=0.02)))

    train_settings = None

    # build synthesizer
    synthesizer = CycleGAN(**model_cfg, data_preprocessor=DataPreprocessor())

    # test attributes
    assert synthesizer.__class__.__name__ == 'CycleGAN'
    assert isinstance(synthesizer.generators['photo'], ResnetGenerator)
    assert isinstance(synthesizer.generators['mask'], ResnetGenerator)
    assert isinstance(synthesizer.discriminators['photo'], PatchDiscriminator)
    assert isinstance(synthesizer.discriminators['mask'], PatchDiscriminator)

    # prepare data
    inputs = torch.rand(1, 3, 64, 64)
    targets = torch.rand(1, 3, 64, 64)
    data_batch = {'img_mask': inputs, 'img_photo': targets}

    # prepare optimizer
    optim_cfg = dict(type='Adam', lr=2e-4, betas=(0.5, 0.999))
    optimizer = OptimWrapperDict(
        generators=OptimWrapper(
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(params=getattr(synthesizer, 'generators').parameters()))),
        discriminators=OptimWrapper(
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(
                    params=getattr(synthesizer,
                                   'discriminators').parameters()))))

    # test forward_test
    with torch.no_grad():
        outputs = synthesizer(inputs, target_domain='photo', test_mode=True)
    assert torch.equal(outputs['source'], data_batch['img_mask'])
    assert torch.is_tensor(outputs['target'])
    assert outputs['target'].size() == (1, 3, 64, 64)

    with torch.no_grad():
        outputs = synthesizer(targets, target_domain='mask', test_mode=True)
    assert torch.equal(outputs['source'], data_batch['img_photo'])
    assert torch.is_tensor(outputs['target'])
    assert outputs['target'].size() == (1, 3, 64, 64)

    # test forward_train
    with torch.no_grad():
        outputs = synthesizer(inputs, target_domain='photo', test_mode=True)
    assert torch.equal(outputs['source'], data_batch['img_mask'])
    assert torch.is_tensor(outputs['target'])
    assert outputs['target'].size() == (1, 3, 64, 64)

    with torch.no_grad():
        outputs = synthesizer(targets, target_domain='mask', test_mode=True)
    assert torch.equal(outputs['source'], data_batch['img_photo'])
    assert torch.is_tensor(outputs['target'])
    assert outputs['target'].size() == (1, 3, 64, 64)

    # test train_step
    message_hub = MessageHub.get_instance('cyclegan-test')
    message_hub.update_info('iter', 0)
    inputs = torch.rand(1, 3, 64, 64)
    targets = torch.rand(1, 3, 64, 64)
    data_batch = dict(inputs={'img_mask': inputs, 'img_photo': targets})
    log_vars = synthesizer.train_step(data_batch, optimizer)
    assert isinstance(log_vars, dict)
    for v in [
            'loss_gan_d_mask', 'loss_gan_d_photo', 'loss_gan_g_mask',
            'loss_gan_g_photo', 'cycle_loss', 'id_loss'
    ]:
        assert isinstance(log_vars[v].item(), float)

    # test train_step and forward_test (gpu)
    if torch.cuda.is_available():
        synthesizer = synthesizer.cuda()
        optimizer = OptimWrapperDict(
            generators=OptimWrapper(
                obj_from_dict(
                    optim_cfg, torch.optim,
                    dict(
                        params=getattr(synthesizer,
                                       'generators').parameters()))),
            discriminators=OptimWrapper(
                obj_from_dict(
                    optim_cfg, torch.optim,
                    dict(
                        params=getattr(synthesizer,
                                       'discriminators').parameters()))))

        inputs = torch.rand(1, 3, 64, 64)
        targets = torch.rand(1, 3, 64, 64)
        data_batch = {'img_mask': inputs, 'img_photo': targets}
        data_batch_cuda = copy.deepcopy(data_batch)
        data_batch_cuda['img_mask'] = inputs.cuda()
        data_batch_cuda['img_photo'] = targets.cuda()

        # forward_test
        with torch.no_grad():
            outputs = synthesizer(
                data_batch_cuda['img_mask'],
                target_domain='photo',
                test_mode=True)
        assert torch.equal(outputs['source'].cpu(),
                           data_batch_cuda['img_mask'].cpu())
        assert torch.is_tensor(outputs['target'])
        assert outputs['target'].size() == (1, 3, 64, 64)

        with torch.no_grad():
            outputs = synthesizer(
                data_batch_cuda['img_photo'],
                target_domain='mask',
                test_mode=True)
        assert torch.equal(outputs['source'].cpu(),
                           data_batch_cuda['img_photo'].cpu())
        assert torch.is_tensor(outputs['target'])
        assert outputs['target'].size() == (1, 3, 64, 64)

        # test forward_train
        with torch.no_grad():
            outputs = synthesizer(
                data_batch_cuda['img_mask'],
                target_domain='photo',
                test_mode=False)
        assert torch.equal(outputs['source'], data_batch_cuda['img_mask'])
        assert torch.is_tensor(outputs['target'])
        assert outputs['target'].size() == (1, 3, 64, 64)

        with torch.no_grad():
            outputs = synthesizer(
                data_batch_cuda['img_photo'],
                target_domain='mask',
                test_mode=False)
        assert torch.equal(outputs['source'], data_batch_cuda['img_photo'])
        assert torch.is_tensor(outputs['target'])
        assert outputs['target'].size() == (1, 3, 64, 64)

        # train_step
        inputs = torch.rand(1, 3, 64, 64).cuda()
        targets = torch.rand(1, 3, 64, 64).cuda()
        data_batch_cuda = dict(inputs={
            'img_mask': inputs,
            'img_photo': targets
        })

        log_vars = synthesizer.train_step(data_batch_cuda, optimizer)
        assert isinstance(log_vars, dict)
        for v in [
                'loss_gan_d_mask', 'loss_gan_d_photo', 'loss_gan_g_mask',
                'loss_gan_g_photo', 'cycle_loss', 'id_loss'
        ]:
            assert isinstance(log_vars[v].item(), float)

    # test disc_steps and disc_init_steps
    train_settings = dict(discriminator_steps=2, disc_init_steps=2)
    synthesizer = CycleGAN(
        **model_cfg, **train_settings, data_preprocessor=DataPreprocessor())
    optimizer = OptimWrapperDict(
        generators=OptimWrapper(
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(params=getattr(synthesizer, 'generators').parameters()))),
        discriminators=OptimWrapper(
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(
                    params=getattr(synthesizer,
                                   'discriminators').parameters()))))

    inputs = torch.rand(1, 3, 64, 64)
    targets = torch.rand(1, 3, 64, 64)
    data_batch = dict(inputs={'img_mask': inputs, 'img_photo': targets})
    # iter 0, 1
    for i in range(2):
        message_hub.update_info('iter', i)
        log_vars = synthesizer.train_step(data_batch, optimizer)
        assert isinstance(log_vars, dict)
        for v in [
                'loss_gan_g_mask', 'loss_gan_g_photo', 'cycle_loss', 'id_loss'
        ]:
            assert log_vars.get(v) is None
        assert isinstance(log_vars['loss_gan_d_mask'].item(), float)
        assert isinstance(log_vars['loss_gan_d_photo'].item(), float)

    # iter 2, 3, 4, 5
    for i in range(2, 6):
        message_hub.update_info('iter', i)
        log_vars = synthesizer.train_step(data_batch, optimizer)
        print(log_vars.keys())
        assert isinstance(log_vars, dict)
        log_check_list = [
            'loss_gan_d_mask', 'loss_gan_d_photo', 'loss_gan_g_mask',
            'loss_gan_g_photo', 'cycle_loss', 'id_loss'
        ]
        if (i + 1) % 2 == 1:
            log_None_list = [
                'loss_gan_g_mask', 'loss_gan_g_photo', 'cycle_loss', 'id_loss'
            ]
            for v in log_None_list:
                assert log_vars.get(v) is None
                log_check_list.remove(v)
        for v in log_check_list:
            assert isinstance(log_vars[v].item(), float)

    # test GAN image buffer size = 0
    inputs = torch.rand(1, 3, 64, 64)
    targets = torch.rand(1, 3, 64, 64)
    data_batch = dict(inputs={'img_mask': inputs, 'img_photo': targets})
    train_settings = dict(buffer_size=0)
    synthesizer = CycleGAN(
        **model_cfg, **train_settings, data_preprocessor=DataPreprocessor())
    optimizer = OptimWrapperDict(
        generators=OptimWrapper(
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(params=getattr(synthesizer, 'generators').parameters()))),
        discriminators=OptimWrapper(
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(
                    params=getattr(synthesizer,
                                   'discriminators').parameters()))))
    log_vars = synthesizer.train_step(data_batch, optimizer)
    assert isinstance(log_vars, dict)
    for v in [
            'loss_gan_d_mask', 'loss_gan_d_photo', 'loss_gan_g_mask',
            'loss_gan_g_photo', 'cycle_loss', 'id_loss'
    ]:
        assert isinstance(log_vars[v].item(), float)

    # test get opposite domain
    assert synthesizer._get_opposite_domain('photo') == 'mask'

    # test val_step and test_step
    data = dict(
        inputs=dict(
            img_photo=torch.randn(1, 3, 64, 64),
            img_mask=torch.randn(1, 3, 64, 64)),
        data_samples=[DataSample(mode='test')])
    out = synthesizer.test_step(data)
    assert len(out) == 1
    assert out[0].fake_photo.data.shape == (3, 64, 64)
    assert out[0].fake_mask.data.shape == (3, 64, 64)
    out = synthesizer.val_step(data)
    assert len(out) == 1
    assert out[0].fake_photo.data.shape == (3, 64, 64)
    assert out[0].fake_mask.data.shape == (3, 64, 64)

    data = dict(
        inputs=dict(
            img_photo=torch.randn(1, 3, 64, 64),
            img_mask=torch.randn(1, 3, 64, 64)))
    out = synthesizer.test_step(data)
    assert len(out) == 1
    assert out[0].fake_photo.data.shape == (3, 64, 64)
    assert out[0].fake_mask.data.shape == (3, 64, 64)
    out = synthesizer.val_step(data)
    assert len(out) == 1
    assert out[0].fake_photo.data.shape == (3, 64, 64)
    assert out[0].fake_mask.data.shape == (3, 64, 64)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

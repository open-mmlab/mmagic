# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import torch
from mmengine import Config
from mmengine.optim import OptimWrapper

from mmagic.models.editors import (DeepFillEncoderDecoder, DeepFillRefiner,
                                   GLEncoderDecoder)
from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules


def test_deepfill_encdec():
    encdec = DeepFillEncoderDecoder()
    assert isinstance(encdec.stage1, GLEncoderDecoder)
    assert isinstance(encdec.stage2, DeepFillRefiner)

    if torch.cuda.is_available():
        img = torch.rand((2, 3, 256, 256)).cuda()
        mask = img.new_zeros((2, 1, 256, 256))
        mask[..., 20:100, 30:120] = 1.
        input_x = torch.cat([img, torch.ones_like(mask), mask], dim=1)
        encdec.cuda()
        stage1_res, stage2_res = encdec(input_x)
        assert stage1_res.shape == (2, 3, 256, 256)
        assert stage2_res.shape == (2, 3, 256, 256)
        encdec = DeepFillEncoderDecoder(return_offset=True).cuda()
        stage1_res, stage2_res, offset = encdec(input_x)
        assert offset.shape == (2, 32, 32, 32, 32)


def test_deepfillv1_inpaintor():
    register_all_modules()

    config_file = osp.join(
        osp.dirname(__file__), '../../..', 'configs', 'deepfillv1_test.py')
    cfg = Config.fromfile(config_file)

    deepfillv1 = MODELS.build(cfg.model)

    assert deepfillv1.__class__.__name__ == 'DeepFillv1Inpaintor'

    if torch.cuda.is_available():
        deepfillv1.cuda()

    # check architecture
    assert deepfillv1.stage1_loss_type == ('loss_l1_hole', 'loss_l1_valid',
                                           'loss_composed_percep', 'loss_tv')
    assert deepfillv1.stage2_loss_type == ('loss_l1_hole', 'loss_l1_valid',
                                           'loss_gan')
    assert deepfillv1.with_l1_hole_loss
    assert deepfillv1.with_l1_valid_loss
    assert deepfillv1.with_composed_percep_loss
    assert not deepfillv1.with_out_percep_loss
    assert deepfillv1.with_gan

    # prepare data
    gt_img = torch.rand((3, 256, 256))
    mask = torch.zeros((1, 256, 256))
    mask[..., 50:180, 60:170] = 1.
    masked_img = gt_img.unsqueeze(0) * (1. - mask)
    mask_bbox = [100, 100, 110, 110]
    data_batch = {
        'inputs':
        masked_img,
        'data_samples': [
            DataSample(
                metainfo=dict(mask_bbox=mask_bbox),
                mask=mask,
                gt_img=gt_img,
            )
        ]
    }

    # prepare model and optimizer
    optim_g = torch.optim.Adam(deepfillv1.generator.parameters(), lr=0.0001)
    optim_d = torch.optim.Adam(deepfillv1.disc.parameters(), lr=0.0001)
    optims = dict(generator=OptimWrapper(optim_g), disc=OptimWrapper(optim_d))

    # check train_step with standard deepfillv1 model
    for i in range(5):
        log_vars = deepfillv1.train_step(data_batch, optims)

        if i % 2 == 0:
            assert 'real_loss_global' in log_vars
            assert 'fake_loss_global' in log_vars
            assert 'real_loss_local' in log_vars
            assert 'fake_loss_local' in log_vars
            assert 'loss' in log_vars
        else:
            assert 'real_loss_global' in log_vars
            assert 'fake_loss_global' in log_vars
            assert 'real_loss_local' in log_vars
            assert 'fake_loss_local' in log_vars
            assert 'stage1_loss_l1_hole' in log_vars
            assert 'stage1_loss_l1_valid' in log_vars
            assert 'stage2_loss_l1_hole' in log_vars
            assert 'stage2_loss_l1_valid' in log_vars
            assert 'stage2_loss_g_fake' in log_vars


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

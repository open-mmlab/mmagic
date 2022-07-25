# Copyright (c) OpenMMLab. All rights reserved.
from os.path import dirname, join

import torch
from mmcv import Config

from mmedit.core import build_optimizers
from mmedit.data_element import EditDataSample, PixelData
from mmedit.registry import MODELS, register_all_modules


def test_deepfillv1_inpaintor():
    register_all_modules()

    config_file = join(dirname(__file__), 'configs', 'deepfillv1_test.py')
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
    masked_img = gt_img * (1. - mask)
    mask_bbox = [100, 100, 110, 110]
    data_batch = [{
        'inputs':
        masked_img,
        'data_sample':
        EditDataSample(
            mask=PixelData(data=mask),
            mask_bbox=mask_bbox,
            gt_img=PixelData(data=gt_img),
        )
    }]

    # prepare model and optimizer
    optimizers_config = dict(
        generator=dict(type='Adam', lr=0.0001),
        disc=dict(type='Adam', lr=0.0001))

    optims = build_optimizers(deepfillv1, optimizers_config)

    # check train_step with standard deepfillv1 model
    for i in range(5):
        log_vars = deepfillv1.train_step(data_batch, optims)
        print(log_vars.keys())

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

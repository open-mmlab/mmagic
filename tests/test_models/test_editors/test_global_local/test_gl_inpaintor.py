# Copyright (c) OpenMMLab. All rights reserved.
from os.path import dirname, join

import torch
from mmengine import Config
from mmengine.optim import OptimWrapper

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules


def test_gl_inpaintor():
    register_all_modules()

    config_file = join(dirname(__file__), '../../..', 'configs', 'gl_test.py')
    cfg = Config.fromfile(config_file)

    gl = MODELS.build(cfg.model)

    assert gl.__class__.__name__ == 'GLInpaintor'

    if torch.cuda.is_available():
        gl.cuda()
    gt_img = torch.randn(3, 256, 256)
    mask = torch.zeros_like(gt_img)[0:1, ...]
    mask[..., 100:210, 100:210] = 1.
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

    optim_g = torch.optim.SGD(gl.generator.parameters(), lr=0.1)
    optim_d = torch.optim.SGD(gl.disc.parameters(), lr=0.1)
    optim_dict = dict(
        generator=OptimWrapper(optim_g), disc=OptimWrapper(optim_d))

    for i in range(5):
        log_vars = gl.train_step(data_batch, optim_dict)

        if i <= 2:
            assert 'loss_l1_hole' in log_vars
            assert 'fake_loss' not in log_vars
            assert 'real_loss' not in log_vars
            assert 'loss_g_fake' not in log_vars
        elif i == 3:
            assert 'loss_l1_hole' not in log_vars
            assert 'fake_loss' in log_vars
            assert 'real_loss' in log_vars
            assert 'loss_g_fake' not in log_vars
        else:
            assert 'loss_l1_hole' in log_vars
            assert 'fake_loss' in log_vars
            assert 'real_loss' in log_vars
            assert 'loss_g_fake' in log_vars

    gl_dirty = MODELS.build(cfg.model_dirty)
    if torch.cuda.is_available():
        gl_dirty.cuda()
    res, loss = gl_dirty.generator_loss(gt_img, gt_img, gt_img, gt_img, mask,
                                        masked_img)
    assert len(loss) == 0


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

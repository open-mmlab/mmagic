from pathlib import Path

import torch
from mmcv import Config

from mmedit.models import build_model


def test_gl_inpaintor():
    cfg = Config.fromfile(
        Path(__file__).parent.joinpath('data/inpaintor_config/gl_test.py'))

    gl = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    assert gl.__class__.__name__ == 'GLInpaintor'

    if torch.cuda.is_available():
        gt_img = torch.randn(1, 3, 256, 256)
        mask = torch.zeros_like(gt_img)[:, 0:1, ...]
        mask[..., 100:210, 100:210] = 1.
        masked_img = gt_img * (1. - mask)
        mask_bbox = torch.tensor([[100, 100, 110, 110]])
        gl.cuda()
        data_batch = dict(
            gt_img=gt_img.cuda(),
            mask=mask.cuda(),
            masked_img=masked_img.cuda(),
            mask_bbox=mask_bbox.cuda())

        optim_g = torch.optim.SGD(gl.generator.parameters(), lr=0.1)
        optim_d = torch.optim.SGD(gl.disc.parameters(), lr=0.1)
        optim_dict = dict(generator=optim_g, disc=optim_d)

        for i in range(5):
            outputs = gl.train_step(data_batch, optim_dict)

            if i <= 2:
                assert 'loss_l1_hole' in outputs['log_vars']
                assert 'fake_loss' not in outputs['log_vars']
                assert 'real_loss' not in outputs['log_vars']
                assert 'loss_g_fake' not in outputs['log_vars']
            elif i == 3:
                assert 'loss_l1_hole' not in outputs['log_vars']
                assert 'fake_loss' in outputs['log_vars']
                assert 'real_loss' in outputs['log_vars']
                assert 'loss_g_fake' not in outputs['log_vars']
            else:
                assert 'loss_l1_hole' in outputs['log_vars']
                assert 'fake_loss' in outputs['log_vars']
                assert 'real_loss' in outputs['log_vars']
                assert 'loss_g_fake' in outputs['log_vars']

        gl_dirty = build_model(
            cfg.model_dirty, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        gl_dirty.cuda()
        res, loss = gl_dirty.generator_loss(gt_img, gt_img, gt_img, data_batch)
        assert len(loss) == 0

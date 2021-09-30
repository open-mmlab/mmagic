# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from unittest.mock import patch

import pytest
import torch
from mmcv import Config

from mmedit.models import build_model
from mmedit.models.losses import PerceptualVGG


@patch.object(PerceptualVGG, 'init_weights')
def test_pconv_inpaintor(init_weights):
    cfg = Config.fromfile(
        'tests/data/inpaintor_config/pconv_inpaintor_test.py')

    if torch.cuda.is_available():
        pconv_inpaintor = build_model(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        assert pconv_inpaintor.__class__.__name__ == 'PConvInpaintor'
        pconv_inpaintor.cuda()
        gt_img = torch.randn((1, 3, 256, 256)).cuda()
        mask = torch.zeros_like(gt_img)
        mask[..., 50:160, 100:210] = 1.
        masked_img = gt_img * (1. - mask)
        data_batch = dict(gt_img=gt_img, mask=mask, masked_img=masked_img)
        optim_g = torch.optim.SGD(
            pconv_inpaintor.generator.parameters(), lr=0.1)
        optim_dict = dict(generator=optim_g)

        outputs = pconv_inpaintor.train_step(data_batch, optim_dict)
        assert outputs['results']['fake_res'].shape == (1, 3, 256, 256)
        assert outputs['results']['final_mask'].shape == (1, 3, 256, 256)
        assert 'loss_l1_hole' in outputs['log_vars']
        assert 'loss_l1_valid' in outputs['log_vars']
        assert 'loss_tv' in outputs['log_vars']

        # test forward dummy
        res = pconv_inpaintor.forward_dummy(
            torch.cat([masked_img, mask], dim=1))
        assert res.shape == (1, 3, 256, 256)

        # test forward test w/o save image
        outputs = pconv_inpaintor.forward_test(
            masked_img[0:1], mask[0:1], gt_img=gt_img[0:1, ...])
        assert 'eval_result' in outputs
        assert outputs['eval_result']['l1'] > 0
        assert outputs['eval_result']['psnr'] > 0
        assert outputs['eval_result']['ssim'] > 0

        # test forward test w/o eval metrics
        pconv_inpaintor.test_cfg = dict()
        pconv_inpaintor.eval_with_metrics = False
        outputs = pconv_inpaintor.forward_test(masked_img[0:1], mask[0:1])
        for key in ['fake_res', 'fake_img']:
            assert outputs[key].size() == (1, 3, 256, 256)

        # test forward test w/ save image
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = pconv_inpaintor.forward_test(
                masked_img[0:1],
                mask[0:1],
                save_image=True,
                save_path=tmpdir,
                iteration=4396,
                meta=[dict(gt_img_path='igccc.png')])

            assert os.path.exists(os.path.join(tmpdir, 'igccc_4396.png'))

        # test forward test w/ save image w/ gt_img
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = pconv_inpaintor.forward_test(
                masked_img[0:1],
                mask[0:1],
                save_image=True,
                save_path=tmpdir,
                meta=[dict(gt_img_path='igccc.png')],
                gt_img=gt_img[0:1, ...])

            assert os.path.exists(os.path.join(tmpdir, 'igccc.png'))

            with pytest.raises(AssertionError):
                outputs = pconv_inpaintor.forward_test(
                    masked_img[0:1],
                    mask[0:1],
                    save_image=True,
                    save_path=tmpdir,
                    iteration=4396,
                    gt_img=gt_img[0:1, ...])
            with pytest.raises(AssertionError):
                outputs = pconv_inpaintor.forward_test(
                    masked_img[0:1],
                    mask[0:1],
                    save_image=True,
                    save_path=None,
                    iteration=4396,
                    meta=[dict(gt_img_path='igccc.png')],
                    gt_img=gt_img[0:1, ...])

    # reset mock to clear some memory usage
    init_weights.reset_mock()

from pathlib import Path
from unittest.mock import patch

import torch
from mmcv import Config
from mmedit.models import build_model
from mmedit.models.losses import PerceptualVGG


@patch.object(PerceptualVGG, 'init_weights')
def test_pconv_inpaintor(init_weights):
    cfg = Config.fromfile(
        Path(__file__).parent.joinpath(
            'data/inpaintor_config/pconv_inpaintor_test.py'))

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

        output = pconv_inpaintor.forward_test(**data_batch)
        assert output['fake_res'].shape == (1, 3, 256, 256)

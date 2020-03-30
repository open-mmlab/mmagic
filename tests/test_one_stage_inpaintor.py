from pathlib import Path

import pytest
import torch
from mmcv import Config
from mmedit.models import build_model
from mmedit.models.backbones import GLEncoderDecoder


def test_os_inpaintor():
    cfg = Config.fromfile(
        Path(__file__).parent.joinpath('data/inpaintor_config/os_gl.py'))
    os_inpaintor = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # test attributes
    assert os_inpaintor.__class__.__name__ == 'OneStageInpaintor'
    assert isinstance(os_inpaintor.generator, GLEncoderDecoder)
    assert not os_inpaintor.with_l1_hole_loss
    assert not os_inpaintor.with_l1_valid_loss
    assert not os_inpaintor.with_tv_loss
    assert os_inpaintor.with_composed_percep_loss
    assert not os_inpaintor.with_out_percep_loss
    assert os_inpaintor.with_gan
    assert os_inpaintor.with_gp_loss
    assert not os_inpaintor.with_disc_shift_loss
    assert os_inpaintor.is_train
    assert os_inpaintor.train_cfg['disc_step'] == 1
    assert os_inpaintor.test_cfg['test'] == 1
    assert os_inpaintor.disc_step_count == 0

    input_x = torch.randn(1, 3, 256, 256)
    with pytest.raises(NotImplementedError):
        os_inpaintor.forward_train(input_x)

    if torch.cuda.is_available():
        gt_img = torch.randn(1, 3, 256, 256).cuda()
        mask = torch.zeros_like(gt_img)[:, 0:1, ...]
        mask[..., 20:100, 100:120] = 1.
        masked_img = gt_img * (1. - mask)
        os_inpaintor = os_inpaintor.cuda()
        data_batch = dict(gt_img=gt_img, mask=mask, masked_img=masked_img)
        output = os_inpaintor.forward_test(data_batch)
        assert output['fake_res'].shape == (1, 3, 256, 256)
        assert output['fake_img'].shape == (1, 3, 256, 256)

        output = os_inpaintor.val_step(data_batch)
        assert output['fake_res'].shape == (1, 3, 256, 256)
        assert output['fake_img'].shape == (1, 3, 256, 256)

        optim_g = torch.optim.SGD(os_inpaintor.generator.parameters(), lr=0.1)
        optim_d = torch.optim.SGD(os_inpaintor.disc.parameters(), lr=0.1)
        optim_dict = dict(generator=optim_g, disc=optim_d)

        outputs = os_inpaintor.train_step(data_batch, optim_dict)
        assert outputs['num_samples'] == 1
        results = outputs['results']
        assert results['fake_res'].shape == (1, 3, 256, 256)

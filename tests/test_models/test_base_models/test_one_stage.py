# Copyright (c) OpenMMLab. All rights reserved.
import copy
from os.path import dirname, join
from unittest.mock import patch

import pytest
import torch
from mmengine import Config

from mmagic.models import GLEncoderDecoder
from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules


def test_one_stage_inpaintor():
    register_all_modules()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_file = join(
        dirname(__file__), '..', '..', 'configs', 'one_stage_gl.py')
    cfg = Config.fromfile(config_file)

    # mock perceptual loss for test speed
    cfg.model.loss_composed_percep = None
    inpaintor = MODELS.build(cfg.model)

    # modify attributes for mocking
    inpaintor.with_composed_percep_loss = True
    inpaintor.loss_percep = None

    # test attributes
    assert inpaintor.__class__.__name__ == 'OneStageInpaintor'
    assert isinstance(inpaintor.generator, GLEncoderDecoder)
    assert inpaintor.with_l1_hole_loss
    assert inpaintor.with_l1_valid_loss
    assert inpaintor.with_tv_loss
    assert inpaintor.with_composed_percep_loss
    assert inpaintor.with_out_percep_loss
    assert inpaintor.with_gan
    assert inpaintor.with_gp_loss
    assert inpaintor.with_disc_shift_loss
    assert inpaintor.is_train
    assert inpaintor.train_cfg['disc_step'] == 1
    assert inpaintor.disc_step_count == 0

    with patch.object(
            inpaintor, 'loss_percep', return_value=(torch.tensor(1.0), None)):

        input_x = torch.randn(1, 3, 256, 256)
        with pytest.raises(NotImplementedError):
            inpaintor.forward_train(input_x)

        inpaintor.to(device=device)

        # construct inputs
        gt_img = torch.randn(3, 256, 256).to(device=device)
        mask = torch.zeros_like(gt_img)[0:1, ...]
        mask[..., 20:100, 100:120] = 1.
        masked_img = gt_img * (1. - mask)
        mask_bbox = [100, 100, 110, 110]
        inputs = masked_img.unsqueeze(0)
        data_sample = DataSample(
            mask=mask,
            mask_bbox=mask_bbox,
            gt_img=gt_img,
        )
        data_samples = DataSample.stack([data_sample])
        data_batch = {'inputs': inputs, 'data_samples': [data_sample]}

        # test forward_tensor
        fake_reses, fake_imgs = inpaintor.forward_tensor(inputs, data_samples)
        assert fake_reses.shape == fake_imgs.shape == (1, 3, 256, 256)

        # test forward test
        predictions = inpaintor.forward_test(inputs, data_samples)
        assert predictions.fake_img.shape == (1, 3, 256, 256)

        # test train_step
        optim_g = torch.optim.SGD(inpaintor.generator.parameters(), lr=0.1)
        optim_d = torch.optim.SGD(inpaintor.disc.parameters(), lr=0.1)
        optim_dict = dict(generator=optim_g, disc=optim_d)

        log_vars = inpaintor.train_step(data_batch, optim_dict)
        assert 'loss_l1_hole' in log_vars
        assert 'loss_l1_valid' in log_vars
        assert 'loss_composed_percep' in log_vars
        assert 'loss_composed_style' not in log_vars
        assert 'loss_out_percep' in log_vars
        assert 'loss_out_style' not in log_vars
        assert 'loss_tv' in log_vars
        assert 'fake_loss' in log_vars
        assert 'real_loss' in log_vars
        assert 'loss_g_fake' in log_vars

        # Another test
        cfg_ = copy.deepcopy(cfg)
        cfg_.model.train_cfg.disc_step = 2
        inpaintor = MODELS.build(cfg_.model)
        inpaintor.to(device=device)
        assert inpaintor.train_cfg.disc_step == 2
        log_vars = inpaintor.train_step(data_batch, optim_dict)
        assert 'loss_l1_hole' not in log_vars


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

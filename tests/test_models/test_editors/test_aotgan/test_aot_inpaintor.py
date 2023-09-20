# Copyright (c) OpenMMLab. All rights reserved.
from os.path import dirname, join

import torch
from mmengine import Config
from mmengine.optim import OptimWrapper

from mmagic.models import AOTEncoderDecoder
from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules


def test_aot_inpaintor():
    register_all_modules()

    config_file = join(dirname(__file__), '../../..', 'configs', 'aot_test.py')
    cfg = Config.fromfile(config_file)

    inpaintor = MODELS.build(cfg.model)

    # optimizer
    optim_g = torch.optim.Adam(
        inpaintor.generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optim_d = torch.optim.Adam(
        inpaintor.disc.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optims = dict(generator=OptimWrapper(optim_g), disc=OptimWrapper(optim_d))

    # test attributes
    assert inpaintor.__class__.__name__ == 'AOTInpaintor'
    assert isinstance(inpaintor.generator, AOTEncoderDecoder)
    assert inpaintor.with_l1_valid_loss
    assert inpaintor.with_composed_percep_loss
    assert inpaintor.with_out_percep_loss
    assert inpaintor.with_gan
    assert inpaintor.is_train
    assert inpaintor.train_cfg['disc_step'] == 1
    assert inpaintor.disc_step_count == 0

    if torch.cuda.is_available():
        inpaintor = inpaintor.cuda()

    gt_img = torch.randn(3, 64, 64)
    mask = torch.zeros((1, 64, 64))
    mask[..., 12:45, 14:42] = 1.
    masked_img = gt_img.unsqueeze(0) * (1. - mask) + mask
    mask_bbox = [25, 25, 27, 27]
    data_batch = {
        'inputs':
        masked_img,
        'data_samples':
        [DataSample(
            mask=mask,
            mask_bbox=mask_bbox,
            gt_img=gt_img,
        )]
    }

    # check train_step
    for i in range(5):
        log_vars = inpaintor.train_step(data_batch, optims)

        assert 'loss_l1_valid' in log_vars
        assert 'loss_out_percep' in log_vars
        assert 'disc_losses' in log_vars
        assert 'loss_g_fake' in log_vars

    # # check forward_test
    data = inpaintor.data_preprocessor(data_batch, True)
    data_inputs, data_sample = data['inputs'], data['data_samples']
    output = inpaintor.forward_test(data_inputs, data_sample)
    prediction = output
    assert 'fake_res' in prediction
    assert 'fake_img' in prediction
    assert 'pred_img' in prediction
    assert prediction.pred_img.shape == (1, 3, 64, 64)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

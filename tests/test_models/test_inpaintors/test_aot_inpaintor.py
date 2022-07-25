# Copyright (c) OpenMMLab. All rights reserved.
from os.path import dirname, join

import torch
from mmcv import Config

from mmedit.core import build_optimizers
from mmedit.data_element import EditDataSample, PixelData
from mmedit.models.inpaintors import AOTEncoderDecoder
from mmedit.registry import MODELS, register_all_modules


def test_aot_inpaintor():
    register_all_modules()

    config_file = join(dirname(__file__), 'configs', 'aot_test.py')
    cfg = Config.fromfile(config_file)

    inpaintor = MODELS.build(cfg.model)
    optims = build_optimizers(inpaintor, cfg.optim_wrapper)

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

    gt_img = torch.randn(3, 256, 256)
    mask = torch.zeros((1, 256, 256))
    mask[..., 50:180, 60:170] = 1.
    masked_img = gt_img * (1. - mask) + mask
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

    # check train_step
    for i in range(5):
        log_vars = inpaintor.train_step(data_batch, optims)

        assert 'loss_l1_valid' in log_vars
        assert 'loss_out_percep' in log_vars
        assert 'disc_losses' in log_vars
        assert 'loss_g_fake' in log_vars

    # # check forward_test
    data_inputs, data_sample = inpaintor.data_preprocessor(data_batch, True)
    output = inpaintor.forward_test(data_inputs, data_sample)
    prediction = output[0]
    assert 'fake_res' in prediction
    assert '_pred_img' in prediction
    assert 'fake_img' in prediction
    assert 'pred_img' in prediction
    assert prediction.pred_img.shape == (256, 256)

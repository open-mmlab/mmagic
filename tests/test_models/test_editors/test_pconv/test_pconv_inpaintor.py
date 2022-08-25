# Copyright (c) OpenMMLab. All rights reserved.
from os.path import dirname, join

import torch
from mmengine import Config
from mmengine.optim import OptimWrapper

from mmedit.registry import MODELS
from mmedit.structures import EditDataSample, PixelData
from mmedit.utils import register_all_modules


def test_pconv_inpaintor():
    register_all_modules()

    config_file = join(
        dirname(__file__), '../../..', 'configs', 'pconv_test.py')
    cfg = Config.fromfile(config_file)

    inpaintor = MODELS.build(cfg.model)

    assert inpaintor.__class__.__name__ == 'PConvInpaintor'

    if torch.cuda.is_available():
        inpaintor = inpaintor.cuda()

    gt_img = torch.randn(3, 256, 256)
    mask = torch.zeros_like(gt_img)[0:1, ...]
    mask[..., 100:210, 100:210] = 1.
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

    optim_g = torch.optim.Adam(inpaintor.generator.parameters(), lr=0.0001)
    optim_g = OptimWrapper(optim_g)

    for i in range(5):
        log_vars = inpaintor.train_step(data_batch, optim_g)
        assert 'loss_l1_hole' in log_vars
        assert 'loss_l1_valid' in log_vars
        assert 'loss_tv' in log_vars

    # check for forward_test
    data_inputs, data_sample = inpaintor.data_preprocessor(data_batch, True)
    output = inpaintor.forward_test(data_inputs, data_sample)
    prediction = output[0]
    assert 'fake_res' in prediction
    assert '_pred_img' in prediction
    assert 'fake_img' in prediction
    assert 'pred_img' in prediction
    assert prediction.pred_img.shape == (256, 256)

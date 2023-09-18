# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

import torch
from mmengine.optim import OptimWrapper
from torch.optim import Adam

from mmagic.models import DIC, DataPreprocessor, DICNet, LightCNN
from mmagic.models.losses import (GANLoss, L1Loss, LightCNNFeatureLoss,
                                  PerceptualVGG)
from mmagic.structures import DataSample


@patch.object(PerceptualVGG, 'init_weights')
def test_dic(init_weights):

    model = DIC(
        generator=dict(
            type='DICNet', in_channels=3, out_channels=3, mid_channels=4),
        discriminator=dict(type='LightCNN', in_channels=3),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        align_loss=dict(type='MSELoss', loss_weight=0.1, reduction='mean'),
        feature_loss=dict(
            type='LightCNNFeatureLoss',
            pretrained=None,
            loss_weight=0.1,
            criterion='l1'),
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            loss_weight=0.005,
            real_label_val=1.0,
            fake_label_val=0),
        train_cfg=dict(),
        test_cfg=dict(),
        data_preprocessor=DataPreprocessor(
            mean=[129.795, 108.12, 96.39],
            std=[255, 255, 255],
        ))

    assert isinstance(model, DIC)
    assert isinstance(model.generator, DICNet)
    assert isinstance(model.discriminator, LightCNN)
    assert isinstance(model.pixel_loss, L1Loss)
    assert isinstance(model.feature_loss, LightCNNFeatureLoss)
    assert isinstance(model.gan_loss, GANLoss)

    optimizer_g = Adam(
        model.generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    optimizer_d = Adam(
        model.discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    optim_wrapper = dict(
        generator=OptimWrapper(optimizer_g),
        discriminator=OptimWrapper(optimizer_d))

    # prepare data
    inputs = torch.rand(1, 3, 16, 16)
    target = torch.rand(3, 128, 128)
    data_sample = DataSample(gt_img=target, gt_heatmap=torch.rand(68, 32, 32))
    data = dict(inputs=inputs, data_samples=[data_sample])

    # train
    log_vars = model.train_step(data, optim_wrapper)
    log_vars = model.train_step(data, optim_wrapper)
    assert isinstance(log_vars, dict)
    assert set(log_vars.keys()) == set([
        'loss_pixel_v0', 'loss_align_v0', 'loss_pixel_v1', 'loss_align_v1',
        'loss_pixel_v2', 'loss_align_v2', 'loss_pixel_v3', 'loss_align_v3',
        'loss_feature', 'loss_gan', 'loss_d_real', 'loss_d_fake'
    ])

    # val
    output = model.val_step(data)
    assert output[0].output.pred_img.shape == (3, 128, 128)

    # feat
    output = model(torch.rand(1, 3, 16, 16), mode='tensor')
    assert output.shape == (1, 3, 128, 128)

    # reset mock to clear some memory usage
    init_weights.reset_mock()


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

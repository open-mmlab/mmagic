# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

import torch
from mmengine.optim import OptimWrapper
from torch.optim import Adam

from mmedit.models import (LTE, TTSR, EditDataPreprocessor, SearchTransformer,
                           TTSRDiscriminator, TTSRNet)
from mmedit.models.losses import (GANLoss, L1Loss, PerceptualVGG,
                                  TransferalPerceptualLoss)
from mmedit.registry import MODELS
from mmedit.structures import EditDataSample, PixelData


@patch.object(PerceptualVGG, 'init_weights')
def test_ttsr(init_weights):
    model_cfg = dict(
        type='TTSR',
        generator=dict(
            type='TTSRNet',
            in_channels=3,
            out_channels=3,
            mid_channels=4,
            num_blocks=(1, 1, 1, 1)),
        extractor=dict(type='LTE', load_pretrained_vgg=False),
        transformer=dict(type='SearchTransformer'),
        discriminator=dict(type='TTSRDiscriminator', in_size=128),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        perceptual_loss=dict(
            type='PerceptualLoss',
            layer_weights={'29': 1.0},
            vgg_type='vgg19',
            perceptual_weight=1e-2,
            style_weight=0.001,
            criterion='mse'),
        transferal_perceptual_loss=dict(
            type='TransferalPerceptualLoss',
            loss_weight=1e-2,
            use_attention=False,
            criterion='mse'),
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            loss_weight=1e-3,
            real_label_val=1.0,
            fake_label_val=0),
        data_preprocessor=EditDataPreprocessor(
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
        ))

    # build restorer
    model = MODELS.build(model_cfg)

    # test attributes
    assert isinstance(model, TTSR)
    assert isinstance(model.generator, TTSRNet)
    assert isinstance(model.discriminator, TTSRDiscriminator)
    assert isinstance(model.transformer, SearchTransformer)
    assert isinstance(model.extractor, LTE)
    assert isinstance(model.pixel_loss, L1Loss)
    assert isinstance(model.transferal_perceptual_loss,
                      TransferalPerceptualLoss)
    assert isinstance(model.gan_loss, GANLoss)

    optimizer_g = Adam(
        model.generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    optimizer_d = Adam(
        model.discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    optim_wrapper = dict(
        generator=OptimWrapper(optimizer_g),
        extractor=OptimWrapper(optimizer_g),
        discriminator=OptimWrapper(optimizer_d))

    # prepare data
    inputs = torch.rand(1, 3, 32, 32)
    data_sample = EditDataSample(
        gt_img=PixelData(data=torch.rand(3, 128, 128)),
        ref_img=PixelData(data=torch.rand(3, 128, 128)),
        img_lq=PixelData(data=torch.rand(3, 128, 128)),
        ref_lq=PixelData(data=torch.rand(3, 128, 128)))
    data = dict(inputs=inputs, data_samples=[data_sample])

    # train
    log_vars = model.train_step(data, optim_wrapper)
    log_vars = model.train_step(data, optim_wrapper)
    assert isinstance(log_vars, dict)
    assert set(log_vars.keys()) == set([
        'loss_pix', 'loss_perceptual', 'loss_style', 'loss_transferal',
        'loss_gan', 'loss_d_real', 'loss_d_fake'
    ])

    # val
    output = model.val_step(data)
    assert output[0].output.pred_img.data.shape == (3, 128, 128)

    # feat
    output = model(torch.rand(1, 3, 32, 32), [data_sample], mode='tensor')
    assert output.shape == (1, 3, 128, 128)

    # reset mock to clear some memory usage
    init_weights.reset_mock()

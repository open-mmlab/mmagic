# Copyright (c) OpenMMLab. All rights reserved.
model = dict(
    type='PConvInpaintor',
    encdec=dict(
        type='PConvEncoderDecoder',
        encoder=dict(type='PConvEncoder'),
        decoder=dict(type='PConvDecoder')),
    loss_l1_hole=dict(type='L1Loss', loss_weight=1.0),
    loss_l1_valid=dict(type='L1Loss', loss_weight=1.0),
    loss_composed_percep=dict(
        type='PerceptualLoss',
        layer_weights={'0': 1.},
        perceptual_weight=0.1,
        style_weight=0),
    loss_out_percep=True,
    loss_tv=dict(type='MaskedTVLoss', loss_weight=0.01),
    pretrained=None)

train_cfg = dict(disc_step=0)
test_cfg = dict(metrics=['l1', 'psnr', 'ssim'])

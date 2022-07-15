# Copyright (c) OpenMMLab. All rights reserved.
model = dict(
    type='AOTInpaintor',
    encdec=dict(
        type='AOTEncoderDecoder',
        encoder=dict(type='AOTEncoder'),
        decoder=dict(type='AOTDecoder'),
        dilation_neck=dict(type='AOTBlockNeck')),
    disc=dict(
        type='SoftMaskPatchDiscriminator',
        in_channels=3,
        base_channels=64,
        num_conv=3,
        with_spectral_norm=True,
    ),
    loss_gan=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=0.01,
    ),
    loss_composed_percep=dict(
        type='PerceptualLoss',
        layer_weights={'0': 1.},
        perceptual_weight=0.1,
        style_weight=0,
    ),
    loss_out_percep=True,
    loss_l1_valid=dict(type='L1Loss', loss_weight=1.0),
    pretrained=None)

train_cfg = dict(disc_step=1)
test_cfg = dict(metrics=['l1', 'psnr', 'ssim'])

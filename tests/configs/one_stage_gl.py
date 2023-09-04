# Copyright (c) OpenMMLab. All rights reserved.
model = dict(
    type='OneStageInpaintor',
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[127.5],
        std=[127.5],
    ),
    encdec=dict(
        type='GLEncoderDecoder',
        encoder=dict(type='GLEncoder'),
        decoder=dict(type='GLDecoder'),
        dilation_neck=dict(type='GLDilationNeck')),
    disc=dict(
        type='MultiLayerDiscriminator',
        in_channels=3,
        max_channels=512,
        fc_in_channels=512 * 4 * 4,
        fc_out_channels=1024,
        num_convs=6,
        norm_cfg=dict(type='BN'),
    ),
    loss_gan=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=0.001,
    ),
    loss_gp=dict(
        type='GradientPenaltyLoss',
        loss_weight=1.,
    ),
    loss_disc_shift=dict(type='DiscShiftLoss', loss_weight=0.001),
    loss_composed_percep=dict(
        type='PerceptualLoss',
        layer_weights={'0': 1.},
        perceptual_weight=0.1,
        style_weight=0,
    ),
    loss_out_percep=True,
    loss_l1_hole=dict(type='L1Loss', loss_weight=1.0),
    loss_l1_valid=dict(type='L1Loss', loss_weight=1.0),
    loss_tv=dict(type='MaskedTVLoss', loss_weight=0.01),
    train_cfg=dict(disc_step=1))


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

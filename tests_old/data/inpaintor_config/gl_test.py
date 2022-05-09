# Copyright (c) OpenMMLab. All rights reserved.
global_disc_cfg = dict(
    in_channels=3,
    max_channels=512,
    fc_in_channels=512 * 4 * 4,
    fc_out_channels=1024,
    num_convs=6,
    norm_cfg=dict(type='BN'))
local_disc_cfg = dict(
    in_channels=3,
    max_channels=512,
    fc_in_channels=512 * 4 * 4,
    fc_out_channels=1024,
    num_convs=5,
    norm_cfg=dict(type='BN'))

model = dict(
    type='GLInpaintor',
    encdec=dict(
        type='GLEncoderDecoder',
        encoder=dict(type='GLEncoder'),
        decoder=dict(type='GLDecoder'),
        dilation_neck=dict(type='GLDilationNeck')),
    disc=dict(
        type='GLDiscs',
        global_disc_cfg=global_disc_cfg,
        local_disc_cfg=local_disc_cfg),
    loss_gan=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=0.001,
    ),
    loss_l1_hole=dict(type='L1Loss', loss_weight=1.0),
    loss_l1_valid=dict(type='L1Loss', loss_weight=1.0),
    pretrained=None)

train_cfg = dict(
    disc_step=1, start_iter=0, iter_tc=2, iter_td=3, local_size=(128, 128))
test_cfg = dict()

model_dirty = dict(
    type='GLInpaintor',
    encdec=dict(
        type='GLEncoderDecoder',
        encoder=dict(type='GLEncoder'),
        decoder=dict(type='GLDecoder'),
        dilation_neck=dict(type='GLDilationNeck')),
    disc=dict(
        type='GLDiscs',
        global_disc_cfg=global_disc_cfg,
        local_disc_cfg=local_disc_cfg),
    loss_gan=None,
    loss_l1_hole=None,
    loss_l1_valid=None,
    pretrained=None)

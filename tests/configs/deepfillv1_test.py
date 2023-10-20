# Copyright (c) OpenMMLab. All rights reserved.

model = dict(
    type='DeepFillv1Inpaintor',
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[127.5],
        std=[127.5],
    ),
    encdec=dict(type='DeepFillEncoderDecoder'),
    disc=dict(
        type='DeepFillv1Discriminators',
        global_disc_cfg=dict(
            type='MultiLayerDiscriminator',
            in_channels=3,
            max_channels=256,
            fc_in_channels=256 * 16 * 16,
            fc_out_channels=1,
            num_convs=4,
            norm_cfg=None,
            act_cfg=dict(type='ELU'),
            out_act_cfg=dict(type='LeakyReLU', negative_slope=0.2)),
        local_disc_cfg=dict(
            type='MultiLayerDiscriminator',
            in_channels=3,
            max_channels=512,
            fc_in_channels=512 * 8 * 8,
            fc_out_channels=1,
            num_convs=4,
            norm_cfg=None,
            act_cfg=dict(type='ELU'),
            out_act_cfg=dict(type='LeakyReLU', negative_slope=0.2))),
    stage1_loss_type=('loss_l1_hole', 'loss_l1_valid', 'loss_composed_percep',
                      'loss_tv'),
    stage2_loss_type=('loss_l1_hole', 'loss_l1_valid', 'loss_gan'),
    loss_gan=dict(
        type='GANLoss',
        gan_type='hinge',
        loss_weight=1,
    ),
    loss_l1_hole=dict(
        type='L1Loss',
        loss_weight=1.0,
    ),
    loss_l1_valid=dict(
        type='L1Loss',
        loss_weight=1.0,
    ),
    loss_composed_percep=dict(
        type='PerceptualLoss',
        vgg_type='vgg16',
        layer_weights={
            '4': 1.,
            '9': 1.,
            '16': 1.,
        },
        perceptual_weight=0.05,
        style_weight=120,
        pretrained=('torchvision://vgg16')),
    loss_tv=dict(
        type='MaskedTVLoss',
        loss_weight=0.1,
    ),
    loss_gp=dict(type='GradientPenaltyLoss', loss_weight=10.),
    loss_disc_shift=dict(type='DiscShiftLoss', loss_weight=0.001),
    train_cfg=dict(disc_step=2, start_iter=0, local_size=(128, 128)),
)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

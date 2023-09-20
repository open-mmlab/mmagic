# Copyright (c) OpenMMLab. All rights reserved.

model = dict(
    type='TwoStageInpaintor',
    disc_input_with_mask=True,
    train_cfg=dict(disc_step=1, start_iter=0),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[127.5],
        std=[127.5],
    ),
    encdec=dict(
        type='DeepFillEncoderDecoder',
        stage1=dict(
            type='GLEncoderDecoder',
            encoder=dict(
                type='DeepFillEncoder',
                conv_type='gated_conv',
                channel_factor=0.75,
                padding_mode='reflect'),
            decoder=dict(
                type='DeepFillDecoder',
                conv_type='gated_conv',
                in_channels=96,
                channel_factor=0.75,
                out_act_cfg=dict(type='Tanh'),
                padding_mode='reflect'),
            dilation_neck=dict(
                type='GLDilationNeck',
                in_channels=96,
                conv_type='gated_conv',
                act_cfg=dict(type='ELU'),
                padding_mode='reflect')),
        stage2=dict(
            type='DeepFillRefiner',
            encoder_attention=dict(
                type='DeepFillEncoder',
                encoder_type='stage2_attention',
                conv_type='gated_conv',
                channel_factor=0.75,
                padding_mode='reflect'),
            encoder_conv=dict(
                type='DeepFillEncoder',
                encoder_type='stage2_conv',
                conv_type='gated_conv',
                channel_factor=0.75,
                padding_mode='reflect'),
            dilation_neck=dict(
                type='GLDilationNeck',
                in_channels=96,
                conv_type='gated_conv',
                act_cfg=dict(type='ELU'),
                padding_mode='reflect'),
            contextual_attention=dict(
                type='ContextualAttentionNeck',
                in_channels=96,
                conv_type='gated_conv',
                padding_mode='reflect'),
            decoder=dict(
                type='DeepFillDecoder',
                in_channels=192,
                conv_type='gated_conv',
                out_act_cfg=dict(type='Tanh'),
                padding_mode='reflect'))),
    disc=dict(
        type='MultiLayerDiscriminator',
        in_channels=4,
        max_channels=256,
        fc_in_channels=None,
        num_convs=6,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        out_act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        with_spectral_norm=True,
    ),
    stage1_loss_type=('loss_l1_hole', 'loss_l1_valid', 'loss_composed_percep',
                      'loss_tv'),
    stage2_loss_type=('loss_l1_hole', 'loss_l1_valid', 'loss_gan'),
    loss_gan=dict(
        type='GANLoss',
        gan_type='hinge',
        loss_weight=0.1,
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
    loss_disc_shift=dict(type='DiscShiftLoss', loss_weight=0.001),
)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

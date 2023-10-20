# Copyright (c) OpenMMLab. All rights reserved.

input_shape = (256, 256)

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
    train_cfg=dict(
        disc_step=1, start_iter=0, iter_tc=2, iter_td=3,
        local_size=(128, 128)))

model_dirty = dict(
    type='GLInpaintor',
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
        type='GLDiscs',
        global_disc_cfg=global_disc_cfg,
        local_disc_cfg=local_disc_cfg),
    loss_gan=None,
    loss_l1_hole=None,
    loss_l1_valid=None)

model_inference = dict(
    type='GLInpaintor',
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
        type='GLDiscs',
        global_disc_cfg=dict(
            in_channels=3,
            max_channels=512,
            fc_in_channels=512 * 4 * 4,
            fc_out_channels=1024,
            num_convs=6,
            norm_cfg=dict(type='SyncBN'),
        ),
        local_disc_cfg=dict(
            in_channels=3,
            max_channels=512,
            fc_in_channels=512 * 4 * 4,
            fc_out_channels=1024,
            num_convs=5,
            norm_cfg=dict(type='SyncBN'),
        ),
    ),
    loss_gan=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=0.001,
    ),
    loss_l1_hole=dict(
        type='L1Loss',
        loss_weight=1.0,
    ))

test_pipeline = [
    dict(type='LoadImageFromFile', key='gt'),
    dict(
        type='LoadMask',
        mask_mode='bbox',
        mask_config=dict(
            max_bbox_shape=(128, 128),
            max_bbox_delta=40,
            min_margin=20,
            img_shape=input_shape)),
    dict(
        type='Crop',
        keys=['gt'],
        crop_size=(384, 384),
        random_crop=True,
    ),
    dict(
        type='Resize',
        keys=['gt'],
        scale=input_shape,
        keep_ratio=False,
    ),
    dict(type='GetMaskedImage'),
    dict(type='PackInputs'),
]


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()

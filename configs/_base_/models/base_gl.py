# DistributedDataParallel
model_wrapper_cfg = dict(type='MMSeparateDistributedDataParallel')

model = dict(
    type='GLInpaintor',
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[127.5],
        std=[127.5],
    ),
    encdec=dict(
        type='GLEncoderDecoder',
        encoder=dict(type='GLEncoder', norm_cfg=dict(type='SyncBN')),
        decoder=dict(type='GLDecoder', norm_cfg=dict(type='SyncBN')),
        dilation_neck=dict(
            type='GLDilationNeck', norm_cfg=dict(type='SyncBN'))),
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

# optimizer
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper', optimizer=dict(type='Adam', lr=0.0004)),
    disc=dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=0.0004)))

# learning policy
# Fixed

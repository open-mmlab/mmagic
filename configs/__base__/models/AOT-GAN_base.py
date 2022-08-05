# DistributedDataParallel
model_wrapper_cfg = dict(type='MMSeparateDistributedDataParallel')

model = dict(
    type='AOTInpaintor',
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[127.5],
        std=[127.5],
    ),
    encdec=dict(
        type='AOTEncoderDecoder',
        encoder=dict(type='AOTEncoder'),
        decoder=dict(type='AOTDecoder'),
        dilation_neck=dict(
            type='AOTBlockNeck', dilation_rates=(1, 2, 4, 8), num_aotblock=8)),
    disc=dict(
        type='SoftMaskPatchDiscriminator',
        in_channels=3,
        base_channels=64,
        num_conv=3,
        with_spectral_norm=True,
    ),
    loss_gan=dict(
        type='GANLoss',
        gan_type='smgan',
        loss_weight=0.01,
    ),
    loss_composed_percep=dict(
        type='PerceptualLoss',
        vgg_type='vgg19',
        layer_weights={
            '1': 1.,
            '6': 1.,
            '11': 1.,
            '20': 1.,
            '29': 1.,
        },
        layer_weights_style={
            '8': 1.,
            '17': 1.,
            '26': 1.,
            '31': 1.,
        },
        perceptual_weight=0.1,
        style_weight=250),
    loss_out_percep=True,
    loss_l1_valid=dict(
        type='L1Loss',
        loss_weight=1.,
    ),
)

# optimizer
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.0, 0.9))),
    disc=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.0, 0.9))))

# learning policy
# Fixed

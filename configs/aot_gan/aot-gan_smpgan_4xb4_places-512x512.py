_base_ = [
    '../_base_/inpaint_default_runtime.py', '../_base_/datasets/places.py'
]

experiment_name = 'aot-gan_smpgan_4xb4_places-512x512'
save_dir = './work_dirs'

input_shape = (512, 512)

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
    train_cfg=dict(
        disc_step=1,
        start_iter=0,
    ))

mask_root = 'data/pconv_mask'
train_pipeline = [
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(
        type='LoadMask',
        mask_mode='set',
        mask_config=dict(
            mask_list_file=f'{mask_root}/train_mask_list.txt',
            prefix=mask_root,
            io_backend='disk',
            flag='unchanged',
            color_type='unchanged',
            file_client_kwargs=dict())),
    dict(
        type='RandomResizedCrop',
        keys=['gt'],
        crop_size=input_shape,
    ),
    dict(type='Flip', keys=['gt', 'mask'], direction='horizontal'),
    dict(
        type='Resize',
        keys=['mask'],
        scale=input_shape,
        keep_ratio=False,
        interpolation='nearest'),
    dict(type='RandomRotation', keys=['mask'], degrees=(0.0, 45.0)),
    dict(
        type='ColorJitter',
        keys=['gt'],
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.5),
    dict(type='GetMaskedImage'),
    dict(type='PackEditInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(
        type='LoadMask',
        mask_mode='set',
        mask_config=dict(
            mask_list_file=f'{mask_root}/mask_0.5-0.6_list.txt',
            prefix=mask_root,
            io_backend='disk',
            color_type='unchanged',
            flag='unchanged',
            file_client_kwargs=dict())),
    dict(
        type='RandomResizedCrop',
        keys=['gt'],
        crop_size=(512, 512),
    ),
    dict(type='GetMaskedImage'),
    dict(type='PackEditInputs'),
]

train_dataloader = dict(
    batch_size=4,
    sampler=dict(shuffle=False),
    dataset=dict(pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(pipeline=test_pipeline),
)

test_dataloader = val_dataloader

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=500002,
    val_interval=50000,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.0, 0.9))),
    disc=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.0, 0.9))))

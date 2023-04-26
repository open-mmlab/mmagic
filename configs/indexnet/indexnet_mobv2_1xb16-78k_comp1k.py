_base_ = [
    '../_base_/datasets/comp1k.py', '../_base_/matting_default_runtime.py'
]

experiment_name = 'indexnet_mobv2_1xb16-78k_comp1k'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    type='IndexNet',
    data_preprocessor=dict(
        type='MattorPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        proc_trimap='rescale_to_zero_one',
    ),
    backbone=dict(
        type='SimpleEncoderDecoder',
        encoder=dict(
            type='IndexNetEncoder',
            in_channels=4,
            freeze_bn=True,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://mmedit/mobilenet_v2')),
        decoder=dict(type='IndexNetDecoder')),
    loss_alpha=dict(type='CharbonnierLoss', loss_weight=0.5, sample_wise=True),
    loss_comp=dict(
        type='CharbonnierCompLoss', loss_weight=1.5, sample_wise=True),
    test_cfg=dict(
        resize_method='interp',
        resize_mode='bicubic',
        size_divisor=32,
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', color_type='grayscale'),
    dict(type='LoadImageFromFile', key='fg'),
    dict(type='LoadImageFromFile', key='bg'),
    dict(type='LoadImageFromFile', key='merged'),
    dict(type='GenerateTrimapWithDistTransform', dist_thr=20),
    dict(
        type='CropAroundUnknown',
        keys=['alpha', 'merged', 'fg', 'bg', 'trimap'],
        crop_sizes=[320, 480, 640],
        interpolations=['bicubic', 'bicubic', 'bicubic', 'bicubic',
                        'nearest']),
    dict(
        type='Resize',
        keys=['trimap'],
        scale=(320, 320),
        keep_ratio=False,
        interpolation='nearest'),
    dict(
        type='Resize',
        keys=['alpha', 'merged', 'fg', 'bg'],
        scale=(320, 320),
        keep_ratio=False,
        interpolation='bicubic'),
    dict(type='Flip', keys=['alpha', 'merged', 'fg', 'bg', 'trimap']),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='alpha',
        color_type='grayscale',
        save_original_img=True),
    dict(
        type='LoadImageFromFile',
        key='trimap',
        color_type='grayscale',
        save_original_img=True),
    dict(type='LoadImageFromFile', key='merged'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(pipeline=test_pipeline),
)

test_dataloader = val_dataloader

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=78000,
    val_interval=2600,
)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-2),
    paramwise_cfg=dict(custom_keys={'encoder.layers': dict(lr_mult=0.01)}),
)
# learning policy
param_scheduler = dict(
    type='MultiStepLR',
    milestones=[52000, 67600],
    gamma=0.1,
    by_epoch=False,
)

# checkpoint saving
default_hooks = dict(checkpoint=dict(interval=2600, out_dir=save_dir))

# runtime settings
# inheritate from _base_
